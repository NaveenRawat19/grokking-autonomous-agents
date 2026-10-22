"""Create an actionable plan and response using OpenAI models."""

import argparse
import base64
import json
import logging
import os
import sys
from email.message import EmailMessage
from textwrap import dedent

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI


logger = logging.getLogger(__name__)


AGENTS = {
	"coordinator": {
		"name": "Planning Coordinator",
		"focus": "decompose goals, sequence work, and resolve dependencies",
	},
	"researcher": {
		"name": "Research Analyst",
		"focus": "analyze provided information, identify unknowns, and propose verifiable research",
	},
	"builder": {
		"name": "Implementation Agent",
		"focus": "create or modify concrete deliverables with available tools",
	},
	"communicator": {
		"name": "Communications Agent",
		"focus": "draft and refine messages, documents, and presentations",
	},
	"reviewer": {
		"name": "Quality Reviewer",
		"focus": "check deliverables against requirements, risks, and completion criteria",
	},
	"email_sender": {
		"name": "Email Delivery Agent",
		"focus": "send an explicitly approved email through the authenticated Gmail account",
	},
}


PLAN_PROMPT = dedent(
	"""\
	Create a concise plan with no more than five necessary subtasks. For simple
	requests, use one to three. Each subtask needs a short purpose (thought), one
	concrete action, an observable completion criterion (observation), and an agent
	id from the roster. Keep each field to one short sentence. Do not add prose
	outside the JSON object, repeat the request, or include extended reasoning.
	Delegate each action to the agent whose focus best matches it. Respect the
	user's constraints, state assumptions, and do not claim actions are done.

	Available agents:
	{agent_roster}

	User request:
	{user_input}

	Reflexion memory from earlier drafts:
	{reflection_memory}

	Email delivery requirement:
	{email_delivery_instruction}
	"""
)


EVALUATION_PROMPT = dedent(
	"""\
	You are the evaluator in a Reflexion planning loop. Evaluate EVERY subtask
	individually against the user's goal. Score each criterion from 0 to 10:
	- relevance: directly advances the goal or satisfies an explicit constraint
	- actionability: concrete, clear, and verifiable rather than vague
	- feasibility: reasonable and correctly sequenced given available context
	- agent_fit: the assigned specialist is appropriate for the action

	Penalize unrelated, redundant, speculative, or unsupported actions. Do not
	assume external facts have been checked. Give concise, specific feedback that
	will help the next draft fix weak actions. Also provide an overall reflection
	identifying omissions, unnecessary work, ordering problems, and assumptions.
	Return exactly one indexed evaluation for every subtask, in order.

	User request:
	{user_input}

	Candidate subtasks:
	{steps}
	"""
)


AGENT_TASK_PROMPT = dedent(
	"""\
	You are {agent_name}. Your specialty is: {agent_focus}.
	Complete the assigned subtask using only the provided request and context.
	You have no direct browser, filesystem, or email tools. Email delivery, when
	requested, is handled separately by the email_sender agent after confirmation.
	You may draft, analyze, and reason from supplied information, but do not claim
	to have performed external or real-world actions. If an action needs such a
	tool, return any useful draft or analysis and mark the subtask as blocked with
	the exact remaining action.

	Original user request:
	{user_input}

	Assigned subtask:
	{subtask}

	Additional instructions:
	{agent_instructions}

	Earlier subtask results, if useful:
	{previous_results}
	"""
)


FINAL_RESPONSE_PROMPT = dedent(
	"""\
	Respond directly to the user's original request using the specialist results.
	Provide the actual requested deliverable when the results contain one, rather
	than returning only a plan. Be concise but complete. Do not claim external
	actions were performed. Clearly state any blocked action and what remains to be
	done. Do not invent facts or omit useful limitations.

	Original user request:
	{user_input}

	Specialist results:
	{results}
	"""
)


MAX_PLAN_STEPS = 5

PLAN_SCHEMA = {
	"type": "object",
	"properties": {
		"steps": {
			"type": "array",
			"minItems": 1,
			"maxItems": MAX_PLAN_STEPS,
			"items": {
				"type": "object",
				"properties": {
					"thought": {"type": "string", "maxLength": 120},
					"action": {"type": "string", "maxLength": 200},
					"observation": {"type": "string", "maxLength": 160},
					"agent": {"type": "string", "enum": list(AGENTS)},
				},
				"required": ["thought", "action", "observation", "agent"],
				"additionalProperties": False,
			},
		},
	},
	"required": ["steps"],
	"additionalProperties": False,
}

EVALUATION_SCHEMA = {
	"type": "object",
	"properties": {
		"steps": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"index": {"type": "integer", "minimum": 1},
					"relevance": {"type": "number", "minimum": 0, "maximum": 10},
					"actionability": {"type": "number", "minimum": 0, "maximum": 10},
					"feasibility": {"type": "number", "minimum": 0, "maximum": 10},
					"agent_fit": {"type": "number", "minimum": 0, "maximum": 10},
					"feedback": {"type": "string"},
				},
				"required": [
					"index", "relevance", "actionability", "feasibility", "agent_fit", "feedback"
				],
				"additionalProperties": False,
			},
		},
		"reflection": {"type": "string"},
	},
	"required": ["steps", "reflection"],
	"additionalProperties": False,
}

AGENT_RESULT_SCHEMA = {
	"type": "object",
	"properties": {
		"status": {"type": "string", "enum": ["completed", "blocked"]},
		"result": {"type": "string"},
		"remaining_action": {"type": "string"},
	},
	"required": ["status", "result", "remaining_action"],
	"additionalProperties": False,
}

MAX_PLAN_ATTEMPTS = 3
MIN_HEURISTIC_SCORE = 7.0
MIN_RELEVANCE_SCORE = 6.0
MIN_AGENT_FIT_SCORE = 6.0


def configure_logging() -> None:
	"""Configure console logging from LOG_LEVEL after environment loading."""
	level_name = os.getenv("LOG_LEVEL", "INFO").upper()
	level = getattr(logging, level_name, logging.INFO)
	logging.basicConfig(
		level=level,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
	logger.setLevel(level)


def openai_response_text(response) -> str:
	"""Extract text from an OpenAI chat-completion response."""
	if not response.choices:
		return ""
	content = response.choices[0].message.content or ""
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		return "\n".join(
			block.text
			for block in content
			if getattr(block, "type", None) == "text" and getattr(block, "text", None)
		)
	return ""


def request_structured_output(
	client,
	model: str,
	prompt: str,
	schema: dict,
	token_limit: int,
	stage: str = "structured output",
) -> dict:
	"""Request schema-constrained JSON from OpenAI and validate the response."""
	json_prompt = (
		prompt
		+ "\n\nGenerate a JSON object matching this schema:\n"
		+ json.dumps(schema, ensure_ascii=False)
	)
	response = None
	for attempt in range(2):
		logger.debug(
			"OpenAI request started stage=%s model=%s attempt=%d max_tokens=%d",
			stage,
			model,
			attempt + 1,
			token_limit,
		)
		response = client.chat.completions.create(
			model=model,
			messages=[{"role": "user", "content": json_prompt}],
			response_format={
				"type": "json_schema",
				"json_schema": {"name": "structured_response", "strict": True, "schema": schema},
			},
			temperature=0.2,
			max_tokens=token_limit,
		)
		if not response.choices:
			logger.error("OpenAI returned no choices stage=%s", stage)
			raise RuntimeError("OpenAI returned no response choices.")
		choice = response.choices[0]
		finish_reason = str(getattr(choice, "finish_reason", "")).lower()
		usage = getattr(response, "usage", None)
		logger.info(
			"OpenAI response received stage=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s",
			stage,
			finish_reason or "unknown",
			getattr(usage, "prompt_tokens", "unknown"),
			getattr(usage, "completion_tokens", "unknown"),
		)
		if finish_reason not in ("length", "max_tokens"):
			break
		if attempt == 0:
			token_limit *= 2
			logger.warning(
				"OpenAI output truncated; retrying stage=%s with max_tokens=%d",
				stage,
				token_limit,
			)
			continue
		raise RuntimeError(
			"OpenAI stopped before completing structured output after retry "
			f"(stage={stage}, finish_reason={finish_reason}, max_tokens={token_limit})."
		)

	message = response.choices[0].message
	if getattr(message, "refusal", None):
		logger.warning("OpenAI refused structured request stage=%s", stage)
		raise RuntimeError("OpenAI model refused the structured-output request.")
	response_text = openai_response_text(response).strip()
	if response_text.startswith("```"):
		response_text = response_text.split("\n", 1)[-1]
		if response_text.endswith("```"):
			response_text = response_text[:-3].strip()
	try:
		result = json.loads(response_text)
	except (TypeError, json.JSONDecodeError) as error:
		preview = repr(response_text)[:240]
		logger.error("OpenAI returned invalid JSON stage=%s", stage)
		raise RuntimeError(
			f"OpenAI returned invalid structured JSON (stage={stage}); response preview: {preview}"
		) from error
	if not isinstance(result, dict):
		logger.error("OpenAI returned non-object JSON stage=%s", stage)
		raise RuntimeError("OpenAI returned an invalid structured result.")
	logger.debug("Structured response parsed stage=%s", stage)
	return result


def validate_plan(plan: dict) -> dict:
	"""Validate that a plan contains complete, non-empty subtasks."""
	steps = plan.get("steps") if isinstance(plan, dict) else None
	if not isinstance(steps, list) or not steps:
		raise RuntimeError("The generated plan must contain at least one subtask.")
	if len(steps) > MAX_PLAN_STEPS:
		raise RuntimeError(f"The generated plan exceeds the {MAX_PLAN_STEPS}-subtask limit.")
	for index, step in enumerate(steps, start=1):
		if not isinstance(step, dict):
			raise RuntimeError(f"Subtask {index} must be an object.")
		for field in ("thought", "action", "observation"):
			value = step.get(field)
			if not isinstance(value, str) or not value.strip():
				raise RuntimeError(f"Subtask {index} is missing a non-empty {field}.")
		if step.get("agent") not in AGENTS:
			raise RuntimeError(f"Subtask {index} has no valid delegated agent.")
	return plan


def ensure_email_delivery_step(plan: dict) -> dict:
	"""Ensure requested delivery is the final step, after an email draft exists."""
	steps = [step for step in plan["steps"] if step["agent"] != "email_sender"]
	draft_step = next((step for step in steps if step.get("email_draft")), None)
	if draft_step is None:
		draft_step = next(
			(
				step
				for step in steps
				if step["agent"] == "communicator"
				and "email" in step["action"].lower()
				and any(word in step["action"].lower() for word in ("draft", "write", "compose"))
			),
			None,
		)
	if draft_step is None:
		steps.append(
			{
				"thought": "Prepare the message before delivery.",
				"action": "Draft the requested email with a clear subject and body.",
				"observation": "A complete email draft is available for review.",
				"agent": "communicator",
				"email_draft": True,
			}
		)
	else:
		draft_step["email_draft"] = True
	steps.append(
		{
			"thought": "Deliver the reviewed message only after the user approves it.",
			"action": "Send the email to the specified recipient after explicit confirmation.",
			"observation": "Gmail accepts the message or a delivery failure is reported.",
			"agent": "email_sender",
		}
	)
	plan["steps"] = steps
	return plan


def score_subtask(evaluation: dict) -> float:
	"""Combine evaluator criteria into a weighted 0-10 heuristic score."""
	return round(
		evaluation["relevance"] * 0.4
		+ evaluation["actionability"] * 0.25
		+ evaluation["feasibility"] * 0.2
		+ evaluation["agent_fit"] * 0.15,
		2,
	)


def evaluate_plan(client, model: str, user_input: str, plan: dict) -> dict:
	"""Score every subtask and validate evaluator output before using it."""
	evaluation = request_structured_output(
		client,
		model,
		EVALUATION_PROMPT.format(
			user_input=user_input,
			steps=json.dumps(plan["steps"], ensure_ascii=False),
		),
		EVALUATION_SCHEMA,
		1600,
		stage="plan evaluation",
	)
	reviews = evaluation.get("steps")
	if not isinstance(reviews, list) or len(reviews) != len(plan["steps"]):
		raise RuntimeError("The evaluator must score every subtask exactly once.")
	if not isinstance(evaluation.get("reflection"), str):
		raise RuntimeError("The evaluator returned invalid reflection feedback.")

	for index, review in enumerate(reviews, start=1):
		if not isinstance(review, dict) or review.get("index") != index:
			raise RuntimeError("The evaluator returned subtasks out of order or with missing indexes.")
		for criterion in ("relevance", "actionability", "feasibility", "agent_fit"):
			score = review.get(criterion)
			if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 10:
				raise RuntimeError(f"The evaluator returned an invalid {criterion} score.")
		if not isinstance(review.get("feedback"), str):
			raise RuntimeError("The evaluator returned invalid subtask feedback.")
		review["heuristic_score"] = score_subtask(review)
		logger.info(
			"Plan score step=%d agent=%s total=%.2f relevance=%.1f actionability=%.1f feasibility=%.1f agent_fit=%.1f",
			index,
			plan["steps"][index - 1]["agent"],
			review["heuristic_score"],
			review["relevance"],
			review["actionability"],
			review["feasibility"],
			review["agent_fit"],
		)
	return evaluation


def parse_email_draft(draft: str) -> tuple[str, str]:
	"""Extract a subject and body from the communicator's email draft."""
	lines = draft.strip().splitlines()
	if lines and lines[0].startswith("```html"):
		lines = lines[1:]
	if lines and lines[0].startswith("```"):
		lines = lines[1:]
	if lines and lines[-1].strip() == "```":
		lines = lines[:-1]
	subject_index = next(
		(index for index, line in enumerate(lines) if line.lower().startswith("subject:")),
		None,
	)
	if subject_index is None:
		raise ValueError("Email draft must start with a 'Subject:' line.")
	subject = lines[subject_index].split(":", 1)[1].strip()
	body = "\n".join(lines[subject_index + 1:]).strip()
	if not subject or not body:
		raise ValueError("Email draft must include both a subject and a body.")
	return subject, body


def send_email_via_gmail(recipient: str, subject: str, body: str) -> str:
	"""Send an approved email with the Gmail API and return its message id."""
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	credentials_path = os.getenv(
		"GMAIL_CREDENTIALS_FILE", os.path.join(project_root, "credentials.json")
	)
	token_path = os.getenv("GMAIL_TOKEN_FILE", os.path.join(project_root, "token.json"))
	scopes = ["https://www.googleapis.com/auth/gmail.send"]

	credentials = None
	if os.path.exists(token_path):
		credentials = Credentials.from_authorized_user_file(token_path, scopes)
	if credentials and credentials.expired and credentials.refresh_token:
		logger.info("Refreshing Gmail OAuth credentials")
		credentials.refresh(Request())
	if not credentials or not credentials.valid:
		if not os.path.exists(credentials_path):
			raise ValueError(
				"Download Gmail Desktop OAuth credentials as credentials.json "
				"or set GMAIL_CREDENTIALS_FILE."
			)
		logger.info("Starting Gmail OAuth authorization")
		flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
		credentials = flow.run_local_server(port=0)
	else:
		logger.debug("Loaded cached Gmail OAuth credentials")

	with open(token_path, "w", encoding="utf-8") as token_file:
		token_file.write(credentials.to_json())

	message = EmailMessage()
	message["To"] = recipient
	message["Subject"] = subject
	message.set_content(body)
	raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
	logger.info("Submitting approved email through Gmail API")
	service = build("gmail", "v1", credentials=credentials)
	result = service.users().messages().send(
		userId="me", body={"raw": raw_message}
	).execute()
	message_id = result.get("id")
	if not message_id:
		logger.error("Gmail API returned no message id")
		raise RuntimeError("Gmail accepted no message id for the send request.")
	logger.info("Gmail API accepted message id=%s", message_id)
	return message_id


def execute_subtasks(
	client,
	model: str,
	user_input: str,
	plan: dict,
	send_email: bool = False,
	recipient: str | None = None,
) -> list:
	"""Delegate accepted subtasks to their specialist prompts in plan order."""
	results = []
	for index, step in enumerate(plan["steps"], start=1):
		agent = AGENTS[step["agent"]]
		logger.info("Starting subtask step=%d agent=%s", index, agent["name"])
		if step["agent"] == "email_sender":
			draft_result = None
			subject = body = None
			for candidate in reversed(results):
				if not candidate.get("email_draft"):
					continue
				try:
					subject, body = parse_email_draft(candidate["result"])
				except ValueError:
					logger.warning(
						"Email draft step did not produce a parseable draft status=%s",
						candidate["status"],
					)
					continue
				draft_result = candidate
				break
			if not draft_result:
				failed_draft = next(
					(result for result in reversed(results) if result.get("email_draft")),
					None,
				)
				logger.error("Email delivery blocked because no parseable email draft is available")
				print(
					"Email not sent: no valid draft was produced, so confirmation was skipped.",
					file=sys.stderr,
				)
				results.append(
					{
						"step": index,
						"agent": agent["name"],
						"agent_id": step["agent"],
						"action": step["action"],
						"status": "blocked",
						"result": "No email was sent because the email draft step did not complete.",
						"remaining_action": (
							failed_draft["remaining_action"]
							if failed_draft and failed_draft.get("remaining_action")
							else "Complete and review the email draft before retrying delivery."
						),
					}
				)
				continue
			if not send_email or not recipient:
				logger.info("Email delivery skipped; send was not authorized")
				results.append(
					{
						"step": index,
						"agent": agent["name"],
						"agent_id": step["agent"],
						"action": step["action"],
						"status": "blocked",
						"result": "Email draft is ready but was not sent.",
						"remaining_action": "Rerun with --send-email --to RECIPIENT and approve the send prompt.",
					}
				)
				continue
			print(f"\nEmail to: {recipient}\nSubject: {subject}\n\n{body}\n")
			logger.info("Awaiting explicit email confirmation")
			sys.stdout.flush()
			confirmation = input("Type YES to send this email: ").strip().upper()
			if confirmation != "YES":
				logger.info("Email delivery declined by user")
				results.append(
					{
						"step": index,
						"agent": agent["name"],
						"agent_id": step["agent"],
						"action": step["action"],
						"status": "blocked",
						"result": "Email draft is ready; user declined delivery.",
						"remaining_action": "No email was sent.",
					}
				)
				continue
			try:
				message_id = send_email_via_gmail(recipient, subject, body)
			except Exception as error:
				logger.exception("Gmail delivery failed")
				results.append(
					{
						"step": index,
						"agent": agent["name"],
						"agent_id": step["agent"],
						"action": step["action"],
						"status": "blocked",
						"result": "Email draft is ready, but Gmail delivery failed.",
						"remaining_action": f"Check Gmail OAuth/API setup and retry ({type(error).__name__}).",
					}
				)
				continue
			results.append(
				{
					"step": index,
					"agent": agent["name"],
					"agent_id": step["agent"],
					"action": step["action"],
					"status": "completed",
					"result": f"Email sent through Gmail to {recipient} (message id: {message_id}).",
					"remaining_action": "",
				}
			)
			logger.info("Email delivery completed step=%d", index)
			continue
		previous_results = [
			{
				"step": result["step"],
				"agent": result["agent"],
				"status": result["status"],
				"result": result["result"][:2000],
			}
			for result in results
		]
		result = request_structured_output(
			client,
			model,
			AGENT_TASK_PROMPT.format(
				agent_name=agent["name"],
				agent_focus=agent["focus"],
				user_input=user_input,
				subtask=json.dumps(step, ensure_ascii=False),
				previous_results=json.dumps(previous_results, ensure_ascii=False),
				agent_instructions=(
					"You must produce an email draft even if details are missing. Use a "
					"reasonable, neutral assumption or a clear placeholder rather than "
					"blocking. Begin with 'Subject: ...', a blank line, and the email body."
					if send_email and step.get("email_draft")
					else "Follow the assigned subtask and required output format."
				),
			),
			AGENT_RESULT_SCHEMA,
			1200,
			stage=f"specialist {agent['name']}",
		)
		if result.get("status") not in ("completed", "blocked"):
			raise RuntimeError(f"Specialist {agent['name']} returned an invalid status.")
		if not isinstance(result.get("result"), str) or not result["result"].strip():
			raise RuntimeError(f"Specialist {agent['name']} returned an empty result.")
		if not isinstance(result.get("remaining_action"), str):
			raise RuntimeError(f"Specialist {agent['name']} returned an invalid remaining action.")
		if result["status"] == "blocked" and not result["remaining_action"].strip():
			raise RuntimeError(f"Specialist {agent['name']} did not explain the blocked action.")
		results.append(
			{
				"step": index,
				"agent": agent["name"],
				"agent_id": step["agent"],
				"email_draft": bool(step.get("email_draft")),
				"action": step["action"],
				**result,
			}
		)
		logger.info("Finished subtask step=%d agent=%s status=%s", index, agent["name"], result["status"])
	return results


def synthesize_response(client, model: str, user_input: str, results: list) -> str:
	"""Turn specialist work into a direct response to the user's request."""
	logger.info("Starting final response synthesis specialist_results=%d", len(results))
	response = client.chat.completions.create(
		model=model,
		messages=[{
			"role": "user",
			"content": FINAL_RESPONSE_PROMPT.format(
				user_input=user_input,
				results=json.dumps(results, ensure_ascii=False),
			),
		}],
		temperature=0.3,
		max_tokens=1600,
	)
	content = openai_response_text(response).strip()
	if not content or not content.strip():
		logger.error("OpenAI returned an empty final response")
		raise RuntimeError("OpenAI returned an empty final response.")
	logger.info("Final response synthesis completed")
	return content


def create_plan(user_input: str, send_email: bool = False, recipient: str | None = None) -> str:
	"""Send a user request to the configured OpenAI model."""
	if not user_input.strip():
		raise ValueError("The task description cannot be empty.")

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("Set OPENAI_API_KEY in the environment or .env file.")
	if send_email and not recipient:
		raise ValueError("Provide --to RECIPIENT when using --send-email.")

	model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	logger.info("Starting task workflow model=%s email_delivery=%s", model, send_email)
	client = OpenAI(api_key=api_key, timeout=60, max_retries=0)
	reflections = []
	for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
		logger.info("Starting plan draft attempt=%d/%d", attempt, MAX_PLAN_ATTEMPTS)
		prompt = PLAN_PROMPT.format(
			user_input=user_input,
			reflection_memory=json.dumps(reflections, ensure_ascii=False),
			agent_roster=json.dumps(AGENTS, ensure_ascii=False),
			email_delivery_instruction=(
				f"Delivery is requested. Include an email_sender step last; the recipient is {recipient}."
				if send_email
				else "No email delivery was authorized. Do not send an email."
			),
		)
		plan = validate_plan(
			request_structured_output(
				client, model, prompt, PLAN_SCHEMA, 1200, stage=f"plan draft {attempt}"
			)
		)
		if send_email:
			plan = ensure_email_delivery_step(plan)
		logger.info("Plan draft contains %d subtasks", len(plan["steps"]))
		evaluation = evaluate_plan(client, model, user_input, plan)
		weak_steps = []
		for step, review in zip(plan["steps"], evaluation["steps"]):
			if (
				review["heuristic_score"] < MIN_HEURISTIC_SCORE
				or review["relevance"] < MIN_RELEVANCE_SCORE
				or review["agent_fit"] < MIN_AGENT_FIT_SCORE
			):
				weak_steps.append(
					{
						"action": step["action"],
						"agent": step["agent"],
						"scores": {
							"relevance": review["relevance"],
							"actionability": review["actionability"],
							"feasibility": review["feasibility"],
							"agent_fit": review["agent_fit"],
							"weighted": review["heuristic_score"],
						},
						"feedback": review["feedback"],
					}
				)
		if not weak_steps:
			logger.info("Plan passed evaluation attempt=%d", attempt)
			results = execute_subtasks(
				client, model, user_input, plan, send_email=send_email, recipient=recipient
			)
			return synthesize_response(client, model, user_input, results)
		if attempt == MAX_PLAN_ATTEMPTS:
			logger.error("Plan exhausted all attempts with weak subtasks count=%d", len(weak_steps))
			weak_summary = "; ".join(
				f"'{step['action']}' ({step['agent']}) scored "
				f"{step['scores']['weighted']}/10: {step['feedback']}"
				for step in weak_steps
			)
			raise RuntimeError(
				f"Plan did not meet quality thresholds after {MAX_PLAN_ATTEMPTS} drafts. "
				+ weak_summary
			)
		reflections.append(
			{
				"attempt": attempt,
				"overall_reflection": evaluation["reflection"],
				"weak_subtasks": weak_steps,
			}
		)
		logger.warning(
			"Plan needs revision attempt=%d weak_subtasks=%d; reflection stored",
			attempt,
			len(weak_steps),
		)
	raise RuntimeError("Planning ended without an accepted plan.")


def main() -> int:
	load_dotenv()
	configure_logging()
	parser = argparse.ArgumentParser(description="Plan and complete a task using OpenAI agents.")
	parser.add_argument("--send-email", action="store_true", help="Allow sending a drafted email after confirmation")
	parser.add_argument("--to", dest="recipient", help="Email recipient address")
	parser.add_argument("task", nargs="*", help="Task description")
	args = parser.parse_args()
	if args.recipient and not args.send_email:
		parser.error("--to requires --send-email")
	if args.send_email and not args.recipient:
		parser.error("--send-email requires --to RECIPIENT")
	user_input = " ".join(args.task).strip()
	if not user_input:
		user_input = input("Describe the task to plan: ").strip()

	try:
		logger.info("CLI started task_length=%d email_delivery=%s", len(user_input), args.send_email)
		print(create_plan(user_input, send_email=args.send_email, recipient=args.recipient))
	except ValueError as error:
		logger.error("Invalid task configuration: %s", error)
		print(f"Error: {error}", file=sys.stderr)
		return 2
	except Exception as error:
		logger.exception("Task workflow failed")
		print(f"Unable to create a plan: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
