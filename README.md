# grokking-autonomous-agents

## Task planner

The `TaskPlannar` agent turns a task description into an ordered action plan
using a Reflexion loop and specialist agents through the OpenAI API.

1. Install dependencies with `pip install -r requirements.txt`.
2. Create a `.env` file in the repository root with your OpenAI API key and,
   optionally, a different OpenAI model:

   ```dotenv
   OPENAI_API_KEY=your_api_key
   OPENAI_MODEL=gpt-4o-mini
   ```

3. Run the planner with a task as an argument, or run it without arguments to
   enter a task interactively:

   ```powershell
   python TaskPlannar/main.py "Plan a small website launch"
   python TaskPlannar/main.py
   ```

Plans are limited to five concise subtasks. The planner prefers schema-constrained
JSON Schema output supported by OpenAI; local validation enforces plan limits.
Set `LOG_LEVEL=DEBUG` in `.env` for detailed request and fallback diagnostics;
the default is `INFO`. Logs omit prompts, email bodies, and credentials.

For each draft, the planner evaluates every subtask for relevance, actionability,
feasibility, and agent fit. Its heuristic score weights those criteria 40%,
25%, 20%, and 15%. Subtasks below 7/10, or with relevance or agent fit below
6/10, are reflected on and replanned; the planner makes at most three drafts.
Each step is assigned to and processed by a specialist agent, then its results
are synthesized into a direct response to the original request. To send email
through Gmail, enable the Gmail API in a Google Cloud project, create OAuth
credentials for a Desktop app, and save the downloaded file as `credentials.json`
in the repository root. On first send, a browser opens for account authorization;
the resulting `token.json` is cached locally and ignored by Git.

```powershell
python TaskPlannar/main.py --send-email --to manager@example.com "Write and send a reply to my manager"
```

The app displays the composed message and sends only after you type `YES`.
Without the send option, no email is sent. Gmail sends from the account you
authorize in the browser.