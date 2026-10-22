"""Minimal Gmail read/reply MCP server.

Setup:
    pip install -r ..\requirements.txt

    1. Google Cloud Console -> create project -> enable Gmail API.
    2. OAuth consent screen -> add yourself as a test user.
    3. Credentials -> Create OAuth client ID -> "Desktop app" -> download as credentials.json
       (place it next to this file).
    4. Authorize once:   python gmail_mcp.py auth
    5. Run the server:   mcp dev gmail_mcp.py
"""
import base64
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.message import EmailMessage
from mcp.server.fastmcp import FastMCP

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send"
        ]
BASE = Path(__file__).parent
CREDS_FILE = BASE / "credentials.json"
TOKEN_FILE = BASE / "token.json"

mcp = FastMCP("gmail-reader")


def get_credentials(interactive: bool = False) -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    if creds and creds.valid and creds.has_scopes(SCOPES):
        return creds
    if creds and creds.expired and creds.refresh_token and creds.has_scopes(SCOPES):
        creds.refresh(Request())
        if creds.valid and creds.has_scopes(SCOPES):
            TOKEN_FILE.write_text(creds.to_json())
            return creds
    if not interactive:
        raise RuntimeError(
            "Gmail authorization is missing the required read/send scopes. "
            "Run `python gmail_mcp.py auth` to reauthorize."
        )
    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"Gmail OAuth credentials not found: {CREDS_FILE}")
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    if not creds.has_scopes(SCOPES):
        raise RuntimeError("Gmail OAuth did not grant all required scopes.")
    TOKEN_FILE.write_text(creds.to_json())
    return creds


def gmail():
    return build("gmail", "v1", credentials=get_credentials(), cache_discovery=False)


def _header(headers: list[dict], name: str) -> str:
    return next((h["value"] for h in headers if h["name"].lower() == name.lower()), "")


def _extract_body(payload: dict) -> str:
    """Walk MIME parts, prefer text/plain, fall back to the first text/html."""
    plain, html = [], []

    def walk(part: dict):
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(base64.urlsafe_b64decode(data).decode("utf-8", "replace"))
        elif data and mime == "text/html":
            html.append(base64.urlsafe_b64decode(data).decode("utf-8", "replace"))
        for sub in part.get("parts", []):
            walk(sub)

    walk(payload)
    return "\n".join(plain) or "\n".join(html)

ALLOWED_RECIPIENTS = {
    a.strip().lower()
    for a in os.environ.get("GMAIL_ALLOWED_RECIPIENTS", "").split(",")
    if a.strip()
}


@mcp.tool()
def send_reply(message_id: str, body: str, to: str) -> dict:
    """Send a reply to an existing email (same thread) to the given address.

    `to` must be on the server's allowlist. `body` is plain text.
    """
    if to.lower() not in ALLOWED_RECIPIENTS:
        raise ValueError(f"{to} is not an allowed recipient.")

    svc = gmail()
    orig = svc.users().messages().get(
        userId="me", id=message_id, format="metadata",
        metadataHeaders=["Subject", "Message-ID", "References"],
    ).execute()
    h = orig["payload"]["headers"]
    subject = _header(h, "Subject")
    msg_id = _header(h, "Message-ID")

    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    if msg_id:
        msg["In-Reply-To"] = msg_id
        msg["References"] = f"{_header(h, 'References')} {msg_id}".strip()
    msg.set_content(body)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    sent = svc.users().messages().send(
        userId="me", body={"raw": raw, "threadId": orig["threadId"]}
    ).execute()
    return {"sent_id": sent["id"], "to": to, "subject": msg["Subject"]}


@mcp.tool()
def search_emails(query: str = "in:inbox", max_results: int = 10) -> list[dict]:
    """Search Gmail using Gmail search syntax (e.g. 'from:boss is:unread newer_than:2d').

    Returns id, sender, subject, date and snippet for each match.
    """
    svc = gmail()
    resp = svc.users().messages().list(
        userId="me", q=query, maxResults=min(max_results, 50)
    ).execute()
    results = []
    for m in resp.get("messages", []):
        msg = svc.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        h = msg["payload"]["headers"]
        results.append({
            "id": msg["id"],
            "thread_id": msg["threadId"],
            "from": _header(h, "From"),
            "subject": _header(h, "Subject"),
            "date": _header(h, "Date"),
            "snippet": msg.get("snippet", ""),
        })
    return results


@mcp.tool()
def read_email(message_id: str) -> dict:
    """Read the full content of one email by its id (from search_emails)."""
    msg = gmail().users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()
    h = msg["payload"]["headers"]
    return {
        "id": msg["id"],
        "from": _header(h, "From"),
        "to": _header(h, "To"),
        "subject": _header(h, "Subject"),
        "date": _header(h, "Date"),
        "body": _extract_body(msg["payload"])[:20000],  # cap size for the LLM context
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        get_credentials(interactive=True)
        print("Authorized. token.json saved.")  # safe: not running as stdio server here
    else:
        mcp.run()  # stdio; never print() to stdout in this mode