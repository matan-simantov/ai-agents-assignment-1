# Loads the background files (LinkedIn + summary) and holds the model settings.
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)
ME_DIR = BASE_DIR / "me"
LOGS_DIR = BASE_DIR / "logs"

NAME = "Daniel Mor"  # sample (fictional) profile

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ACTOR_MODEL = os.getenv("ACTOR_MODEL") or "gemini-3.5-flash"
CRITIC_MODEL = os.getenv("CRITIC_MODEL") or "gemini-3.5-flash"

MAX_ATTEMPTS = 3  # generation attempts per question, including the first one


def load_background():
    linkedin = (ME_DIR / "linkedin.txt").read_text(encoding="utf-8")
    summary = (ME_DIR / "summary.txt").read_text(encoding="utf-8")
    return linkedin, summary


LINKEDIN, SUMMARY = load_background()


def background_text():
    return f"## LinkedIn profile:\n{LINKEDIN}\n\n## Personal summary:\n{SUMMARY}"


def history_as_text(history, last_n=6):
    """Turns the chat history (list of role/content dicts) into plain text for the critic."""
    if not history:
        return "(no previous messages)"
    lines = []
    for msg in history[-last_n:]:
        who = "User" if msg["role"] == "user" else NAME
        lines.append(f"{who}: {msg['content']}")
    return "\n".join(lines)
