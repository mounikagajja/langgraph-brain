"""Clean raw ingestion data: strip bot-detection, signatures, normalize whitespace."""

from __future__ import annotations

import re

# GitHub login patterns that are clearly automation
BOT_LOGIN_PATTERNS = (
    "[bot]",
    "dependabot",
    "github-actions",
    "pre-commit-ci",
    "renovate",
    "codecov",
    "vercel",
    "netlify",
    "sonarcloud",
    "allcontributors",
)

# Common email signature / footer markers
SIGNATURE_MARKERS = (
    "\n-- \n",
    "\nSent from my iPhone",
    "\nSent from my iPad",
    "\nGet Outlook for ",
    "\n\nThanks,\n",
    "\n\nThank you,\n",
    "\n\nBest,\n",
    "\n\nRegards,\n",
    "\n\nBest regards,\n",
    "\n\nCheers,\n",
)


def is_bot_login(login: str | None) -> bool:
    """True if a GitHub login matches a known automation account pattern."""
    if not login:
        return False
    lower = login.lower()
    return any(p in lower for p in BOT_LOGIN_PATTERNS)


def strip_signature(text: str) -> str:
    """Cut off email-style signatures."""
    for marker in SIGNATURE_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text


def strip_quoted_replies(text: str) -> str:
    """Remove email-style quoted reply blocks (lines starting with >)."""
    lines = text.splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith(">")]
    return "\n".join(kept)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines and trim trailing space per line."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_body(text: str | None) -> str:
    """Apply the full cleaning pipeline to a single body string."""
    if not text:
        return ""
    text = strip_signature(text)
    text = strip_quoted_replies(text)
    text = normalize_whitespace(text)
    return text