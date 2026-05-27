"""Chunk cleaned content by source type.

- docs: split by markdown header (##, ###); if a section exceeds CHAR_BUDGET, split by paragraph
- issues/PRs: title+body = one chunk; each comment = one chunk
- code blocks (```...```) are kept intact, never split mid-block
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ingestion.cleaning import clean_body, is_bot_login

logger = logging.getLogger(__name__)

# 1200 tokens ≈ 4800 chars for English. We use a soft character budget.
CHAR_BUDGET = 4800

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")


@dataclass
class Chunk:
    chunk_id: str
    source_type: str  # 'issue' | 'pr' | 'doc'
    source_id: str
    source_url: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# Doc chunking

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_by_headers(md: str) -> list[tuple[str, str]]:
    """Return list of (header_path, section_text). Includes pre-header content as ('', ...)."""
    matches = list(HEADER_RE.finditer(md))
    if not matches:
        return [("", md)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        pre = md[: matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

    header_stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()

        # maintain parent header path
        header_stack = [h for h in header_stack if h[0] < level]
        header_stack.append((level, title))
        path = " > ".join(t for _, t in header_stack)

        if body:
            sections.append((path, body))
    return sections


def _split_long_section(text: str, budget: int = CHAR_BUDGET) -> list[str]:
    """Split a too-long section by paragraph, never breaking inside a code fence."""
    if len(text) <= budget:
        return [text]

    parts = re.split(r"(```[\s\S]*?```)", text)  # keep code blocks intact
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if len(buf) + len(part) + 2 <= budget:
            buf = (buf + "\n\n" + part).strip() if buf else part
        else:
            if buf:
                chunks.append(buf)
            if len(part) <= budget:
                buf = part
            else:
                # Hard-wrap super long part
                for i in range(0, len(part), budget):
                    chunks.append(part[i : i + budget])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def chunk_doc(path: Path) -> list[Chunk]:
    """Chunk one markdown doc file."""
    text = path.read_text(encoding="utf-8")

    # Pull source URL from our ingestion header if present
    source_url = ""
    title = path.stem
    m = re.match(r"<!--\s*source:\s*(\S+)\s*\n\s*title:\s*(.+?)\s*-->", text)
    if m:
        source_url = m.group(1).strip()
        title = m.group(2).strip()
        text = text[m.end() :].lstrip()

    sections = _split_by_headers(text)
    chunks: list[Chunk] = []
    for section_idx, (header_path, body) in enumerate(sections):
        for part_idx, piece in enumerate(_split_long_section(body)):
            piece = clean_body(piece)
            if not piece:
                continue
            chunk_id = f"doc-{path.stem}-{section_idx}-{part_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_type="doc",
                    source_id=path.stem,
                    source_url=source_url,
                    title=title,
                    text=piece,
                    metadata={
                        "section": header_path,
                        "file": str(path.relative_to(RAW_DIR)),
                    },
                )
            )
    return chunks


# Issue / PR chunking


def _comment_chunks(
    item: dict[str, Any], comments: list[dict[str, Any]], source_type: str
) -> list[Chunk]:
    out: list[Chunk] = []
    for c in comments:
        body = clean_body(c.get("body"))
        if not body:
            continue
        user = c.get("user") or {}
        login = user.get("login")
        out.append(
            Chunk(
                chunk_id=f"{source_type}-{item['number']}-comment-{c['id']}",
                source_type=source_type,
                source_id=str(item["number"]),
                source_url=c.get("html_url") or item.get("html_url", ""),
                title=item.get("title", ""),
                text=body,
                metadata={
                    "kind": "comment",
                    "author": login,
                    "is_bot": is_bot_login(login),
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                },
            )
        )
    return out


def chunk_issue(item: dict[str, Any]) -> list[Chunk]:
    user = item.get("user") or {}
    body = clean_body(item.get("body"))
    title = item.get("title", "")
    parent_text = f"# {title}\n\n{body}".strip() if body else f"# {title}"

    parent = Chunk(
        chunk_id=f"issue-{item['number']}-parent",
        source_type="issue",
        source_id=str(item["number"]),
        source_url=item.get("html_url", ""),
        title=title,
        text=parent_text,
        metadata={
            "kind": "parent",
            "state": item.get("state"),
            "labels": [lbl.get("name") for lbl in item.get("labels") or []],
            "author": user.get("login"),
            "is_bot": is_bot_login(user.get("login")),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "closed_at": item.get("closed_at"),
            "comments_count": item.get("comments", 0),
        },
    )
    chunks = [parent]
    chunks.extend(_comment_chunks(item, item.get("comments_data") or [], "issue"))
    return chunks


def chunk_pr(item: dict[str, Any]) -> list[Chunk]:
    user = item.get("user") or {}
    body = clean_body(item.get("body"))
    title = item.get("title", "")
    parent_text = f"# {title}\n\n{body}".strip() if body else f"# {title}"

    parent = Chunk(
        chunk_id=f"pr-{item['number']}-parent",
        source_type="pr",
        source_id=str(item["number"]),
        source_url=item.get("html_url", ""),
        title=title,
        text=parent_text,
        metadata={
            "kind": "parent",
            "state": "merged" if item.get("merged_at") else item.get("state"),
            "author": user.get("login"),
            "is_bot": is_bot_login(user.get("login")),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "merged_at": item.get("merged_at"),
            "labels": [lbl.get("name") for lbl in item.get("labels") or []],
        },
    )
    chunks = [parent]
    chunks.extend(_comment_chunks(item, item.get("issue_comments_data") or [], "pr"))
    chunks.extend(_comment_chunks(item, item.get("review_comments_data") or [], "pr"))
    return chunks


# Driver


def chunk_all() -> int:
    """Read all raw data, produce chunks.jsonl. Returns total chunks written."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        # Issues
        for jf in sorted((RAW_DIR / "issues").glob("*.json")):
            item = json.loads(jf.read_text(encoding="utf-8"))
            for c in chunk_issue(item):
                out.write(c.to_json_line() + "\n")
                total += 1

        # PRs
        for jf in sorted((RAW_DIR / "prs").glob("*.json")):
            item = json.loads(jf.read_text(encoding="utf-8"))
            for c in chunk_pr(item):
                out.write(c.to_json_line() + "\n")
                total += 1

        # Docs
        for mf in sorted((RAW_DIR / "docs").glob("*.md")):
            if mf.name == "_llms_index.md":
                continue
            for c in chunk_doc(mf):
                out.write(c.to_json_line() + "\n")
                total += 1

    logger.info("Wrote %d chunks to %s", total, OUT_PATH)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    chunk_all()
