"""
E1: text-heavy / semi-structured prose.

Chunks by structure (numbered headings), not fixed token windows, so each
chunk is a coherent unit an LLM can extract semantics from independently.
Heading detection is regex-based here (deterministic, cheap); swap in a
layout-model heading classifier if your documents don't follow numbered
sections.
"""
import re

# Matches "1", "1.1", "1.1.1", "Appendix A." etc. at line start.
HEADING_RE = re.compile(
    r"^(?:(\d+(?:\.\d+)*)\s+|Appendix\s+([A-Z])\.\s+)(.{3,100})$"
)

# A line that is ONLY a heading number (no title text on the same line).
# pymupdf's text extraction frequently splits "1.1 Purpose & Background"
# into two separate lines -- "1.1" then "Purpose & Background" -- when the
# PDF was laid out with the number and title as distinct text runs. Detected
# here and stitched back together before heading matching runs.
LONE_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)$")
LONE_APPENDIX_RE = re.compile(r"^Appendix\s+([A-Z])\.?$")


def _merge_split_headings(lines: list[str]) -> list[str]:
    merged = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m_num = LONE_NUMBER_RE.match(stripped)
        m_app = LONE_APPENDIX_RE.match(stripped)
        if (m_num or m_app) and i + 1 < len(lines) and lines[i + 1].strip():
            prefix = stripped if m_num else f"Appendix {m_app.group(1)}."
            merged.append(f"{prefix} {lines[i + 1].strip()}")
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return merged


def chunk_by_heading(text: str) -> list[dict]:
    lines = _merge_split_headings(text.split("\n"))
    chunks = []
    current = {"heading": None, "heading_number": None, "body": []}

    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            if current["body"] or current["heading"]:
                chunks.append(_finalize(current))
            number = m.group(1) or m.group(2)
            title = m.group(3).strip()
            current = {"heading": title, "heading_number": number, "body": []}
        else:
            current["body"].append(line)

    if current["body"] or current["heading"]:
        chunks.append(_finalize(current))

    return chunks


def _finalize(chunk: dict) -> dict:
    body_text = "\n".join(chunk["body"]).strip()
    return {
        "heading_number": chunk["heading_number"],
        "heading": chunk["heading"],
        "text": body_text,
        "char_count": len(body_text),
    }


def extract(page_text: str) -> dict:
    chunks = chunk_by_heading(page_text)
    if not chunks:
        chunks = [{"heading_number": None, "heading": None, "text": page_text.strip(),
                    "char_count": len(page_text.strip())}]
    return {
        "chunks": chunks,
        "full_text": page_text.strip(),
    }

