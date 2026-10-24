import re
from collections import Counter

from pypdf import PdfReader

EDGE_LINES = 4           # how many lines at the top and bottom of a page count as "edge"
MIN_PAGES_FOR_CLEANING = 4
REPEAT_RATIO = 0.5       # an edge line is junk if it shows up on at least half the pages


def _normalize(line: str) -> str:
    """Make '1/18' and '2/18' (or '- 2 -' and '- 3 -') compare as equal."""
    return re.sub(r"\d+", "#", " ".join(line.split())).lower()


def _remove_repeated_edges(pages: list[dict]) -> list[dict]:
    """Strip running headers/footers and page numbers that repeat across pages."""
    if len(pages) < MIN_PAGES_FOR_CLEANING:
        return pages

    split = []
    seen_on_pages = Counter()
    for p in pages:
        lines = [l for l in p["text"].splitlines() if l.strip()]
        split.append(lines)
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        for norm in {_normalize(l) for l in edges}:
            seen_on_pages[norm] += 1

    cutoff = max(3, int(len(pages) * REPEAT_RATIO))
    junk = {norm for norm, n in seen_on_pages.items() if n >= cutoff}

    cleaned = []
    for p, lines in zip(pages, split):
        kept = []
        for i, line in enumerate(lines):
            at_edge = i < EDGE_LINES or i >= len(lines) - EDGE_LINES
            if at_edge and _normalize(line) in junk:
                continue
            kept.append(line)
        cleaned.append({"page": p["page"], "text": "\n".join(kept)})
    return cleaned


def load_pdf(file) -> list[dict]:
    """Return a list of {"page": int, "text": str}. `file` can be a path or an uploaded file object."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})

    if not any(p["text"].strip() for p in pages):
        raise ValueError(
            "No extractable text found. This may be a scanned PDF (needs OCR).")
    return _remove_repeated_edges(pages)


def load_txt(file) -> list[dict]:
    """Treat a plain-text contract as a single page."""
    if isinstance(file, str):
        with open(file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = file.read().decode("utf-8")
    return [{"page": 1, "text": text}]
