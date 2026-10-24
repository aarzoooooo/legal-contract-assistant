import re
from dataclasses import dataclass

MAX_CHARS = 1200      # target max size of a chunk body
OVERLAP_MAX = 250     # carry the previous sentence into the next chunk only if it is short
LABEL_MAX = 120

# Numbered heading on its own line: "1. Definitions", "12.1 Termination", "ARTICLE IV", "Section 3: Payment"
HEADING_RE = re.compile(
    r"^\s*(?:(?:ARTICLE|Article|SECTION|Section)\s+[IVXLC\d]+[.:]?|\d{1,2}(?:\.\d{1,2})*\.?)\s+[A-Z][^\n]{0,80}$"
)
# Numbered heading and body on the same line: "3. Salary.  During your employment, ..."
INLINE_HEADING_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})*\.?)\s+([A-Z][^.:]{1,70}?[.:])\s+(?=[A-Z(\"“])"
)
# Lettered sub-clause, alone or inline: "(a) Equitable Remedies.  The Employee ..."
SUBCLAUSE_RE = re.compile(
    r"^(\((?:[a-z]{1,2}|[ivx]{1,6})\))\s+([A-Z][^.:]{1,70}?[.:])(?:\s+(?=[A-Z(\"“])|$)"
)
# A new document part: "Appendix B", "Exhibit A" (letters only, so table-of-contents lines like "Exhibit 3" are ignored)
PART_RE = re.compile(
    r"^(?:Appendix|APPENDIX|Annex|ANNEX|Schedule|SCHEDULE|Exhibit|EXHIBIT|Attachment|ATTACHMENT)\s+[A-Z]$"
)

SMALL_WORDS = {"and", "of", "the", "to", "for", "in", "on",
               "a", "an", "or", "by", "with", "at", "from", "as"}
LIST_MARKER_RE = re.compile(
    r"^\(?[0-9A-Za-z]{1,3}[.)]$")   # a lone "4." or "(a)"

# Invisible page markers inserted while reading, used to recover each chunk's real page
PAGE_MARK_RE = re.compile(r"\[\[PAGE(\d+)\]\]")


@dataclass
class Chunk:
    id: int
    text: str       # section label + body (this is what gets embedded)
    section: str    # section label, shown in citations
    page: int       # page where this chunk starts


def looks_like_title(text: str) -> bool:
    """True for 'Annual Bonus.' or 'Entire Agreement; Amendment.', false for ordinary sentences."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
    if not words or len(words) > 9:
        return False
    significant = [w for w in words if w.lower() not in SMALL_WORDS]
    if not significant:
        return False
    capitalized = sum(1 for w in significant if w[0].isupper())
    return capitalized / len(significant) >= 0.7


def is_heading(line: str) -> bool:
    """A numbered heading standing alone on its line."""
    if not HEADING_RE.match(line):
        return False
    stripped = line.rstrip()
    if not stripped.endswith((".", ",", ";")):
        return True
    # Short title-like line that ends with a period, e.g. "4. Miscellaneous."
    parts = stripped.split(None, 1)
    return (
        len(parts) == 2
        and stripped.endswith(".")
        # needs a dotted number like "4." or "12.1"
        and "." in parts[0]
        and looks_like_title(parts[1])
    )


def split_heading(line: str) -> tuple[str, str]:
    """Split a heading-only line into a short label and any body text on the same line."""
    parts = line.split(None, 1)
    if len(parts) < 2:
        return line[:100], ""
    number, rest = parts
    for m in re.finditer(r"[.:](?:\s|$)", rest):
        if m.start() >= 4:                      # skip things like "IV." or "3:"
            label = f"{number} {rest[: m.start() + 1]}"
            return label[:100], rest[m.end():].strip()
    return line[:100], ""


def parse_heading(line: str):
    """Return (kind, label, remainder_of_line) if the line starts something new, else None.

    kind is "top" (numbered heading), "sub" (lettered sub-clause) or "part" (appendix/exhibit).
    """
    m = INLINE_HEADING_RE.match(line)
    if m and looks_like_title(m.group(2)):
        return "top", f"{m.group(1)} {m.group(2)}", line[m.end():].strip()

    m = SUBCLAUSE_RE.match(line)
    if m and looks_like_title(m.group(2)):
        return "sub", f"{m.group(1)} {m.group(2)}", line[m.end():].strip()

    if is_heading(line):
        label, remainder = split_heading(line)
        return "top", label, remainder

    if len(line) <= 30 and PART_RE.match(line):
        return "part", line, ""

    return None


def make_label(kind: str, label: str, part, parent) -> str:
    """Build the citation label, e.g. 'Appendix B > 7. Miscellaneous. > (a) Equitable Remedies.'"""
    if kind == "sub" and parent:
        label = f"{parent} > {label}"
    if part and kind != "part":
        label = f"{part} > {label}"
    return label[:LABEL_MAX]


def split_into_sections(pages: list[dict]) -> list[dict]:
    first_page = pages[0]["page"]
    sections = []
    part, parent = None, None
    current = {"title": "Preamble", "page": first_page,
               "cur_page": first_page, "lines": []}

    for p in pages:
        for raw in p["text"].splitlines():
            line = " ".join(raw.split())        # collapse runs of whitespace
            if not line:
                continue

            parsed = parse_heading(line)
            if parsed:
                kind, label, remainder = parsed
                if kind == "part":
                    part, parent = label, None
                elif kind == "top":
                    parent = label
                if current["lines"]:
                    sections.append(current)
                current = {
                    "title": make_label(kind, label, part, parent),
                    "page": p["page"],
                    "cur_page": p["page"],
                    "lines": [],
                }
                if remainder:
                    current["lines"].append(remainder)
            else:
                if current["cur_page"] != p["page"]:    # section continues onto a new page
                    current["lines"].append(f"[[PAGE{p['page']}]]")
                    current["cur_page"] = p["page"]
                current["lines"].append(line)

    if current["lines"]:
        sections.append(current)
    return sections


def hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort for a single very long sentence: split on word boundaries."""
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + len(word) + 1 > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def split_sentences(text: str) -> list[str]:
    """Split on '.' or ';', but keep a lone list marker like '4.' attached to the next sentence."""
    pieces, carry = [], ""
    for s in re.split(r"(?<=[.;])\s+", text):
        if LIST_MARKER_RE.match(s):
            carry += s + " "
            continue
        pieces.append(carry + s)
        carry = ""
    if carry.strip():
        pieces.append(carry.strip())
    return pieces


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on sentence boundaries, with a short sentence of overlap between chunks."""
    sentences = []
    for s in split_sentences(text):
        sentences.extend(hard_split(s, max_chars)
                         if len(s) > max_chars else [s])

    chunks, current, prev = [], "", ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current.strip())
            overlap = prev if len(prev) <= OVERLAP_MAX else ""
            current = f"{overlap} {s}"
        else:
            current += " " + s
        prev = s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_pages(pages: list[dict]) -> list[Chunk]:
    sections = split_into_sections(pages)
    no_headings = len(sections) == 1 and sections[0]["title"] == "Preamble"

    chunks = []
    for sec in sections:
        body = " ".join(sec["lines"])
        cur_page = sec["page"]
        for part in split_long_text(body):
            marks = list(PAGE_MARK_RE.finditer(part))
            # If the part begins with a page marker it starts on that page; else on the current one
            if marks and not part[: marks[0].start()].strip():
                start_page = int(marks[0].group(1))
            else:
                start_page = cur_page
            if marks:
                cur_page = int(marks[-1].group(1))

            clean = " ".join(PAGE_MARK_RE.sub(" ", part).split())
            if not clean:
                continue

            label = f"Page {start_page}" if no_headings else sec["title"]
            chunks.append(
                Chunk(id=len(chunks),
                      text=f"{label}\n{clean}", section=label, page=start_page)
            )
    return chunks
