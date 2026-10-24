from typing import Literal

from pydantic import BaseModel, ValidationError

from src.generator import generate_with_retry, build_context


class Obligation(BaseModel):
    party: str
    obligation: str
    deadline_or_condition: str
    source_section: str


class TerminationClause(BaseModel):
    trigger: str
    notice_period: str
    consequences: str
    source_section: str


class Risk(BaseModel):
    description: str
    severity: Literal["low", "medium", "high"]
    reason: str
    source_section: str


class ContractAnalysis(BaseModel):
    obligations: list[Obligation]
    termination_clauses: list[TerminationClause]
    risks: list[Risk]


# One targeted query per category, used to pull the relevant chunks
EXTRACTION_QUERIES = [
    "obligations, duties and responsibilities of each party, payment, deadlines",
    "termination of the agreement, notice period, breach, cancellation",
    "liability, indemnity, penalties, confidentiality, governing law, dispute resolution",
]

EXTRACTION_PROMPT = """You are a contract review assistant doing a compliance-style review.
From the contract excerpts provided, extract three things:

1. obligations: what each party must do (party, obligation, deadline_or_condition, source_section)
2. termination_clauses: how and when the contract can end (trigger, notice_period, consequences, source_section)
3. risks: potential risks to a party, such as one-sided terms, unusual penalties, broad liability,
   vague wording, or short deadlines (description, severity, reason, source_section)

Rules:
- Use ONLY the excerpts. Do not invent terms or add outside knowledge.
- source_section must be the section title exactly as it appears in the excerpt header.
- If a detail (such as a notice period) is not stated, write "Not specified".
- You only see excerpts, not the whole contract, so never claim a clause is missing.
  Every risk must point to text that actually appears in the excerpts.
- severity is low, medium or high, and reason must explain it using the contract's wording.
- Return empty lists for categories with nothing in the excerpts.
- This is informational analysis, not legal advice."""


def _gather_chunks(retriever, k: int = 5):
    """Retrieve chunks for every query, drop duplicates, and keep contract order."""
    found = {}
    for q in EXTRACTION_QUERIES:
        # min_score=0 because these queries are broad; ranking still picks the best chunks
        for chunk, score in retriever.retrieve(q, k=k, min_score=0.0):
            found[chunk.id] = (chunk, score)
    return [found[i] for i in sorted(found)]


def analyze_contract(retriever, max_attempts: int = 2) -> ContractAnalysis:
    results = _gather_chunks(retriever)
    prompt = f"Contract excerpts:\n\n{build_context(results)}"

    last_error = None
    for _ in range(max_attempts):
        raw = generate_with_retry(
            prompt, EXTRACTION_PROMPT, response_schema=ContractAnalysis)
        try:
            # Pydantic checks the structure
            return ContractAnalysis.model_validate_json(raw)
        except ValidationError as e:
            last_error = e
            print("[warn] model returned invalid JSON, retrying...")

    raise ValueError(f"Could not get valid structured output: {last_error}")
