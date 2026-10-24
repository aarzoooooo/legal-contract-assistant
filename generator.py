import time

import httpx
from google import genai
from google.genai import errors, types

from src.config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    FALLBACK_MODEL,
    REQUEST_TIMEOUT_MS,
)

# Short timeout and no hidden SDK retries, so our own retry logic stays in control
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

NOT_FOUND = "I couldn't find this in the uploaded contract."

RETRYABLE_CODES = (429, 500, 502, 503, 504)

SYSTEM_PROMPT = f"""You are a contract review assistant. Answer the user's question using ONLY the contract excerpts provided.

Rules:
- Base every statement on the excerpts. Do not use outside knowledge or assumptions.
- Cite the source for each claim like this: (Source 2, Section 4. Termination, p.1).
- Answer only if an excerpt directly addresses what was asked. An excerpt that merely shares a word with the question, or covers a loosely related topic, does not count (for example, a confidentiality clause does not answer a question about payment terms). In that case reply exactly: "{NOT_FOUND}"
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- If the excerpts only partly answer the question, answer the supported part and clearly say what is missing.
- Quote key figures (amounts, dates, notice periods) exactly as written.
- This is an informational summary, not legal advice."""


def build_context(results) -> str:
    parts = []
    for i, (chunk, _score) in enumerate(results, start=1):
        parts.append(
            f"[Source {i} | {chunk.section} | page {chunk.page}]\n{chunk.text}")
    return "\n\n".join(parts)


def generate_with_retry(
    prompt: str,
    system_prompt: str,
    response_schema=None,
    max_retries: int = 2,
) -> str:
    """Try the main model with exponential backoff, then fall back to another model.

    If response_schema (a Pydantic model class) is given, Gemini returns JSON matching it.
    """
    config_kwargs = dict(system_instruction=system_prompt, temperature=0.1)
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    config = types.GenerateContentConfig(**config_kwargs)

    last_error = None
    # no duplicates
    models = list(dict.fromkeys((GENERATION_MODEL, FALLBACK_MODEL)))

    for model in models:
        for attempt in range(max_retries):
            try:
                print(
                    f"[info] calling {model} (attempt {attempt + 1}/{max_retries})")
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return resp.text
            except httpx.TimeoutException as e:
                last_error = e
                print(f"[retry] {model} timed out")
            except errors.APIError as e:
                last_error = e
                if e.code in RETRYABLE_CODES:
                    print(f"[retry] {model} returned {e.code}")
                elif e.code == 404:
                    print(
                        f"[warn] {model} is not available (404), trying next model")
                    break
                else:
                    raise                       # e.g. bad API key: retrying won't help
            time.sleep(2 ** attempt)            # wait 1s, 2s
        print(f"[warn] giving up on {model}")

    raise RuntimeError(
        f"Gemini is currently unavailable. Last error: {last_error}")


def answer_question(question: str, results) -> dict:
    """results: list of (Chunk, score) from Retriever.retrieve()."""
    # Safety net 1: retrieval found nothing relevant, so don't call the LLM at all
    if not results:
        return {"answer": NOT_FOUND, "sources": []}

    prompt = f"Contract excerpts:\n\n{build_context(results)}\n\nQuestion: {question}"
    answer = generate_with_retry(prompt, SYSTEM_PROMPT)

    # Safety net 2: the model said the excerpts don't answer the question,
    # so don't show those excerpts as sources for that answer
    if NOT_FOUND.lower() in answer.lower() and len(answer) < len(NOT_FOUND) + 20:
        return {"answer": NOT_FOUND, "sources": []}

    return {
        "answer": answer,
        "sources": [
            {"section": c.section, "page": c.page,
                "score": round(s, 3), "text": c.text}
            for c, s in results
        ],
    }
