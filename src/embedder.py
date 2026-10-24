import hashlib
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types

from src.config import GEMINI_API_KEY, EMBEDDING_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

BATCH_SIZE = 20
# free tier allows 100 embedding requests per minute; stay under it
RATE_LIMIT_PER_MIN = 90
MAX_RETRIES = 5
CACHE_DIR = Path(".cache")

_sent: deque = deque()    # (timestamp, count) of recent embedding requests


class DailyQuotaExceeded(RuntimeError):
    """The free-tier daily embedding limit has been used up."""


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def _wait_for_quota(n: int) -> None:
    """Block until sending n more requests keeps us under the per-minute limit."""
    while True:
        now = time.time()
        while _sent and now - _sent[0][0] > 60:
            _sent.popleft()
        used = sum(count for _, count in _sent)
        if used + n <= RATE_LIMIT_PER_MIN:
            break
        wait = 60 - (now - _sent[0][0]) + 0.5
        print(f"[info] embedding quota: waiting {wait:.0f}s ...", flush=True)
        time.sleep(wait)
    _sent.append((time.time(), n))


def _retry_delay(e: Exception, default: float = 30.0) -> float:
    """Read 'Please retry in 42.1s' from the API error, if present."""
    m = re.search(r"retry in ([\d.]+)s", str(e))
    return float(m.group(1)) + 1 if m else default


def _embed_batch(batch: list[str], task_type: str) -> list:
    for attempt in range(MAX_RETRIES):
        _wait_for_quota(len(batch))
        try:
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except errors.APIError as e:
            if e.code != 429:
                raise
            if "PerDay" in str(e):      # daily limit: waiting a few seconds will not help
                raise DailyQuotaExceeded(
                    "The free Gemini embedding quota for today is used up. "
                    "It resets at midnight Pacific time (about 12:30 PM in India). Please try again then."
                ) from e
            if attempt == MAX_RETRIES - 1:
                raise
            delay = _retry_delay(e)
            print(
                f"[info] embedding rate limit hit, waiting {delay:.0f}s ...", flush=True)
            time.sleep(delay)


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Embed a list of texts in batches. Returns a normalized float32 array (n, dim)."""
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        all_vecs.extend(_embed_batch(texts[i: i + BATCH_SIZE], task_type))
    return _normalize(np.array(all_vecs, dtype="float32"))


def embed_documents_cached(texts: list[str]) -> np.ndarray:
    """Embed contract chunks, saving progress after every batch.

    The finished result is cached on disk, so each contract is embedded only once.
    If a run is interrupted, the next run resumes from the last saved batch.
    """
    key_source = EMBEDDING_MODEL + "\x00" + "\x00".join(texts)
    key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:16]
    final_path = CACHE_DIR / f"{key}.npy"
    partial_path = CACHE_DIR / f"{key}.partial.npy"

    if final_path.exists():
        return np.load(final_path)

    CACHE_DIR.mkdir(exist_ok=True)
    raw = list(np.load(partial_path)) if partial_path.exists() else []
    if raw:
        print(
            f"[info] resuming from {len(raw)}/{len(texts)} chunks", flush=True)

    for i in range(len(raw), len(texts), BATCH_SIZE):
        raw.extend(_embed_batch(
            texts[i: i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        np.save(partial_path, np.array(raw, dtype="float32"))   # checkpoint
        print(f"[info] embedded {len(raw)}/{len(texts)} chunks", flush=True)

    vecs = _normalize(np.array(raw, dtype="float32"))
    np.save(final_path, vecs)
    partial_path.unlink(missing_ok=True)
    return vecs


def embed_query(query: str) -> np.ndarray:
    """Embed a single question. Returns shape (1, dim). Repeated questions are served from disk."""
    key = hashlib.sha256(
        (EMBEDDING_MODEL + "\x00" + query).encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / f"q_{key}.npy"
    if path.exists():
        return np.load(path)

    vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, vec)
    return vec
