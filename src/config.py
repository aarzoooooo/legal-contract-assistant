import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model names change often. Run list_models.py to see what your key can use.
GENERATION_MODEL = "gemini-3.5-flash"
# used if the main model is overloaded
FALLBACK_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

REQUEST_TIMEOUT_MS = 30_000   # fail a single call after 30 seconds

TOP_K = 4
MIN_SCORE = 0.60   # chunks scoring below this are treated as irrelevant
