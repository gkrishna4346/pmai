"""
Paper Mind AI — backend server
--------------------------------
Python (Flask) backend that:
  1. Extracts text from an uploaded PDF            (pypdf)
  2. Retrieves the most relevant passages          (scikit-learn TF-IDF + cosine similarity)
  3. Identifies the document's domain and answers
     questions grounded ONLY in the document       (LLM providers, tried in order:
                                                    Gemini, Groq, SambaNova, Anthropic,
                                                    Hugging Face — whichever keys you set)

Run locally:
  pip install -r requirements.txt
  cp .env.example .env        # then paste at least one real API key into .env
  python app.py
  open http://localhost:5050

Production (Render etc.):
  gunicorn app:app --timeout 120 --workers 1 --threads 4
"""

import io
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.exceptions import HTTPException

# Always load the .env file that sits next to this script, regardless of
# which directory you happen to run "python app.py" from.
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------------------------
# Small helpers for reading configuration safely
# ---------------------------------------------------------------------------
# Values copied straight from .env.example look like "your_groq_key_here".
# Treat those as "not set" so a placeholder never counts as a real key.
_PLACEHOLDER = re.compile(r"^your[_-]|[_-]here$|^changeme$|^xxx+$", re.IGNORECASE)


def _clean_key(*names: str):
    """Return the first real API key found among the given env var names."""
    for name in names:
        value = (os.getenv(name) or "").strip().strip('"').strip("'")
        if not value:
            continue
        if _PLACEHOLDER.search(value):
            print(f"WARNING: {name} still contains placeholder text — ignoring it. "
                  f"Paste your real key, or delete the line.")
            continue
        return value
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _short(exc: Exception, limit: int = 160) -> str:
    """One-line, length-limited version of an error message for the UI."""
    msg = " ".join(str(exc).split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = _clean_key("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = _clean_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
HF_API_KEY = _clean_key("HF_API_KEY", "HUGGINGFACE_API_KEY")
GROQ_API_KEY = _clean_key("GROQ_API_KEY")
SAMBANOVA_API_KEY = _clean_key("SAMBANOVA_API_KEY")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SAMBANOVA_MODEL = os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")

# Max seconds to wait on any single provider before falling through to the next.
PROVIDER_TIMEOUT = _env_int("PROVIDER_TIMEOUT", 40)
# Upload and text limits keep the server within a small (free-tier) memory budget.
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 25)
MAX_TEXT_CHARS = _env_int("MAX_TEXT_CHARS", 400_000)
MAX_QUESTION_CHARS = 1000

# Safety net: if GEMINI_MODEL was accidentally set to a non-Gemini model name
# (e.g. copied from an ANTHROPIC_MODEL line), fall back to the default instead
# of sending a request that's guaranteed to 404.
if not GEMINI_MODEL.lower().startswith("gemini"):
    print(f"WARNING: GEMINI_MODEL='{GEMINI_MODEL}' doesn't look like a Gemini "
          f"model name — falling back to 'gemini-3.8-flash'. Check your .env file.")
    GEMINI_MODEL = "gemini-3.8-flash"

KNOWN_PROVIDERS = ["gemini", "groq", "sambanova", "anthropic", "huggingface"]
DEFAULT_ORDER = KNOWN_PROVIDERS
_order_env = os.getenv("PROVIDER_ORDER", "")
PROVIDER_ORDER = [p.strip().lower() for p in _order_env.split(",") if p.strip()] or DEFAULT_ORDER
for _name in PROVIDER_ORDER:
    if _name not in KNOWN_PROVIDERS:
        print(f"WARNING: PROVIDER_ORDER contains unknown provider '{_name}' "
              f"(valid: {', '.join(KNOWN_PROVIDERS)}) — ignoring it.")

# ---------------------------------------------------------------------------
# LLM providers — every provider with a real key in .env is registered here as
# a function(system_prompt, user_message) -> str. ask_llm() tries them in
# order and falls through to the next one if a call fails (rate limit, quota
# exceeded, outage, timeout), so several free-tier keys can back each other up.
# ---------------------------------------------------------------------------
_PROVIDER_CALLS = {}

if GOOGLE_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

    def _call_gemini(system_prompt: str, user_message: str) -> str:
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
        response = model.generate_content(
            user_message,
            generation_config={"max_output_tokens": 1000},
            request_options={"timeout": PROVIDER_TIMEOUT},
        )
        return (response.text or "").strip()

    _PROVIDER_CALLS["gemini"] = _call_gemini

if GROQ_API_KEY:
    from openai import OpenAI  # Groq exposes an OpenAI-compatible endpoint
    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        timeout=PROVIDER_TIMEOUT,
        max_retries=0,
    )

    def _call_groq(system_prompt: str, user_message: str) -> str:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return (completion.choices[0].message.content or "").strip()

    _PROVIDER_CALLS["groq"] = _call_groq

if SAMBANOVA_API_KEY:
    from openai import OpenAI  # SambaNova exposes an OpenAI-compatible endpoint
    sambanova_client = OpenAI(
        api_key=SAMBANOVA_API_KEY,
        base_url="https://api.sambanova.ai/v1",
        timeout=PROVIDER_TIMEOUT,
        max_retries=0,
    )

    def _call_sambanova(system_prompt: str, user_message: str) -> str:
        completion = sambanova_client.chat.completions.create(
            model=SAMBANOVA_MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return (completion.choices[0].message.content or "").strip()

    _PROVIDER_CALLS["sambanova"] = _call_sambanova

if ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=PROVIDER_TIMEOUT, max_retries=0)

    def _call_anthropic(system_prompt: str, user_message: str) -> str:
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    _PROVIDER_CALLS["anthropic"] = _call_anthropic

if HF_API_KEY:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(model=HF_MODEL, token=HF_API_KEY, timeout=PROVIDER_TIMEOUT)

    def _call_huggingface(system_prompt: str, user_message: str) -> str:
        completion = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1000,
        )
        return (completion.choices[0].message.content or "").strip()

    _PROVIDER_CALLS["huggingface"] = _call_huggingface

# The actual fallback chain: configured providers, in priority order.
ACTIVE_CHAIN = [p for p in PROVIDER_ORDER if p in _PROVIDER_CALLS]
# Any configured provider not mentioned in PROVIDER_ORDER still gets appended
# at the end, so it's never silently unused.
ACTIVE_CHAIN += [p for p in _PROVIDER_CALLS if p not in ACTIVE_CHAIN]

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

print("-" * 60)
print(f"Paper Mind AI — looking for .env at: {ENV_PATH}")
print(f".env file found on disk: {ENV_PATH.exists()}")
print(f"GOOGLE_API_KEY loaded:    {'yes' if GOOGLE_API_KEY else 'no'}")
print(f"GROQ_API_KEY loaded:      {'yes' if GROQ_API_KEY else 'no'}")
print(f"SAMBANOVA_API_KEY loaded: {'yes' if SAMBANOVA_API_KEY else 'no'}")
print(f"ANTHROPIC_API_KEY loaded: {'yes' if ANTHROPIC_API_KEY else 'no'}")
print(f"HF_API_KEY loaded:        {'yes' if HF_API_KEY else 'no'}")
print(f"Fallback chain (in order): {ACTIVE_CHAIN or 'NONE — set at least one key in .env!'}")
print(f"Per-provider timeout: {PROVIDER_TIMEOUT}s | Max upload: {MAX_UPLOAD_MB} MB")
print("-" * 60)

NOT_FOUND_SENTINEL = "NOT_FOUND_IN_DOCUMENT"
FALLBACK_MSG = "Dear User, Apologies. The searched content is unavailable in the uploaded document."


@app.errorhandler(Exception)
def handle_error(exc):
    """Always answer /api/* requests with JSON, never an HTML error page, so the
    browser can show a readable message instead of 'Unexpected token <'."""
    is_http = isinstance(exc, HTTPException)
    status = exc.code if is_http else 500
    if not is_http:
        app.logger.exception("Unhandled error while serving %s", request.path)

    if request.path.startswith("/api/"):
        if status == 413:
            message = f"That file is too large (limit {MAX_UPLOAD_MB} MB)."
        elif is_http:
            message = exc.description
        else:
            message = "Something went wrong on the server. Please try again."
        return jsonify({"error": message}), status

    return exc if is_http else ("Internal Server Error", 500)


# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(size - overlap, 1)
    return chunks


def relevant_context(text: str, query: str, top_n: int = 5) -> str:
    """Return the passages most relevant to the query using TF-IDF + cosine similarity.
    Short documents are passed through in full for maximum accuracy."""
    if len(text) <= 6000:
        return text

    chunks = chunk_text(text)
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(chunks + [query])
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked = sorted(zip(chunks, similarities), key=lambda pair: pair[1], reverse=True)
        top = [c for c, score in ranked[:top_n] if score > 0]
        if not top:
            top = [c for c, _ in ranked[:top_n]]
        return "\n---\n".join(top)
    except ValueError:
        # e.g. query contains only stop-words
        return "\n---\n".join(chunks[:top_n])


# ---------------------------------------------------------------------------
# LLM helper — tries each configured provider in order (ACTIVE_CHAIN),
# automatically falling through to the next one if a call fails.
# ---------------------------------------------------------------------------
def ask_llm(system_prompt: str, user_message: str) -> str:
    if not ACTIVE_CHAIN:
        raise RuntimeError(
            "No LLM API key is configured on the server. Add at least one real key "
            "(GOOGLE_API_KEY, GROQ_API_KEY, SAMBANOVA_API_KEY, ANTHROPIC_API_KEY or "
            "HF_API_KEY) to your .env file or your hosting dashboard, then restart."
        )

    errors = []
    for name in ACTIVE_CHAIN:
        try:
            result = _PROVIDER_CALLS[name](system_prompt, user_message)
            if result:
                if name != ACTIVE_CHAIN[0]:
                    print(f"[ask_llm] Fell back to '{name}' after earlier provider(s) failed.")
                return result
            errors.append(f"{name}: returned an empty response")
        except Exception as exc:  # noqa: BLE001
            print(f"[ask_llm] Provider '{name}' failed: {exc}")  # full detail stays in server logs
            errors.append(f"{name}: {_short(exc)}")

    raise RuntimeError("All configured AI providers failed. " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    """Lightweight check for hosting platforms; also shows which providers are active."""
    return jsonify({"status": "ok", "providers": ACTIVE_CHAIN})


@app.route("/api/extract", methods=["POST"])
def extract():
    """Accepts a PDF upload, returns its extracted text."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        reader = PdfReader(io.BytesIO(file.read()))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not read this PDF: {_short(exc)}"}), 400

    if len(text) < 20:
        return jsonify({
            "error": "No readable text found in this PDF. Scanned or image-only "
                     "PDFs are not supported yet."
        }), 400

    truncated = len(text) > MAX_TEXT_CHARS
    return jsonify({"text": text[:MAX_TEXT_CHARS], "truncated": truncated})


@app.route("/api/build-kb", methods=["POST"])
def build_kb():
    """Identifies the domain/industry of the document."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No document text supplied."}), 400

    sample = text[:6000]
    try:
        domain = ask_llm(
            "You classify documents. Read the excerpt and reply with ONLY a concise "
            "2-5 word label naming the professional domain, industry, or document type "
            "(e.g. 'Airline HR Policy', 'Medical Research', 'Financial Contract'). "
            "No punctuation, no explanation, no extra text.",
            f"Document excerpt:\n\n{sample}",
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    # Keep only the first line and cap the length in case a model rambles.
    domain = (domain.strip().splitlines() or [""])[0].strip().strip(".\"'")[:60]
    return jsonify({"domain": domain or "General Document"})


@app.route("/api/query", methods=["POST"])
def query():
    """Answers a user question grounded strictly in the uploaded document."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    question = (data.get("query", "") or "").strip()[:MAX_QUESTION_CHARS]
    domain = data.get("domain", "")

    if not text or not question:
        return jsonify({"error": "Missing document text or query."}), 400

    context = relevant_context(text, question)
    system_prompt = (
        "You are Paper Mind AI, a professional document Q&A assistant. Answer the "
        "user's question using ONLY the information contained in the provided document "
        "context below — never use outside knowledge and never guess. Present the answer "
        "in a clear, professional, well-structured way: short paragraphs and bullet points "
        "where helpful, and bold key figures, dates, or terms using **double asterisks**. "
        "If, and only if, the requested information genuinely cannot be found in the "
        f"context, reply with exactly this single token and nothing else: {NOT_FOUND_SENTINEL}"
        f"\n\nDOCUMENT CONTEXT:\n{context}"
    )

    try:
        raw = ask_llm(system_prompt, f"User question: {question}")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    if NOT_FOUND_SENTINEL in raw:
        return jsonify({"found": False, "message": FALLBACK_MSG})

    return jsonify({"found": True, "answer": raw, "domain": domain})


if __name__ == "__main__":
    port = _env_int("PORT", 5050)
    debug = os.getenv("FLASK_DEBUG", "1") != "0"
    app.run(debug=debug, host="0.0.0.0", port=port)
