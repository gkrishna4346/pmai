"""
Paper Mind AI — backend server
--------------------------------
Python (Flask) backend that:
  1. Extracts text from an uploaded PDF        (pypdf)
  2. Retrieves the most relevant passages       (scikit-learn TF-IDF + cosine similarity)
  3. Identifies the document's domain and
     answers user questions grounded ONLY in
     the document                              (Anthropic Claude API)

Run:
  pip install -r requirements.txt
  cp .env.example .env        # then paste your ANTHROPIC_API_KEY into .env
  python app.py
  open http://localhost:5000
"""

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Always load the .env file that sits next to this script, regardless of
# which directory you happen to run "python app.py" from.
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------------------------
# LLM provider setup — supports Anthropic (Claude) OR Google AI Studio (Gemini).
# Whichever key is present in .env is used. If both are present, Gemini wins
# unless LLM_PROVIDER=anthropic is set explicitly.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
FORCE_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()  # "anthropic" | "gemini" | "huggingface" | ""

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# Safety net: if GEMINI_MODEL was accidentally set to a non-Gemini model name
# (e.g. copied from an ANTHROPIC_MODEL line), fall back to the default instead
# of sending a request that's guaranteed to 404.
if not GEMINI_MODEL.lower().startswith("gemini"):
    print(f"WARNING: GEMINI_MODEL='{GEMINI_MODEL}' doesn't look like a Gemini "
          f"model name — falling back to 'gemini-2.0-flash'. Check your .env file.")
    GEMINI_MODEL = "gemini-2.0-flash"

PROVIDER = None
anthropic_client = None
hf_client = None
genai = None

if FORCE_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    PROVIDER = "anthropic"
elif FORCE_PROVIDER == "gemini" and GOOGLE_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    PROVIDER = "gemini"
elif FORCE_PROVIDER == "huggingface" and HF_API_KEY:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(model=HF_MODEL, token=HF_API_KEY)
    PROVIDER = "huggingface"
elif GOOGLE_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    PROVIDER = "gemini"
elif ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    PROVIDER = "anthropic"
elif HF_API_KEY:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(model=HF_MODEL, token=HF_API_KEY)
    PROVIDER = "huggingface"

app = Flask(__name__, static_folder="static", static_url_path="")

print("-" * 60)
print(f"Paper Mind AI — looking for .env at: {ENV_PATH}")
print(f".env file found on disk: {ENV_PATH.exists()}")
print(f"GOOGLE_API_KEY loaded:    {'yes' if GOOGLE_API_KEY else 'no'}")
print(f"ANTHROPIC_API_KEY loaded: {'yes' if ANTHROPIC_API_KEY else 'no'}")
print(f"HF_API_KEY loaded:        {'yes' if HF_API_KEY else 'no'}")
print(f"Active provider: {PROVIDER or 'NONE — set a key in .env!'}")
_model_by_provider = {"gemini": GEMINI_MODEL, "anthropic": ANTHROPIC_MODEL, "huggingface": HF_MODEL}
print(f"Active model: {_model_by_provider.get(PROVIDER, 'n/a')}")
print("-" * 60)

NOT_FOUND_SENTINEL = "NOT_FOUND_IN_DOCUMENT"
FALLBACK_MSG = "Dear User, Apologies. The searched content is unavailable in the uploaded document."


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
# LLM helper — routes to whichever provider was configured above
# ---------------------------------------------------------------------------
def ask_llm(system_prompt: str, user_message: str) -> str:
    if PROVIDER is None:
        raise RuntimeError(
            "No LLM API key configured on the server. Add ANTHROPIC_API_KEY or "
            "GOOGLE_API_KEY to your .env file (make sure the file is named exactly "
            "'.env', not '.env.example') and restart the app."
        )

    if PROVIDER == "gemini":
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
        response = model.generate_content(
            user_message,
            generation_config={"max_output_tokens": 1000},
        )
        return (response.text or "").strip()

    if PROVIDER == "huggingface":
        completion = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1000,
        )
        return (completion.choices[0].message.content or "").strip()

    # anthropic
    response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


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
        return jsonify({"error": f"Could not read this PDF: {exc}"}), 400

    if len(text) < 20:
        return jsonify({"error": "No readable text found in this PDF."}), 400

    return jsonify({"text": text})


@app.route("/api/build-kb", methods=["POST"])
def build_kb():
    """Identifies the domain/industry of the document."""
    data = request.get_json(force=True) or {}
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

    domain = domain.strip().strip(".\"'")
    return jsonify({"domain": domain})


@app.route("/api/query", methods=["POST"])
def query():
    """Answers a user question grounded strictly in the uploaded document."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    question = data.get("query", "")
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
        f"If, and only if, the requested information genuinely cannot be found in the "
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
    port = int(os.getenv("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
