# Paper Mind AI — Code Walkthrough Guide

This is a plain-English tour of every file in the project, in the order you'd
want to read them to understand how the whole thing fits together.

```
paper-mind-ai/
├── app.py              ← the backend "brain" (Python/Flask)
├── static/index.html   ← everything the user sees and clicks (HTML/CSS/JS)
├── requirements.txt    ← list of Python packages the app depends on
├── .env.example         ← template for your secret API key
├── Procfile / render.yaml  ← deployment configuration
├── .gitignore           ← tells git which files NOT to upload
└── README.md            ← setup instructions
```

The big-picture idea: **the browser never talks to any AI service directly.**
It only talks to your own `app.py` server. `app.py` is the only thing that
holds your API key and the only thing that calls Claude/Gemini/Hugging Face.
This is safer (your key is never visible to anyone using the app) and is why
it needs a real server instead of just being a standalone HTML file.

---

## 1. `app.py` — the backend

### a) Imports (lines 18–26)
```python
import io, os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
- `flask` — the web framework. It's what lets Python "listen" for requests
  coming from a browser and send responses back.
- `pypdf` — reads text out of PDF files.
- `sklearn` (scikit-learn) — the actual "NLP" piece: it turns text into
  numbers so we can mathematically measure how *relevant* a passage is to a
  question (explained more in section 1d).
- `dotenv` — reads your `.env` file and loads it into `os.environ`, so
  `os.getenv("GOOGLE_API_KEY")` works.

### b) Loading your `.env` file (lines 28–31)
```python
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
```
This finds `.env` **next to `app.py` itself**, regardless of which folder you
happened to be standing in when you typed `python app.py`. (Earlier versions
used to guess based on your current folder, which is what caused the "key not
found" bug you ran into — this line is the fix.)

### c) Provider detection (lines 33–96)
This whole block just answers one question: *"Which AI service do I have a
key for, and which one should I use?"*

```python
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
```
It reads all three possible keys from `.env`. Whichever one is actually
filled in becomes `PROVIDER` ("gemini", "anthropic", or "huggingface") —
everything downstream just asks "which provider am I using?" once, rather
than repeating this logic everywhere.

The `if/elif` chain also handles the case where you set more than one key —
by default Gemini wins, unless you explicitly set `LLM_PROVIDER=anthropic` (etc.)
in `.env` to force a specific one.

The diagnostic `print()` block right after is what shows you that startup
summary in the terminal (".env file found: True", "Active provider: gemini"...)
— purely there to make debugging config problems fast.

### d) `chunk_text()` and `relevant_context()` (lines 105–133) — the "NLP" part
This is the piece that makes the app work well on **long** documents without
sending the entire PDF to the AI every time (which would be slow and
expensive).

- `chunk_text()` slices the document into overlapping ~1,200-character
  pieces. Overlap prevents an answer from being awkwardly split across two
  chunks with neither having the full context.
- `relevant_context()` is the retrieval step:
  - If the document is short (under 6,000 characters), it just returns the
    whole thing — no need to be clever.
  - If it's long, it uses **TF-IDF** (Term Frequency–Inverse Document
    Frequency) to convert every chunk *and* your question into vectors of
    numbers, where words that are distinctive (not "the", "and", etc.)
    weigh more heavily. It then measures **cosine similarity** — basically,
    "how much does this chunk's numeric fingerprint point in the same
    direction as the question's fingerprint?" — and keeps the top 5 most
    similar chunks. This is a lightweight, classic NLP technique — no AI
    model is needed for this step, it's plain math (`scikit-learn`).

### e) `ask_llm()` (lines 139–172) — the actual AI call
One function, three possible implementations depending on `PROVIDER`:
- **Gemini**: creates a `GenerativeModel` with your system instructions baked
  in, then calls `.generate_content()` with the user's question.
- **Hugging Face**: uses the OpenAI-style `chat_completion()` interface —
  a list of `{"role": ..., "content": ...}` messages, same shape as most
  chat APIs use.
- **Anthropic**: calls `.messages.create()` with a `system` prompt and a
  `messages` list.

Whichever branch runs, the function always returns a plain string — the rest
of the app never needs to know or care which provider actually answered.

### f) The routes — what happens when the browser talks to the server
Flask routes are just "when a request comes in to this URL, run this
function." There are four:

| Route | Called when | What it does |
|---|---|---|
| `GET /` | You load the page | Serves `static/index.html` |
| `POST /api/extract` | You upload a PDF | Reads the PDF with `pypdf`, returns the extracted text as JSON |
| `POST /api/build-kb` | You click "Build Knowledge Base" | Sends the first 6,000 characters to the AI with instructions to output a short domain label |
| `POST /api/query` | You ask a question | Runs `relevant_context()`, builds a strict system prompt ("answer ONLY from this text, or reply with a special token if it's not there"), calls `ask_llm()`, and converts that special token into the apology message if needed |

The "special token" trick (`NOT_FOUND_IN_DOCUMENT`) is worth understanding:
rather than trusting the AI to type out your exact apology sentence
correctly every time, we ask it to output one fixed, unambiguous word when
it can't find an answer. The Python code then does the exact substitution
itself — much more reliable than hoping the AI's wording matches exactly.

### g) The bottom (lines 260–262)
```python
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
```
This only runs when you execute `app.py` directly (not when it's imported as
a module elsewhere, e.g. by `gunicorn` in production). `debug=True` gives you
auto-reload on code changes and detailed error pages — turn this off for a
real public deployment (Render's `gunicorn app:app` command already bypasses
this block entirely, which is exactly right).

---

## 2. `static/index.html` — the frontend

This single file contains three things stacked together: CSS (styling),
HTML (structure), and JavaScript (behavior).

### a) CSS — the visual design
Near the top, `:root { --bg1: ...; --card1: ...; }` defines a small palette
of reusable colors ("CSS variables") so the blue gradient theme stays
consistent everywhere instead of repeating hex codes throughout the file.
Below that, each class (`.hero`, `.card`, `.upload-card`, etc.) styles one
visual block of the page — matching the header banner, description card,
upload box, and so on from your original design mockup.

### b) HTML — the structure
Reading top to bottom, the markup mirrors what you see on screen:
1. `.hero` — logo, title, tagline, refresh button
2. Two-column row — description text + upload box
3. Second row — "Build Knowledge Base" button + domain result box
4. Query row — label + text input + send button
5. Answer card — where results (or the apology message) appear

### c) JavaScript — the behavior
State is kept in three simple variables at the top:
```js
let fullText = '';      // extracted PDF text
let currentDomain = ''; // detected domain label
let kbReady = false;    // whether Build Knowledge Base has succeeded yet
```

Three event listeners drive everything:
- **`fileInput.addEventListener('change', ...)`** — fires when you pick a
  file. Sends it to `/api/extract` as `FormData` (the standard way browsers
  upload files), stores the returned text in `fullText`, and enables the
  "Build Knowledge Base" button.
- **`buildKbBtn.addEventListener('click', ...)`** — sends `fullText` as JSON
  to `/api/build-kb`, displays the returned domain, sets `kbReady = true`,
  and enables the query input.
- **`sendBtn` / Enter key → `handleQuery()`** — sends `{text, query, domain}`
  as JSON to `/api/query`. If the response says `found: false`, it shows the
  apology message; otherwise it renders the answer through
  `renderMarkdownLite()`, a small hand-written function that turns `**bold**`
  and `- bullet` lines into real HTML, since we don't want to pull in a full
  markdown library for such a small need.

Every network call is wrapped in `try/catch` so failures (bad key, rate
limit, network hiccup) show up as a readable message in the UI instead of
silently doing nothing.

---

## 3. The supporting files

- **`requirements.txt`** — the exact list of Python packages `pip install -r
  requirements.txt` installs. Version numbers use `>=` (not `==`) so pip can
  pick whichever compatible version has a ready-made "wheel" for your OS —
  this is what fixed the earlier `cryptography` build failure.
- **`.env.example`** — a *template*, not read by the app itself. You copy it
  to a real `.env` file and fill in a key; `.env` is what `load_dotenv()`
  actually reads.
- **`.gitignore`** — tells `git` to never upload `.env`, your virtual
  environment folder, or Python's cache files to GitHub. This is what keeps
  your API key private when you push the project publicly.
- **`Procfile`** — one line (`web: gunicorn app:app`) telling a hosting
  platform how to start your app in production, using `gunicorn` (a proper
  production web server) instead of Flask's built-in development server.
- **`render.yaml`** — a blueprint Render reads automatically to configure
  the whole deployment (build command, start command, which environment
  variables to ask you for) without you clicking through settings manually.

---

## Where to go from here
If you want to extend this project, the natural next steps in rough order of
difficulty are: (1) support multiple PDFs at once, (2) persist documents in
a real database instead of round-tripping the full text on every request,
(3) add user accounts/sessions, (4) swap the TF-IDF retrieval for a proper
vector-embedding search if you outgrow keyword-based matching. Happy to walk
through any of those whenever you're ready.
