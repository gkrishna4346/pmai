# Paper Mind AI

An intelligent PDF reader & Q&A assistant. Upload a PDF, it identifies the document's
domain, builds a lightweight searchable knowledge base, and answers questions using
only the content of that document.

## Tech stack

| Layer            | Technology                                                            |
|-------------------|------------------------------------------------------------------------|
| Backend           | Python, Flask                                                          |
| PDF text extraction | `pypdf`                                                               |
| Retrieval / NLP   | `scikit-learn` — TF-IDF vectorization + cosine similarity to find the passages most relevant to a query (full document is used directly for short PDFs) |
| Answer generation & domain classification | Anthropic Claude, Google Gemini, **or** Hugging Face-hosted open models (auto-detected from whichever API key is present) |
| Frontend          | Plain HTML / CSS / JavaScript (no framework) served by Flask, calling the backend's JSON API |

There is no client-side call to any AI API — the browser only talks to your own Flask
server, which is the one holding the API key.

## Setup

```bash
cd paper-mind-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then open .env and paste ONE of: Google AI Studio key, Anthropic key, or a
# Hugging Face token (the file must be named exactly ".env")

python app.py
```

### Choosing a provider
The backend checks `.env` at startup:
- If `GOOGLE_API_KEY` is set, it uses **Gemini** (`gemini-2.0-flash` by default).
- Else if `ANTHROPIC_API_KEY` is set, it uses **Claude** (`claude-sonnet-5` by default).
- Else if `HF_API_KEY` is set, it uses **Hugging Face** (`meta-llama/Llama-3.3-70B-Instruct` by default).
- If more than one is set, that priority order applies unless you force one with `LLM_PROVIDER=gemini|anthropic|huggingface`.

The terminal prints a diagnostic block on every startup showing which keys were
found and which provider/model is active — check that first if something's not working.

### Known limitations by provider
- **Gemini (free tier):** rate-limited; some projects get a `0`-quota free tier until
  billing is linked on the Google Cloud project (see README troubleshooting history / ask
  if you hit a `429` with `limit: 0`).
- **Anthropic:** no free tier — pay-as-you-go per token, small one-time trial credit on new accounts.
- **Hugging Face (free tier):** no billing required, but weaker instruction-following than
  Claude/Gemini — the "answer only from the document, else say not found" behavior is
  less reliable with smaller open models. Larger instruct models (like the Llama-3.3-70B
  default) do best. Cold starts of a few seconds are possible on first request.

Open **http://localhost:5000** in your browser.

## How it works

1. **Upload your PDF** — sent to `POST /api/extract`, text is pulled out with `pypdf`.
2. **Build Knowledge Base** — the first ~6,000 characters are sent to Claude
   (`POST /api/build-kb`) with instructions to return a short domain label
   (e.g. "Airline HR Policy").
3. **User Query** — `POST /api/query`. If the document is long, TF-IDF cosine
   similarity picks the most relevant chunks first; Claude is instructed to answer
   *only* from that content, and to return a fixed sentinel token if the answer
   genuinely isn't in the document. The backend converts that sentinel into:

   > Dear User, Apologies. The searched content is unavailable in the uploaded document.

4. **Refresh icon** in the header reloads the page and clears all state.

## Deploying publicly (Render)

This project includes a `render.yaml` blueprint, so deployment is largely one-click:

1. Push the project to a GitHub repo (`.env` is excluded via `.gitignore` — never commit it).
2. On [render.com](https://render.com), choose **New → Blueprint**, point it at your repo.
   Render reads `render.yaml` automatically and sets up the web service, build
   command, and start command for you.
3. During setup Render will prompt for the env vars marked `sync: false`
   (`HF_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `LLM_PROVIDER`). You only
   need to fill in the ONE you're actually using — leave the other two blank.
4. Deploy. You'll get a public URL like `https://paper-mind-ai.onrender.com`.

Notes:
- Render's free tier sleeps after ~15 minutes of inactivity; the first request
  after that takes 30–50 seconds to wake back up.
- If you'd rather set things up manually instead of via the blueprint: New → Web
  Service → select repo → Build Command `pip install -r requirements.txt` →
  Start Command `gunicorn app:app` → add your key under the Environment tab.



- This demo keeps everything in memory in the browser tab (document text is round-tripped
  with each request) rather than a database — fine for single-user local use. For
  multi-user or production use, add server-side session storage.
- Swap in your own logo at `static/` and reference it from `index.html` if you'd like the
  exact brand asset instead of the built-in SVG mark.
