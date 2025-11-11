# Faithful & Observable RAG — Starter

This repo scaffolds a week-long build of a transparent, faithful RAG system with observability and a Streamlit UI.

## What’s included
- `app.py` — Streamlit UI (query box, retrieved context, answer, metrics preview)
- `rag_pipeline.py` — Retrieval + generation pipeline hooks
- `metrics.py` — Precision/recall/latency/faithfulness stubs
- `data/` — Put your documents here (txt/markdown/jsonl). Tiny sample is included.
- `.env.example` — Where to put API keys (if you use OpenAI, etc.)
- `requirements.txt` — Python deps
- `.gitignore` — Python + common cache ignores

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # then edit with your keys if needed

# First run (creates a local ChromaDB index under .chromadb/)
streamlit run app.py
```
