Faithful & Observable RAG — Fullstack Edition

A transparent, debuggable Retrieval-Augmented Generation system with React UI and FastAPI backend.

This upgraded repo contains everything needed to run a production-style RAG system with:

React + Tailwind frontend

FastAPI backend

ChromaDB for retrieval

Per-answer trust score, evidence preview, and relevance badges

Pinned evidence card, hover previews, run history, and session persistence

This system is built for explainability and real-world testing — perfect for demos, research, or prototyping enterprise-grade RAG.

 What’s Included
 Frontend (frontend/)

A polished React UI (Vite + Tailwind) with:

Chat interface

Trust panel

Evidence preview (hover + pin)

Highlighting with <mark>

Top-k slider

Session-persisted chat history

Main files:

frontend/src/pages/ChatPage.tsx   # Full chat + evidence UI
frontend/src/main.tsx
frontend/tailwind.config.js
frontend/package.json

 Backend (backend/)

FastAPI server powering the RAG pipeline:

app.py                # API routes (query, rebuild index)
rag_pipeline.py       # Retrieval + generation logic
embedder.py           # Embedding model wrapper
chunk_utils.py        # Chunking helpers
metrics.py            # Trust score calculation


Backend features include:

ChromaDB vector store

Embedding + retrieval

Per-chunk relevance scoring (Related / Somewhat / Off-topic)

Trust score & latency tracking

Strict retrieval mode

Easy rebuild of entire index

 Data Folder (data/)

Place your documents here:

data/
    ml_basics.txt
    climate_change.txt
    psychology_of_habits.txt


Supports:

.txt

.md

.json (flat text fields)

⚡ Quickstart
1️ Backend Setup
cd backend
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI key or other LLM keys
python app.py


Backend runs at:

http://localhost:8000

2️⃣ Frontend Setup
cd frontend
npm install
npm run dev


Frontend will be available at:

http://localhost:5173

How It Works

1. User asks a question

Frontend sends → backend via /api/query

2. Retrieve chunks

ChromaDB returns top-k chunks with:

distance

relevance

text

source filename

3. LLM generates grounded answer

Answer + sources → returned to frontend.

4. UI displays:

Response

Source pills

Evidence preview card

Highlights

Trust score

Retrieval breakdown

 Key Features
 Evidence Preview

Hover = preview
Click = pin
Scrolling no longer flickers (fixed).

 Relevance Badges

🟢 Related

🔵 Somewhat related

🔴 Off-topic

 Session Persistence

Chat saved in sessionStorage

Run history saved

Clear chat resets everything

 Trust Insights Panel

Shows:

Trust score

Latency

Retrieved chunk count

Relevance breakdown

Mini distance bars

 Index Rebuild

One-click rebuild of all embeddings via /api/rebuild.


Future Improvements

(Some features you can add later — already structured for expansion)

Answer-rating (👍/👎)

Run History panel (per-answer analytics)

Compare runs between model versions

Heatmap for relevance

Automatic query rewriting

Logging & analytics dashboard
