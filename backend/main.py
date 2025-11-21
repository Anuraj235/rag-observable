import time
import json
import uuid
from datetime import datetime
import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline, Document  # assumes rag_pipeline.py is in PYTHONPATH


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class ChunkOut(BaseModel):
    source: str
    chunk: int
    text: str
    distance: Optional[float] = None
    relevance: str


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    trust_score: int
    chunks: List[ChunkOut]


app = FastAPI(title="Faithful RAG API")

# ---------------------------------------------------------
# STEP 1: Logging directory setup
# ---------------------------------------------------------
LOG_DIR = "data"
LOG_PATH = os.path.join(LOG_DIR, "run_logs.jsonl")

# Ensure the directory exists
os.makedirs(LOG_DIR, exist_ok=True)
# ---------------------------------------------------------


# CORS so React (localhost:5173, 3000, etc.) can call the API
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise one global pipeline instance
pipeline = RAGPipeline(
    data_dir="data",
    persist_dir=".chromadb",
    embed_model="all-MiniLM-L6-v2",
)


def compute_trust_score(docs: List[Document]) -> int:
    """Simple heuristic for now so UI has something real to show."""
    if not docs:
        return 0
    sources = {d.metadata.get("source", "unknown") for d in docs}
    base = 60 + min(len(docs) * 5, 20) + min(len(sources) * 5, 20)
    return max(0, min(99, base))


def relevance_label(distance: Optional[float]) -> str:
    """
    Turn cosine distance into a human-readable relevance label.

    Smaller distance = more similar.
    """
    if distance is None:
        return "Unknown"
    if distance < 0.35:
        return "Related"
    if distance < 0.65:
        return "Somewhat related"
    return "Off-topic"


# ---------------------------------------------------------
# STEP 2: Helper to log one RAG run to JSONL
# ---------------------------------------------------------
def log_run_to_file(
    query: str,
    answer: str,
    latency_ms: float,
    trust_score: int,
    docs: List[Document],
    chunks: List[ChunkOut],
    top_k: int,
    model_name: str,
) -> None:
    """
    Append a single RAG run to data/run_logs.jsonl as one JSON line.
    This will be used later for analytics + fine-tuning.
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Build a serializable representation of retrieved docs
    retrieved = []
    n = min(len(docs), len(chunks))  # safeguard
    for i in range(n):
        d = docs[i]
        ch = chunks[i]
        retrieved.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "chunk": d.metadata.get("chunk", 0),
                "doc_id": d.metadata.get("id"),
                "distance": d.metadata.get("distance"),
                "relevance": ch.relevance,
                "text": d.page_content,
            }
        )

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "query": query,
        "answer": answer,
        "latency_ms": latency_ms,
        "trust_score": trust_score,
        "top_k": top_k,
        "model": model_name,
        # To be filled later when you label runs for fine-tuning
        "label": None,       # e.g. "good", "mixed", "off_topic", "no_evidence"
        "eval_notes": None,
        "retrieved": retrieved,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
# ---------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/rebuild")
def rebuild_index():
    pipeline.rebuild_index()
    return {"status": "ok", "message": "Index rebuilt successfully."}


@app.post("/api/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    q = payload.query.strip()
    if not q:
        return QueryResponse(
            answer="Please provide a non-empty question.",
            latency_ms=0.0,
            trust_score=0,
            chunks=[],
        )

    t0 = time.time()
    docs = pipeline.retrieve(q, k=payload.top_k)
    answer = pipeline.generate(q, docs)
    latency_ms = (time.time() - t0) * 1000.0

    trust_score = compute_trust_score(docs)

    chunks: List[ChunkOut] = []
    for d in docs:
        raw_distance = d.metadata.get("distance")
        distance_val: Optional[float] = float(raw_distance) if raw_distance is not None else None
        label = relevance_label(distance_val)

        chunks.append(
            ChunkOut(
                source=d.metadata.get("source", "unknown"),
                chunk=int(d.metadata.get("chunk", 0)),
                text=d.page_content,
                distance=distance_val,
                relevance=label,
            )
        )

    # ---------------------------------------------------------
    # STEP 3: log this run to run_logs.jsonl
    # ---------------------------------------------------------
    try:
        log_run_to_file(
            query=q,
            answer=answer,
            latency_ms=latency_ms,
            trust_score=trust_score,
            docs=docs,
            chunks=chunks,
            top_k=payload.top_k,
            model_name=pipeline.llm_model,
        )
    except Exception as e:
        print(f"[WARN] Failed to log run: {e}")
    # ---------------------------------------------------------

    return QueryResponse(
        answer=answer,
        latency_ms=latency_ms,
        trust_score=trust_score,
        chunks=chunks,
    )
