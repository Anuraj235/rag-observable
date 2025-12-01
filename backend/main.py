import time
import json
import uuid
from datetime import datetime
import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from rag_pipeline import RAGPipeline, Document  # assumes rag_pipeline.py is in PYTHONPATH


# ---------------------------------------------------------
# Request/response models for main query endpoint
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    use_finetuned: bool = True  # 🔹 frontend toggle


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
    model: Optional[str] = None  # 🔹 which model actually answered


# ---------------------------------------------------------
# Run history models (for /api/runs and export)
# ---------------------------------------------------------
class RetrievedChunkRecord(BaseModel):
    source: str
    chunk: int
    doc_id: Optional[str] = None
    distance: Optional[float] = None
    relevance: Optional[str] = None
    text: str


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    query: str
    answer: str
    latency_ms: float
    trust_score: int
    top_k: int
    model: Optional[str] = None
    label: Optional[str] = None      # "good", "mixed", "off_topic", "no_evidence"
    eval_notes: Optional[str] = None
    retrieved: List[RetrievedChunkRecord]


# Label payload for PATCH
class LabelRequest(BaseModel):
    label: str              # "good", "mixed", "off_topic", "no_evidence"
    notes: Optional[str] = None


VALID_LABELS = {"good", "mixed", "off_topic", "no_evidence"}


# ---------------------------------------------------------
# App + logging setup
# ---------------------------------------------------------
app = FastAPI(title="Faithful RAG API")

LOG_DIR = "data"
LOG_PATH = os.path.join(LOG_DIR, "run_logs.jsonl")
DATASET_PATH = os.path.join(LOG_DIR, "fine_tune_dataset.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Global RAG pipeline instance
# ---------------------------------------------------------
pipeline = RAGPipeline(
    data_dir="data",
    persist_dir=".chromadb",
    embed_model="all-MiniLM-L6-v2",
)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
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


def model_name_for_run(use_finetuned: bool) -> str:
    """
    Decide which model name to log + return.
    Falls back safely if attributes are missing.
    """
    ft = getattr(pipeline, "ft_model", None)
    base = getattr(pipeline, "base_model", None) or getattr(
        pipeline, "llm_model", "gpt-4o"
    )

    if use_finetuned and ft:
        return ft
    return base


# ---------------------- Logging helpers ----------------------
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

    record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "query": query,
        "answer": answer,
        "latency_ms": latency_ms,
        "trust_score": trust_score,
        "top_k": top_k,
        "model": model_name,
        "label": None,       # to be filled later via PATCH
        "eval_notes": None,
        "retrieved": retrieved,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_runs_from_file(limit: int = 50) -> List[dict]:
    """
    Read the most recent `limit` runs from run_logs.jsonl.
    Returns a list of dicts, newest first.
    """
    if not os.path.exists(LOG_PATH):
        return []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return []

    selected = lines[-limit:]
    records: List[dict] = []

    for line in reversed(selected):  # newest first
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records


def read_all_runs() -> List[dict]:
    """Read all runs from the log file (used for export / stats)."""
    if not os.path.exists(LOG_PATH):
        return []
    records: List[dict] = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------- Log-aware re-ranker helpers ----------------------
def compute_source_quality_scores() -> Dict[str, float]:
    """
    Very simple 're-ranker model' built from logs:
    - sources with many 'good' labels get a boost
    - 'off_topic' reduces score
    - 'mixed' is neutral-ish

    This is a placeholder for a real fine-tuned critic model.
    """
    runs = read_all_runs()
    if not runs:
        return {}

    stats: Dict[str, Dict[str, int]] = {}

    for run in runs:
        label = run.get("label")
        if label not in VALID_LABELS:
            continue
        for ch in run.get("retrieved", []):
            src = ch.get("source", "unknown")
            if src not in stats:
                stats[src] = {"good": 0, "mixed": 0, "off_topic": 0}
            if label == "good":
                stats[src]["good"] += 1
            elif label == "mixed":
                stats[src]["mixed"] += 1
            elif label == "off_topic":
                stats[src]["off_topic"] += 1

    scores: Dict[str, float] = {}
    for src, counts in stats.items():
        # Simple scoring: good +1, mixed +0.3, off_topic -1
        s = (
            counts["good"] * 1.0
            + counts["mixed"] * 0.3
            - counts["off_topic"] * 1.0
        )
        scores[src] = s

    return scores


# ---------------------------------------------------------
# Basic health + rebuild
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/rebuild")
def rebuild_index():
    pipeline.rebuild_index()
    return {"status": "ok", "message": "Index rebuilt successfully."}


# ---------------------------------------------------------
# Main RAG query (baseline ranking)
# ---------------------------------------------------------
@app.post("/api/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    q = payload.query.strip()
    if not q:
        return QueryResponse(
            answer="Please provide a non-empty question.",
            latency_ms=0.0,
            trust_score=0,
            chunks=[],
            model=None,
        )

    t0 = time.time()
    docs = pipeline.retrieve(q, k=payload.top_k)
    answer = pipeline.generate(q, docs, use_finetuned=payload.use_finetuned)
    latency_ms = (time.time() - t0) * 1000.0

    trust_score = compute_trust_score(docs)
    model_used = model_name_for_run(payload.use_finetuned)

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

    try:
        log_run_to_file(
            query=q,
            answer=answer,
            latency_ms=latency_ms,
            trust_score=trust_score,
            docs=docs,
            chunks=chunks,
            top_k=payload.top_k,
            model_name=model_used,
        )
    except Exception as e:
        print(f"[WARN] Failed to log run: {e}")

    return QueryResponse(
        answer=answer,
        latency_ms=latency_ms,
        trust_score=trust_score,
        chunks=chunks,
        model=model_used,
    )


# ---------------------------------------------------------
# RAG query with simple log-aware re-ranking
# ---------------------------------------------------------
@app.post("/api/query_rerank", response_model=QueryResponse)
def query_rag_rerank(payload: QueryRequest):
    """
    Demo endpoint that uses historical labels to re-rank retrieved chunks.
    This is a placeholder for a fine-tuned critic model.
    """
    q = payload.query.strip()
    if not q:
        return QueryResponse(
            answer="Please provide a non-empty question.",
            latency_ms=0.0,
            trust_score=0,
            chunks=[],
            model=None,
        )

    t0 = time.time()
    docs = pipeline.retrieve(q, k=payload.top_k)

    # Get source quality scores from labeled history
    source_scores = compute_source_quality_scores()

    # Build scores per doc: lower distance + higher source score = better
    scored_docs: List[tuple[float, Document]] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        dist = d.metadata.get("distance") or 0.0
        src_score = source_scores.get(src, 0.0)
        # Combine: smaller distance is better, so subtract it; add source score.
        combined = src_score - float(dist)
        scored_docs.append((combined, d))

    # Sort by combined score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    reranked_docs = [d for _, d in scored_docs]

    answer = pipeline.generate(q, reranked_docs, use_finetuned=payload.use_finetuned)
    latency_ms = (time.time() - t0) * 1000.0
    trust_score = compute_trust_score(reranked_docs)
    model_used = model_name_for_run(payload.use_finetuned)

    chunks: List[ChunkOut] = []
    for d in reranked_docs:
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

    try:
        log_run_to_file(
            query=q,
            answer=answer,
            latency_ms=latency_ms,
            trust_score=trust_score,
            docs=reranked_docs,
            chunks=chunks,
            top_k=payload.top_k,
            model_name=model_used,
        )
    except Exception as e:
        print(f"[WARN] Failed to log reranked run: {e}")

    return QueryResponse(
        answer=answer,
        latency_ms=latency_ms,
        trust_score=trust_score,
        chunks=chunks,
        model=model_used,
    )


# ---------------------------------------------------------
# Run history + labeling
# ---------------------------------------------------------
@app.get("/api/runs", response_model=List[RunRecord])
def list_runs(limit: int = 50):
    """
    Return the most recent `limit` runs, newest first.
    """
    records = read_runs_from_file(limit=limit)
    return [RunRecord(**rec) for rec in records]


@app.patch("/api/runs/{run_id}")
def update_run_label(run_id: str, payload: LabelRequest):
    """
    Update the label + notes of a specific run inside run_logs.jsonl.
    """
    if payload.label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail="Invalid label value")

    if not os.path.exists(LOG_PATH):
        raise HTTPException(status_code=404, detail="No run logs found")

    updated = False
    new_lines: List[str] = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("run_id") == run_id:
                record["label"] = payload.label
                record["eval_notes"] = payload.notes
                updated = True

            new_lines.append(json.dumps(record, ensure_ascii=False))

    if not updated:
        raise HTTPException(status_code=404, detail="Run ID not found")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for ln in new_lines:
            f.write(ln + "\n")

    return {"status": "ok", "run_id": run_id, "label": payload.label}


# ---------------------------------------------------------
# Export dataset for OpenAI fine-tuning
# ---------------------------------------------------------
@app.get("/api/export-dataset", response_class=PlainTextResponse)
def export_dataset():
    """
    Build an OpenAI fine-tune JSONL dataset from labeled runs.
    Each line is a chat-style example:
    {
      "messages": [
        {"role": "system", ...},
        {"role": "user", ...},
        {"role": "assistant", ...}
      ],
      "metadata": {...}
    }
    Only runs with a non-null label are included.
    """
    runs = read_all_runs()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found")

    lines: List[str] = []
    for run in runs:
        label = run.get("label")
        if label not in VALID_LABELS:
            continue  # skip unlabeled runs

        # Build context from retrieved chunks
        retrieved = run.get("retrieved", [])
        numbered_ctx_parts = []
        for idx, ch in enumerate(retrieved, start=1):
            src = ch.get("source", "unknown")
            chunk_id = ch.get("chunk", 0)
            text = ch.get("text", "")
            numbered_ctx_parts.append(f"[{idx}] ({src}#{chunk_id}) {text}")
        context_block = "\n\n".join(numbered_ctx_parts)

        system_msg = {
            "role": "system",
            "content": (
                "You are a careful instructor for a Retrieval-Augmented Generation (RAG) system. "
                "Explain concepts clearly and step by step using ONLY the provided context. "
                "If something is not supported by the context, explicitly say you don't have enough information. "
                "First write a short 1-2 sentence summary, then a few concise bullet points. "
                "Always include inline citations like [1], [2] that refer to the provided context items."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Question: {run.get('query', '')}\n\n"
                f"Context items:\n\n{context_block}"
            ),
        }
        assistant_msg = {
            "role": "assistant",
            "content": run.get("answer", ""),
        }

        dataset_record = {
            "messages": [system_msg, user_msg, assistant_msg],
            "metadata": {
                "run_id": run.get("run_id"),
                "label": label,
                "trust_score": run.get("trust_score"),
                "top_k": run.get("top_k"),
                "model": run.get("model"),
            },
        }

        lines.append(json.dumps(dataset_record, ensure_ascii=False))

    if not lines:
        raise HTTPException(status_code=400, detail="No labeled runs to export")

    # Optionally write to disk as well
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    # Return as plain text so frontend can download/save
    return "\n".join(lines)


# ---------------------------------------------------------
# Delete/reset logs
# ---------------------------------------------------------
@app.delete("/api/runs")
def clear_runs():
    """
    Delete/reset the run_logs.jsonl file.
    """
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    return {"status": "ok", "message": "Run logs cleared"}
