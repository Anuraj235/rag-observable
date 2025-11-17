import os
import time
import textwrap
from typing import List, Dict
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline, Document
from metrics import precision_at_k, recall_at_k

load_dotenv()

st.set_page_config(page_title="Faithful & Observable RAG", layout="wide")

# --- cache last run outputs so metrics don't break before first Run ---
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_k" not in st.session_state:
    st.session_state.last_k = 3
if "eval_log" not in st.session_state:
    # list of dicts: one row per evaluated run
    st.session_state.eval_log = []

# Sidebar config
st.sidebar.title("RAG Controls")
k = st.sidebar.slider("Top-k retrieval", min_value=1, max_value=10, value=3)
rebuild = st.sidebar.checkbox("Rebuild index", value=False)
show_chunks = st.sidebar.checkbox("Show retrieved chunks", value=True)

st.title("🔍 Faithful & Observable RAG")
st.caption("Baseline → Observe → Adapt → Execute")

# Init pipeline (lazy)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(
        data_dir="data",
        persist_dir=".chromadb",
        embed_model="all-MiniLM-L6-v2",
    )

# Force a rebuild if the checkbox is checked
if rebuild:
    st.session_state.pipeline.rebuild_index()
    st.info("Index rebuilt.")

query = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g., What is this project about?"
)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # --- Run button handler: compute, then store results in session ---
    if st.button("Run") and query.strip():
        t0 = time.time()
        docs = st.session_state.pipeline.retrieve(query, k=k)
        answer = st.session_state.pipeline.generate(query, docs)
        latency = (time.time() - t0) * 1000.0

        # store for right column + persistence across reruns
        st.session_state.last_latency_ms = latency
        st.session_state.last_answer = answer
        st.session_state.last_docs = docs
        st.session_state.last_query = query
        st.session_state.last_k = k

        # clear any old relevance labels from previous runs
        for key in list(st.session_state.keys()):
            if key.startswith("rel_label_"):
                del st.session_state[key]

    # --- Display from session state so UI persists across reruns ---
    if st.session_state.last_answer:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)

        if show_chunks and st.session_state.last_docs:
            st.subheader("Retrieved Chunks")

            for i, d in enumerate(st.session_state.last_docs, 1):
                chunk_label = d.metadata.get("source", "unknown")
                with st.expander(f"Chunk {i}: {chunk_label}"):
                    st.code(d.page_content)

                    # Relevance labeling UI for this chunk
                    label_key = f"rel_label_{i}"
                    current = st.session_state.get(label_key, "Unlabeled")

                    st.radio(
                        "Is this chunk relevant to the question?",
                        ["Unlabeled", "Relevant", "Not relevant"],
                        index=["Unlabeled", "Relevant", "Not relevant"].index(current)
                        if current in ["Unlabeled", "Relevant", "Not relevant"]
                        else 0,
                        key=label_key,
                        horizontal=True,
                    )
    else:
        st.info("Enter a question and press Run.")

with col2:
    st.subheader("Metrics")

    # Build rel_labels list from the per-chunk radios for the LAST run
    rel_labels: List[bool] = []
    for i, _ in enumerate(st.session_state.last_docs, 1):
        label = st.session_state.get(f"rel_label_{i}", "Unlabeled")
        rel_labels.append(label == "Relevant")  # True only if explicitly marked Relevant

    # Use the k that was used during retrieval, not necessarily the current slider
    k_eval = st.session_state.get("last_k", k)
    if st.session_state.last_docs:
        k_eval = min(k_eval, len(st.session_state.last_docs))

    if rel_labels:
        p_at_k = precision_at_k(rel_labels, k_eval)
        r_at_k = recall_at_k(rel_labels, k_eval)
    else:
        p_at_k = 0.0
        r_at_k = 0.0

    lat = st.session_state.last_latency_ms
    st.metric("Latency (ms)", value=f"{lat:.0f}" if isinstance(lat, (int, float)) else "-")
    st.metric("Precision@k", value=f"{p_at_k:.2f}")
    st.metric("Recall@k", value=f"{r_at_k:.2f}")

    st.caption(f"Evaluated with k = {k_eval}")

    st.markdown("---")
    st.markdown("**Evaluation log**")

    # Button to save current evaluation row
    if st.button("Save this run to log"):
        if st.session_state.last_docs:
            st.session_state.eval_log.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "query": st.session_state.last_query,
                    "k": k_eval,
                    "latency_ms": float(lat) if isinstance(lat, (int, float)) else None,
                    "precision_at_k": p_at_k,
                    "recall_at_k": r_at_k,
                    "num_retrieved": len(st.session_state.last_docs),
                }
            )
            st.success("Saved current run to evaluation log.")
        else:
            st.warning("Run a query first before saving to the log.")

    # If we have any logged rows, show table, chart, and CSV download
    if st.session_state.eval_log:
        df = pd.DataFrame(st.session_state.eval_log)
        st.dataframe(df, use_container_width=True)

        # Simple precision/recall chart across runs
        if len(df) > 1:
            st.line_chart(df[["precision_at_k", "recall_at_k"]])

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download evaluation CSV",
            data=csv,
            file_name="rag_eval_log.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Tip: Start with MiniLM; upgrade embeddings & prompts as you iterate. "
    "Log traces for observation, then tighten prompts for faithfulness."
)
