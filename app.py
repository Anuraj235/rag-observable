import os
import time
import textwrap
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline, Document
from metrics import precision_at_k, recall_at_k

load_dotenv()

st.set_page_config(page_title="Faithful & Observable RAG", layout="wide")

# Sidebar config
st.sidebar.title("RAG Controls")
k = st.sidebar.slider("Top-k retrieval", min_value=1, max_value=10, value=3)
rebuild = st.sidebar.checkbox("Rebuild index", value=False)
show_chunks = st.sidebar.checkbox("Show retrieved chunks", value=True)

st.title("🔍 Faithful & Observable RAG")
st.caption("Baseline → Observe → Adapt → Execute")

# Init pipeline (lazy)
if "pipeline" not in st.session_state or rebuild:
    st.session_state.pipeline = RAGPipeline(
        data_dir="data",
        persist_dir=".chromadb",
        embed_model="all-MiniLM-L6-v2",
    )
    if rebuild:
        st.info("Index rebuilt.")

query = st.text_input("Ask a question about your documents:", placeholder="e.g., What is this project about?")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    if st.button("Run") and query.strip():
        t0 = time.time()
        docs = st.session_state.pipeline.retrieve(query, k=k)
        answer = st.session_state.pipeline.generate(query, docs)
        latency = (time.time() - t0) * 1000.0

        st.subheader("Answer")
        st.write(answer)

        if show_chunks:
            st.subheader("Retrieved Chunks")
            for i, d in enumerate(docs, 1):
                with st.expander(f"Chunk {i}: {d.metadata.get('source','unknown')}"):
                    st.code(d.page_content)

with col2:
    st.subheader("Metrics (demo)")
    # Placeholder demo metrics — swap with real labels later
    rel_labels = st.session_state.get("rel_labels", [])
    if st.button("Mark retrieved as relevant (toy)"):
        # In a real app, collect a list of booleans from the UI
        rel_labels = [True] + [False]*(max(0,k-1))
        st.session_state["rel_labels"] = rel_labels

    p_at_k = precision_at_k(rel_labels, k) if rel_labels else 0.0
    r_at_k = recall_at_k(rel_labels, k) if rel_labels else 0.0

    st.metric("Latency (ms)", value=f"{latency:.0f}" if query else "-")
    st.metric("Precision@k", value=f"{p_at_k:.2f}")
    st.metric("Recall@k", value=f"{r_at_k:.2f}")
    
st.markdown("---")
st.caption("Tip: Start with MiniLM; upgrade embeddings & prompts as you iterate. Log traces for observation, then tighten prompts for faithfulness.")
