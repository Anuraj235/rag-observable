import os
import glob
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- LLM (OpenAI) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # fallback if lib not installed yet


@dataclass
class Document:
    """
    Simple container used by the UI and pipeline.
    """
    page_content: str
    metadata: Dict


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      - builds a Chroma index from local text files
      - retrieves top-k chunks for a query
      - calls an LLM (OpenAI) to generate an answer grounded in those chunks
    """

    def __init__(
        self,
        data_dir: str,
        persist_dir: str,
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
    ):
        load_dotenv()  # read .env once here

        self.data_dir = data_dir
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # Keep the model name so we can reuse it in rebuild_index()
        self.embed_model = embed_model

        # ---------- Embeddings ----------
        # Kept in case you want to use it directly later (e.g., custom eval)
        self.embedder = SentenceTransformer(self.embed_model)

        # ---------- Vector store (Chroma) ----------
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model
            ),
        )

        # Build index if empty
        if self.collection.count() == 0:
            self._build_index()

        # ---------- LLM ----------
        # Model name from param -> env -> default
        self.llm_model = (
            llm_model
            or os.getenv("LLM_MODEL")
            or "gpt-4o"
        )

        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._hf_key = os.getenv("HUGGINGFACE_API_KEY")  # reserved if you add HF later

        # Create OpenAI client if possible
        self._openai_client = None
        if OpenAI and self._openai_key:
            try:
                self._openai_client = OpenAI(api_key=self._openai_key)
            except Exception:
                # If something goes wrong, we just fall back to baseline answers.
                self._openai_client = None

    # ---------------------- Indexing ----------------------

    def _iter_source_files(self):
        """
        Yield all supported text-like files under data_dir.
        """
        exts = ("*.txt", "*.md")
        for ext in exts:
            for path in glob.glob(os.path.join(self.data_dir, ext)):
                yield path

    def _read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _chunk(self, text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
        """
        Very simple word-based chunker.
        chunk_size and overlap are in *words*, not tokens.
        """
        words = text.split()
        chunks: List[str] = []
        i = 0
        while i < len(words):
            chunk_words = words[i: i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += max(1, chunk_size - overlap)
        return chunks

    def _build_index(self) -> None:
        """
        Build the Chroma collection from scratch from files in data_dir.

        IMPORTANT: Chunks are now numbered *globally* across all files
        so chunk ids are unique (0..N-1) instead of restarting at 0 per file.
        """
        docs: List[str] = []
        ids: List[str] = []
        metas: List[Dict] = []

        global_chunk_idx: int = 0  # 🔹 global counter for chunk numbers

        for path in self._iter_source_files():
            text = self._read_file(path)
            base = os.path.basename(path)

            for chunk_text in self._chunk(text):
                docs.append(chunk_text)
                # Use global chunk index in both id and metadata
                ids.append(f"{base}::{global_chunk_idx}")
                metas.append({
                    "source": base,
                    "chunk": global_chunk_idx,
                })
                global_chunk_idx += 1

        if docs:
            self.collection.add(documents=docs, metadatas=metas, ids=ids)

    def rebuild_index(self) -> None:
        """
        Drop and rebuild the 'docs' collection using the same embedding model.
        Called when the user checks 'Rebuild index' in the UI.
        """
        try:
            self.client.delete_collection("docs")
        except Exception:
            # ok if it didn't exist
            pass

        # Recreate using the same embedding function/model name
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model
            ),
        )
        self._build_index()

    # ---------------------- Retrieval ----------------------

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve top-k chunks from Chroma for a given query.
        Also attaches similarity scores and ids into metadata for observability.
        """
        # Some Chroma versions don't allow "ids" in include; ids come back by default.
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs: List[Document] = []

        docs_raw = res.get("documents", [[]])[0]
        metas_raw = res.get("metadatas", [[]])[0]
        dists_raw = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs_raw)
        ids_raw = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs_raw)

        for doc_text, meta, dist, doc_id in zip(docs_raw, metas_raw, dists_raw, ids_raw):
            # Copy and enrich metadata so we don't mutate Chroma's internal objects
            meta = dict(meta) if meta is not None else {}
            if doc_id is not None:
                meta.setdefault("id", doc_id)
            if dist is not None:
                # distance is smaller = more similar (cosine space)
                meta.setdefault("distance", float(dist))
            docs.append(Document(page_content=doc_text, metadata=meta))

        return docs

    # ---------------------- Generation ----------------------

    def _build_prompt(self, query: str, docs: List[Document]) -> Tuple[Tuple[str, str], str]:
        """
        Build (system, user) messages and a human-readable sources block.
        Returns ((system, user), sources_block).
        """
        numbered = []
        for idx, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            chunk = d.metadata.get("chunk", 0)
            numbered.append(f"[{idx}] ({src}#{chunk}) {d.page_content}")

        context_block = "\n\n".join(numbered)
        sources_block = "\n".join(
            f"[{i}] {d.metadata.get('source', 'unknown')} (chunk {d.metadata.get('chunk', 0)})"
            for i, d in enumerate(docs, start=1)
        )

        system = (
            "You are a careful assistant. Answer ONLY using the provided context. "
            "If the answer isn't in the context, say you don't have enough information. "
            "Keep it concise and include inline citations like [1], [2] that refer to the context items."
        )

        user = (
            f"Question: {query}\n\n"
            f"Context items (each line is a different source chunk with its citation tag):\n\n"
            f"{context_block}\n\n"
            "Now write the answer with inline citations."
        )

        return (system, user), sources_block

    def _generate_with_openai(self, query: str, docs: List[Document]) -> Optional[str]:
        """
        Use OpenAI chat completions if configured; otherwise return None.
        """
        if not self._openai_client:
            return None

        (system, user), sources_block = self._build_prompt(query, docs)

        try:
            resp = self._openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
            if sources_block:
                answer += f"\n\nSources:\n{sources_block}"
            return answer
        except Exception:
            # fall back to baseline on any error
            return None

    def _baseline_answer(self, query: str, docs: List[Document]) -> str:
        """
        Very simple fallback "answer" if no LLM is available.
        """
        context = "\n\n".join(
            [f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)]
        )
        sources_block = "\n".join(
            f"[{i}] {d.metadata.get('source', 'unknown')} (chunk {d.metadata.get('chunk', 0)})"
            for i, d in enumerate(docs, start=1)
        )
        return (
            "Using only the provided context, here is a concise answer.\n\n"
            f"Question: {query}\n\n"
            f"Context (truncated):\n{context[:1200]}\n\n"
            f"Sources:\n{sources_block}"
        )

    def generate(self, query: str, docs: List[Document]) -> str:
        """
        Preferred path: OpenAI chat. If unavailable, uses baseline.
        """
        answer = self._generate_with_openai(query, docs)
        if answer is not None:
            return answer
        return self._baseline_answer(query, docs)
