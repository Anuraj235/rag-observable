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

    Now supports:
      - base + fine-tuned model
      - toggle-ready API via `use_finetuned` / `force_model`
    """

    def __init__(
        self,
        data_dir: str,
        persist_dir: str,
        embed_model: str = "all-MiniLM-L6-v2",
        llm_base_model: Optional[str] = None,
        llm_finetuned_model: Optional[str] = None,
        use_finetuned_by_default: bool = True,
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

        # ---------- LLM model names ----------

        # Base model: explicit arg -> env -> legacy LLM_MODEL -> default
        self.base_model: str = (
            llm_base_model
            or os.getenv("LLM_BASE_MODEL")
            or os.getenv("LLM_MODEL")
            or "gpt-4o-mini-2024-07-18"
        )

        # Fine-tuned model: explicit arg -> env -> None
        self.ft_model: Optional[str] = (
            llm_finetuned_model
            or os.getenv("FINE_TUNED_MODEL")  # e.g. ft:gpt-4o-mini-2024-07-18:anuraj::CgLG9sx0
        )

        # Default behavior when caller doesn't specify
        self.use_finetuned_by_default: bool = use_finetuned_by_default

        # System prompt aligned with what you used during fine-tuning
        self.answer_system_prompt: str = (
            "You are a careful instructor for a Retrieval-Augmented Generation (RAG) system. "
            "Explain concepts clearly and step by step using ONLY the provided context. "
            "If something is not supported by the context, explicitly say you don't have enough information. "
            "First write a short 1-2 sentence summary, then a few concise bullet points. "
            "Always include inline citations like [1], [2] that refer to the provided context items."
        )

        # ---------- API keys & client ----------
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._hf_key = os.getenv("HUGGINGFACE_API_KEY")  # reserved if you add HF later

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

    # ---------------------- Prompt building ----------------------

    def _build_context_and_sources(self, docs: List[Document]) -> Tuple[str, str]:
        """
        Build:
          - context_block: numbered chunks with [1], [2] etc for the model
          - sources_block: human-readable citation list we append under the answer
        """
        numbered_lines = []
        for idx, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            chunk = d.metadata.get("chunk", 0)
            numbered_lines.append(f"[{idx}] ({src}#{chunk}) {d.page_content}")

        context_block = "\n\n".join(numbered_lines)

        sources_block = "\n".join(
            f"[{i}] {d.metadata.get('source', 'unknown')} (chunk {d.metadata.get('chunk', 0)})"
            for i, d in enumerate(docs, start=1)
        )

        return context_block, sources_block

    def _build_messages(self, query: str, docs: List[Document]):
        """
        Build messages for the Responses API, aligned with fine-tune format.
        Returns (messages, sources_block).
        """
        context_block, sources_block = self._build_context_and_sources(docs)

        user_content = (
            f"Question: {query}\n\n"
            f"Context items:\n\n"
            f"{context_block}"
        )

        messages = [
            {"role": "system", "content": self.answer_system_prompt},
            {"role": "user", "content": user_content},
        ]

        return messages, sources_block

    # ---------------------- Generation ----------------------

    def _generate_with_openai(
        self,
        query: str,
        docs: List[Document],
        model_name: str,
    ) -> Optional[str]:
        """
        Use OpenAI Responses API with the given model.
        Returns answer text, or None on any error.
        """
        if not self._openai_client:
            return None

        messages, sources_block = self._build_messages(query, docs)

        try:
            resp = self._openai_client.responses.create(
                model=model_name,
                input=messages,
                temperature=0.2,
            )

            # Grab the first text part of the first output
            content_items = resp.output[0].content
            text_parts: List[str] = []

            for item in content_items:
                # Newer SDK exposes `.text`; be defensive in case it's dict-like
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    text_parts.append(text)

            if not text_parts:
                return None

            answer = "\n".join(part.strip() for part in text_parts if part.strip())

            if sources_block:
                answer += f"\n\nSources:\n{sources_block}"

            return answer.strip()

        except Exception:
            # fall back to baseline / other model
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

    def generate(
        self,
        query: str,
        docs: List[Document],
        use_finetuned: Optional[bool] = None,
        force_model: Optional[str] = None,
    ) -> str:
        """
        Main entry point for answering.

        Args:
            query: user question
            docs: retrieved Document list
            use_finetuned:
                - True  => prefer fine-tuned model (if available)
                - False => force base model
                - None  => use `self.use_finetuned_by_default`
            force_model:
                - If set, override everything and call that exact model name.

        Safety fallbacks:
            1. Try primary (FT or base).
            2. If primary was FT and fails, try base.
            3. If all LLM calls fail or client missing, return baseline answer.
        """
        # Determine primary model
        if force_model:
            primary_model = force_model
        else:
            use_ft = self.use_finetuned_by_default if use_finetuned is None else use_finetuned
            if use_ft and self.ft_model:
                primary_model = self.ft_model
            else:
                primary_model = self.base_model

        # 1) Try primary model (if we have an OpenAI client)
        answer: Optional[str] = None
        if self._openai_client and primary_model:
            answer = self._generate_with_openai(query, docs, primary_model)

        # 2) If that failed AND it was the fine-tuned model, try falling back to base
        if (
            answer is None
            and self._openai_client
            and self.base_model
            and primary_model != self.base_model
        ):
            answer = self._generate_with_openai(query, docs, self.base_model)

        # 3) If everything failed, use baseline fallback
        if answer is None:
            return self._baseline_answer(query, docs)

        return answer
