import os
import glob
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

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
    page_content: str
    metadata: Dict


class RAGPipeline:
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
                self._openai_client = None  # fall back to baseline if init fails

    # ---------------------- Indexing ----------------------

    def _iter_source_files(self):
        exts = ("*.txt", "*.md")
        for ext in exts:
            for path in glob.glob(os.path.join(self.data_dir, ext)):
                yield path

    def _read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _chunk(self, text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += max(1, chunk_size - overlap)
        return chunks

    def _build_index(self):
        docs, ids, metas = [], [], []
        for path in self._iter_source_files():
            text = self._read_file(path)
            base = os.path.basename(path)
            for i, chunk in enumerate(self._chunk(text)):
                docs.append(chunk)
                ids.append(f"{base}::{i}")
                metas.append({"source": base, "chunk": i})
        if docs:
            self.collection.add(documents=docs, metadatas=metas, ids=ids)

    def rebuild_index(self):
        """Drop and rebuild the 'docs' collection using the same embedding model."""
        try:
            self.client.delete_collection("docs")
        except Exception:
            pass  # ok if it didn't exist

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
        res = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        for doc, meta in zip(
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
        ):
            docs.append(Document(page_content=doc, metadata=meta))
        return docs

    # ---------------------- Generation ----------------------

    def _build_prompt(self, query: str, docs: List[Document]) -> Tuple[str, str]:
        """
        Returns (prompt, sources_block)
        """
        numbered = []
        for idx, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            chunk = d.metadata.get("chunk", 0)
            numbered.append(f"[{idx}] ({src}#{chunk}) {d.page_content}")

        context_block = "\n\n".join(numbered)
        sources_block = "\n".join(
            f"[{i}] {d.metadata.get('source','unknown')} (chunk {d.metadata.get('chunk',0)})"
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

        prompt = (system, user)
        return prompt, sources_block

    def _generate_with_openai(self, query: str, docs: List[Document]) -> Optional[str]:
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
        # Minimal fallback if no LLM
        context = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])
        sources_block = "\n".join(
            f"[{i}] {d.metadata.get('source','unknown')} (chunk {d.metadata.get('chunk',0)})"
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
