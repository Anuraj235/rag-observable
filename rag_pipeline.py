import os
import glob
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

@dataclass
class Document:
    page_content: str
    metadata: Dict

class RAGPipeline:
    def __init__(self, data_dir: str, persist_dir: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # Embeddings
        self.embedder = SentenceTransformer(embed_model)

        # Vector store (ChromaDB)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        )

        # Build index if empty
        if self.collection.count() == 0:
            self._build_index()

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
            chunk_words = words[i:i+chunk_size]
            chunks.append(" ".join(chunk_words))
            i += (chunk_size - overlap)
        return chunks

    def _build_index(self):
        docs = []
        ids = []
        metas = []
        for path in self._iter_source_files():
            text = self._read_file(path)
            for i, chunk in enumerate(self._chunk(text)):
                docs.append(chunk)
                ids.append(f"{os.path.basename(path)}::{i}")
                metas.append({"source": os.path.basename(path), "chunk": i})
        if docs:
            self.collection.add(documents=docs, metadatas=metas, ids=ids)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        res = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        # Chroma returns lists; pick first
        for doc, meta in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]):
            docs.append(Document(page_content=doc, metadata=meta))
        return docs

    def generate(self, query: str, docs: List[Document]) -> str:
        # Minimal baseline generator that *only* cites provided docs.
        # Swap this with an LLM call of your choice.
        context = "\n\n".join([d.page_content for d in docs])
        answer = f"""
Using only the provided context, here is a concise answer.

Question: {query}

Context (truncated): 
{context[:1000]}
"""
        return answer.strip()
