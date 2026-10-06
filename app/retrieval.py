"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion, with metadata filters."""
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25
from .chunking import chunk_document
from .embeddings import Embedder
from .vectorstore import VectorStore

STRATEGIES = ("vector", "bm25", "hybrid")


@dataclass
class Hit:
    id: str
    text: str
    source: str
    section: str
    score: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Combine ranked id lists. Robust to the very different score scales of BM25 and cosine."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, chunk_words: int = 120, chunk_overlap: int = 30):
        self.store, self.embedder = store, embedder
        self.chunk_words, self.chunk_overlap = chunk_words, chunk_overlap
        self.texts: dict[str, str] = {}
        self.metas: dict[str, dict] = {}
        self.bm25 = BM25()
        self._rebuild_from_store()

    # ---------- indexing ----------
    def _rebuild_from_store(self) -> None:
        ids, texts, metas = self.store.all_records()
        self.texts = dict(zip(ids, texts))
        self.metas = dict(zip(ids, metas))
        self.bm25 = BM25()
        self.bm25.add(ids, texts)

    def add_document(self, text: str, source: str) -> int:
        if source in self.sources():
            self.delete_document(source)  # re-upload replaces the old version
        chunks = chunk_document(text, source, self.chunk_words, self.chunk_overlap)
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metas = [{"source": c.source, "section": c.section, "chunk_index": c.chunk_index} for c in chunks]
        self.store.add(ids, texts, self.embedder.embed(texts), metas)
        self.texts.update(zip(ids, texts))
        self.metas.update(zip(ids, metas))
        self.bm25.add(ids, texts)
        return len(chunks)

    def add_file(self, path: str | Path) -> int:
        path = Path(path)
        return self.add_document(extract_text(path), path.name)

    def delete_document(self, source: str) -> None:
        self.store.delete_source(source)
        self._rebuild_from_store()

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.metas.values():
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    # ---------- search ----------
    def search(self, query: str, k: int = 5, strategy: str = "hybrid", source: str | None = None) -> list[Hit]:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        where = {"source": source} if source else None
        allowed = {i for i, m in self.metas.items() if m.get("source") == source} if source else None
        pool = max(k * 4, 20)

        if strategy == "vector":
            ranked = self.store.query(self.embedder.embed([query])[0], k, where)
        elif strategy == "bm25":
            ranked = self.bm25.search(query, k, allowed)
        else:
            dense = [i for i, _ in self.store.query(self.embedder.embed([query])[0], pool, where)]
            sparse = [i for i, _ in self.bm25.search(query, pool, allowed)]
            ranked = reciprocal_rank_fusion([dense, sparse])[:k]

        return [
            Hit(
                id=i,
                text=self.texts[i],
                source=self.metas[i]["source"],
                section=self.metas[i].get("section", ""),
                score=round(float(s), 4),
            )
            for i, s in ranked
            if i in self.texts
        ]


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")
