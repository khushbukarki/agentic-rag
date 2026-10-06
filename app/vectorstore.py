"""Vector stores: Chroma (persistent) and a NumPy in-memory store with the same interface."""
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def add(self, ids, texts, embeddings, metadatas) -> None: ...
    def query(self, embedding, k: int, where: dict | None = None) -> list[tuple[str, float]]: ...
    def all_records(self) -> tuple[list[str], list[str], list[dict]]: ...
    def delete_source(self, source: str) -> None: ...


def _matches(meta: dict, where: dict | None) -> bool:
    return not where or all(meta.get(k) == v for k, v in where.items())


class InMemoryStore:
    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []
        self.vecs: list[np.ndarray] = []

    def add(self, ids, texts, embeddings, metadatas):
        self.ids += list(ids)
        self.texts += list(texts)
        self.metas += list(metadatas)
        self.vecs += [np.asarray(e, dtype=np.float32) for e in embeddings]

    def query(self, embedding, k, where=None):
        if not self.ids:
            return []
        mat = np.vstack(self.vecs)
        sims = mat @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(-sims)
        out = []
        for i in order:
            if _matches(self.metas[i], where):
                out.append((self.ids[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    def all_records(self):
        return list(self.ids), list(self.texts), list(self.metas)

    def delete_source(self, source):
        keep = [i for i, m in enumerate(self.metas) if m.get("source") != source]
        self.ids = [self.ids[i] for i in keep]
        self.texts = [self.texts[i] for i in keep]
        self.metas = [self.metas[i] for i in keep]
        self.vecs = [self.vecs[i] for i in keep]


class ChromaStore:
    def __init__(self, persist_dir: str, collection: str = "documents"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids, texts, embeddings, metadatas):
        self.col.upsert(
            ids=list(ids),
            documents=list(texts),
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=list(metadatas),
        )

    def query(self, embedding, k, where=None):
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=min(k, self.col.count()),
            where=where or None,
        )
        # cosine distance -> similarity
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def all_records(self):
        res = self.col.get(include=["documents", "metadatas"])
        return res["ids"], res["documents"], res["metadatas"]

    def delete_source(self, source):
        self.col.delete(where={"source": source})


def build_store(kind: str, persist_dir: str) -> VectorStore:
    return InMemoryStore() if kind == "memory" else ChromaStore(persist_dir)
