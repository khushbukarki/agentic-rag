"""A small, dependency-free Okapi BM25 implementation."""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by",
    "with", "at", "as", "it", "this", "that", "what", "how", "do", "does", "i", "can", "my",
}


def stem(word: str) -> str:
    """Very light suffix stripping so approve/approves/approved/approving match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()

    def add(self, ids: list[str], texts: list[str]) -> None:
        for doc_id, text in zip(ids, texts):
            toks = tokenize(text)
            tf = Counter(toks)
            self.ids.append(doc_id)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())

    @property
    def avgdl(self) -> float:
        return sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q or not self.ids:
            return []
        avgdl = self.avgdl or 1.0
        scores = []
        for doc_id, tf, dl in zip(self.ids, self.doc_tf, self.doc_len):
            if allowed is not None and doc_id not in allowed:
                continue
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if f:
                    s += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
