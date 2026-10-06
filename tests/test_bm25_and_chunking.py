import pytest

from app.bm25 import BM25, tokenize
from app.chunking import chunk_document


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The manager approves the claims") == ["manager", "approv", "claim"]


def test_bm25_ranks_matching_document_first():
    bm = BM25()
    bm.add(["a", "b", "c"], ["annual leave days", "expense claims receipts", "password length rules"])
    assert bm.search("how many leave days", k=1)[0][0] == "a"


def test_bm25_respects_allowed_ids():
    bm = BM25()
    bm.add(["a", "b"], ["leave policy", "leave expenses"])
    assert [i for i, _ in bm.search("leave", allowed={"b"})] == ["b"]


def test_chunking_overlap_and_sections():
    text = "# Title\n" + " ".join(f"w{i}" for i in range(250))
    chunks = chunk_document(text, "doc.md", chunk_words=100, overlap=20)
    assert len(chunks) == 3
    assert all(c.section == "Title" for c in chunks)
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[1:21]  # overlap carried over


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_document("x", "d", chunk_words=10, overlap=10)
