from app.embeddings import HashEmbedder
from app.retrieval import HybridRetriever, reciprocal_rank_fusion
from app.vectorstore import InMemoryStore


def make_retriever():
    r = HybridRetriever(InMemoryStore(), HashEmbedder())
    r.add_document("# Leave\nStaff get 20 days of annual leave.", "leave.md")
    r.add_document("# Security\nPasswords must be 14 characters long.", "security.md")
    return r


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert {fused[0][0], fused[1][0]} == {"a", "b"}
    assert fused[-1][0] in {"c", "d"}


def test_each_strategy_finds_the_right_document():
    r = make_retriever()
    for strategy in ("vector", "bm25", "hybrid"):
        assert r.search("password characters", k=1, strategy=strategy)[0].source == "security.md"


def test_metadata_filter_restricts_source():
    r = make_retriever()
    hits = r.search("password", k=5, source="leave.md")
    assert hits and all(h.source == "leave.md" for h in hits)


def test_reupload_replaces_document():
    r = make_retriever()
    r.add_document("# Leave\nStaff now get 25 days.", "leave.md")
    assert r.sources()["leave.md"] == 1
    assert "25 days" in r.search("annual leave days", k=1, source="leave.md")[0].text


def test_delete_document():
    r = make_retriever()
    r.delete_document("leave.md")
    assert "leave.md" not in r.sources()
    assert all(h.source != "leave.md" for h in r.search("leave", k=5))
