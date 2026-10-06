"""Quality gate: CI fails if retrieval quality on the labelled set drops below a floor."""
from app.retrieval import STRATEGIES
from eval.run_eval import build_retriever, groundedness, load_eval, retrieval_metrics, ROOT


def test_hybrid_retrieval_quality_floor():
    items = load_eval(ROOT / "eval/eval_set.jsonl")
    r = build_retriever()
    recall = sum(
        retrieval_metrics(r.search(i["question"], k=3, strategy="hybrid"), set(i["relevant_sources"]),
                          i["expected_keywords"])["recall"]
        for i in items
    ) / len(items)
    assert recall >= 0.9


def test_groundedness_detects_unsupported_sentences():
    ctx = ["Staff get 20 days of annual leave."]
    assert groundedness("Staff get 20 days of annual leave.", ctx) == 1.0
    assert groundedness("Staff get 20 days of annual leave. Bonuses are paid in March.", ctx) == 0.5


def test_strategies_constant():
    assert set(STRATEGIES) == {"vector", "bm25", "hybrid"}
