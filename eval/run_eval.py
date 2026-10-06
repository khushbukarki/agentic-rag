"""Retrieval-evaluation harness.

Compares vector, BM25 and hybrid retrieval on a labelled question set, reporting:
  * precision@k   - share of retrieved chunks that come from a relevant document
  * recall@k      - share of relevant documents that appear in the top k
  * hit rate      - share of questions whose top k contains the expected answer text
  * MRR           - mean reciprocal rank of the first relevant chunk
  * groundedness  - (with --answers) share of answer sentences supported by retrieved text

Usage:
    python -m eval.run_eval                       # real embeddings (sentence-transformers)
    EMBEDDER=hash python -m eval.run_eval         # fast, no model download
    python -m eval.run_eval --answers             # also score generated answers for groundedness
"""
import argparse
import json
import os
import re
from pathlib import Path

from app.agent import DocumentAgent
from app.bm25 import tokenize
from app.embeddings import build_embedder
from app.retrieval import STRATEGIES, HybridRetriever
from app.vectorstore import InMemoryStore

ROOT = Path(__file__).resolve().parent.parent


def load_eval(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieval_metrics(hits, relevant: set[str], keywords: list[str]) -> dict:
    sources = [h.source for h in hits]
    precision = sum(s in relevant for s in sources) / len(sources) if sources else 0.0
    recall = len(relevant & set(sources)) / len(relevant) if relevant else 0.0
    joined = " ".join(h.text.lower() for h in hits)
    hit = float(all(kw.lower() in joined for kw in keywords))
    rr = next((1.0 / (i + 1) for i, s in enumerate(sources) if s in relevant), 0.0)
    return {"precision": precision, "recall": recall, "hit_rate": hit, "mrr": rr}


def groundedness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Share of answer sentences whose content words mostly appear in the retrieved context."""
    ctx = set(tokenize(" ".join(contexts)))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[[^\]]+\]", "", answer)) if tokenize(s)]
    if not sentences:
        return 0.0
    supported = sum(len(set(tokenize(s)) & ctx) / len(set(tokenize(s))) >= threshold for s in sentences)
    return supported / len(sentences)


def build_retriever() -> HybridRetriever:
    embedder = build_embedder(os.getenv("EMBEDDER", "sentence-transformers"),
                              os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    r = HybridRetriever(InMemoryStore(), embedder)
    for p in sorted((ROOT / "data/sample_docs").glob("*.md")):
        r.add_file(p)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--answers", action="store_true", help="also generate answers and score groundedness")
    ap.add_argument("--out", default=str(ROOT / "eval/results.json"))
    args = ap.parse_args()

    items = load_eval(ROOT / "eval/eval_set.jsonl")
    retriever = build_retriever()
    report = {"k": args.k, "n_questions": len(items), "strategies": {}}

    for strategy in STRATEGIES:
        totals = {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0}
        for item in items:
            hits = retriever.search(item["question"], k=args.k, strategy=strategy)
            m = retrieval_metrics(hits, set(item["relevant_sources"]), item["expected_keywords"])
            for key in totals:
                totals[key] += m[key]
        report["strategies"][strategy] = {k: round(v / len(items), 3) for k, v in totals.items()}

    if args.answers:
        agent = DocumentAgent(retriever, os.getenv("OPENAI_API_KEY", ""), os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        scores = []
        for item in items:
            res = agent.ask(item["question"])
            scores.append(groundedness(res.answer, [s["text"] for s in res.sources]))
        report["answer_groundedness"] = round(sum(scores) / len(scores), 3)
        report["answer_mode"] = "llm" if agent.client else "extractive"

    print(f"\nRetrieval evaluation on {len(items)} questions (k={args.k})\n")
    print(f"{'strategy':<10}{'precision':>11}{'recall':>9}{'hit rate':>10}{'MRR':>7}")
    for s, m in report["strategies"].items():
        print(f"{s:<10}{m['precision']:>11.3f}{m['recall']:>9.3f}{m['hit_rate']:>10.3f}{m['mrr']:>7.3f}")
    if "answer_groundedness" in report:
        print(f"\nAnswer groundedness ({report['answer_mode']}): {report['answer_groundedness']:.3f}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
