"""Agentic layer: the LLM decides which tool to call (search, filtered search, list docs) via function calling.

If no OPENAI_API_KEY is set, it falls back to an extractive answer built from the top retrieved
chunks, so the app still works for demos and CI.
"""
import json
import re
from dataclasses import dataclass, field

from .bm25 import tokenize
from .retrieval import HybridRetriever

SYSTEM_PROMPT = """You answer questions using ONLY the user's uploaded documents.
Always call a tool to find evidence before answering. Use `search_documents` for general questions,
pass `source` when the user names a specific document, and use `list_documents` when they ask what is available.
Cite evidence inline as [source#chunk]. If the documents do not contain the answer, say so plainly."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Hybrid (vector + keyword) search over document chunks. Optionally restrict to one document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "source": {"type": "string", "description": "Exact document filename to restrict to (optional)."},
                    "k": {"type": "integer", "description": "Number of chunks, 1-8.", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the documents that have been uploaded and how many chunks each has.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class AgentResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    mode: str = "llm"


class DocumentAgent:
    def __init__(self, retriever: HybridRetriever, api_key: str = "", model: str = "gpt-4o-mini", max_steps: int = 4):
        self.retriever, self.model, self.max_steps = retriever, model, max_steps
        self.client = None
        if api_key:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)

    # ---------- tools ----------
    def _run_tool(self, name: str, args: dict, collected: dict) -> str:
        if name == "list_documents":
            return json.dumps(self.retriever.sources())
        if name == "search_documents":
            k = max(1, min(int(args.get("k", 4)), 8))
            hits = self.retriever.search(args["query"], k=k, source=args.get("source") or None)
            for h in hits:
                collected[h.id] = h
            return json.dumps([{"id": h.id, "section": h.section, "text": h.text} for h in hits])
        return json.dumps({"error": f"unknown tool {name}"})

    # ---------- main entry ----------
    def ask(self, question: str, source: str | None = None) -> AgentResult:
        if self.client is None:
            return self._extractive(question, source)

        collected: dict = {}
        calls: list[dict] = []
        user_msg = question if not source else f"{question}\n(Only use the document: {source})"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]

        for _ in range(self.max_steps):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS, temperature=0
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return AgentResult(
                    answer=msg.content or "",
                    sources=[h.to_dict() for h in collected.values()],
                    tool_calls=calls,
                )
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                calls.append({"tool": tc.function.name, "args": args})
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": self._run_tool(tc.function.name, args, collected)}
                )

        return AgentResult(
            answer="I couldn't reach a confident answer within the step limit.",
            sources=[h.to_dict() for h in collected.values()],
            tool_calls=calls,
        )

    # ---------- no-LLM fallback ----------
    def _extractive(self, question: str, source: str | None) -> AgentResult:
        hits = self.retriever.search(question, k=3, source=source)
        call = [{"tool": "search_documents", "args": {"query": question, "source": source}}]
        if not hits:
            return AgentResult("No relevant passages found in the uploaded documents.", [], call, "extractive")
        q = set(tokenize(question))
        sentences = []
        for h in hits:
            body = h.text.split(": ", 1)[-1]
            for s in re.split(r"(?<=[.!?])\s+", body):
                # weight shared words by rarity so "approves" beats "claim"
                overlap = sum(self.retriever.bm25.idf(t) for t in q & set(tokenize(s)))
                if overlap:
                    sentences.append((overlap, s.strip(), h.id))
        sentences.sort(key=lambda x: x[0], reverse=True)
        picked = sentences[:2] or [(0, hits[0].text.split(": ", 1)[-1][:300], hits[0].id)]
        answer = " ".join(f"{s} [{cid}]" for _, s, cid in picked)
        return AgentResult(answer, [h.to_dict() for h in hits], call, "extractive")
