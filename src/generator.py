"""Grounded answer generation with Gemini.

The model only sees the retrieved chunks, numbered [1]..[k], and must cite
them. If the context does not contain the answer it must say so instead of
guessing - this refusal behaviour is measured in the evaluation notebook."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .gemini_rest import post
from .retriever import Hit

NOT_FOUND = "I could not find this in the knowledge base."

SYSTEM_PROMPT = f"""You are Pasalytics, the internal knowledge assistant of an Indonesian intellectual
property firm. Answer staff questions using ONLY the numbered context passages.

Rules:
- Cite every factual statement with the passage number in square brackets, e.g. [2].
- Quote numbers (fees, deadlines, days) exactly as written in the context.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
- Context passages may be in Indonesian; answer in the language of the question.
- Be concise: at most 6 sentences or a short bullet list.
- Internal procedures are firm policy, not legal advice. Do not invent law."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int]

    @property
    def is_refusal(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.text.lower()


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        loc = f", page {c['page']}" if c.get("page") else ""
        blocks.append(f"[{i}] ({c['doc_id']} - {c['section']}{loc})\n{c['text']}")
    return "\n\n".join(blocks)


class GeminiGenerator:
    def __init__(self, model: str = config.CHAT_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or config.GEMINI_API_KEY

    def complete(self, prompt: str, system: str = SYSTEM_PROMPT, temperature: float = 0.1) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = post(f"models/{self.model}:generateContent", payload, self.api_key)
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
        text = self.complete(prompt)
        groups = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)  # [1] and [1, 5]
        cited = sorted({int(n) for g in groups for n in re.split(r"\s*,\s*", g) if 0 < int(n) <= len(hits)})
        return Answer(question, text, hits, cited)


class RAGPipeline:
    def __init__(self, retriever, generator: GeminiGenerator, k: int = config.TOP_K):
        self.retriever, self.generator, self.k = retriever, generator, k

    def ask(self, question: str) -> Answer:
        return self.generator.answer(question, self.retriever.search(question, self.k))
