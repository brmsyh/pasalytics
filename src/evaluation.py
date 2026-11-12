"""Evaluation utilities.

Retrieval: Hit@k and MRR against the expected source document of each question.
Generation: LLM-as-judge scoring of correctness and faithfulness (1-5), plus
refusal accuracy on questions the knowledge base cannot answer."""
from __future__ import annotations

import json
import re

import pandas as pd

from .generator import Answer, GeminiGenerator, format_context


def retrieval_metrics(retriever, eval_df: pd.DataFrame, ks=(1, 3, 5)) -> tuple[pd.DataFrame, dict]:
    answerable = eval_df[eval_df["question_type"] != "unanswerable"]
    rows = []
    for q in answerable.itertuples():
        hits = retriever.search(q.question, max(ks))
        doc_ids = [h.chunk["doc_id"] for h in hits]
        rank = doc_ids.index(q.expected_doc_id) + 1 if q.expected_doc_id in doc_ids else None
        row = {"qid": q.qid, "question_type": q.question_type, "expected_doc_id": q.expected_doc_id,
               "top1_doc_id": doc_ids[0], "rank": rank, "rr": 1 / rank if rank else 0.0}
        for k in ks:
            row[f"hit@{k}"] = int(rank is not None and rank <= k)
        rows.append(row)
    per_q = pd.DataFrame(rows)
    summary = {f"hit@{k}": per_q[f"hit@{k}"].mean() for k in ks}
    summary["mrr"] = per_q["rr"].mean()
    summary["n_questions"] = len(per_q)
    return per_q, summary


JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented assistant.
Return ONLY a JSON object with integer scores from 1 to 5:
{"correctness": n, "faithfulness": n, "reason": "<one short sentence>"}
- correctness: does the answer match the reference answer (all key facts and numbers)?
- faithfulness: is every claim in the answer supported by the provided context?"""


def judge_answer(judge: GeminiGenerator, answer: Answer, reference: str) -> dict:
    prompt = (f"Question: {answer.question}\n\nReference answer: {reference}\n\n"
              f"Context given to the assistant:\n{format_context(answer.hits)}\n\n"
              f"Assistant answer:\n{answer.text}")
    raw = judge.complete(prompt, system=JUDGE_SYSTEM, temperature=0.0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {"correctness": None, "faithfulness": None, "reason": f"unparseable: {raw[:80]}"}
