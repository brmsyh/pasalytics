"""Retrievers: BM25 (lexical), dense (Gemini embeddings) and a hybrid of both
combined with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

_STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
me my of on or our should the their there this to was we what when where which who why will with
you your yang dan di ke dari untuk dengan atau pada dalam""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int


def load_chunks(path=config.CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text_for_index"]) for c in chunks])

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order)]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict], embeddings: np.ndarray, embedder):
        assert len(chunks) == len(embeddings), "chunks and embeddings are out of sync - rebuild the index"
        self.chunks, self.embeddings, self.embedder = chunks, embeddings, embedder

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        q = self.embedder.embed_query(query)
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order)]


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank)) over retrievers.
    Lexical search catches exact codes and numbers ("SOP-TM-05", "31 months");
    dense search catches paraphrases and cross-language matches (English
    question vs Indonesian regulation text)."""

    name = "hybrid"

    def __init__(self, retrievers: list, candidates: int = 20, rrf_k: int = config.RRF_K):
        self.retrievers, self.candidates, self.rrf_k = retrievers, candidates, rrf_k

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for retriever in self.retrievers:
            for hit in retriever.search(query, self.candidates):
                cid = hit.chunk["chunk_id"]
                by_id[cid] = hit.chunk
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + hit.rank)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(by_id[cid], score, r + 1) for r, (cid, score) in enumerate(ranked)]


def build_retriever(mode: str = "hybrid"):
    """Factory used by the app and notebooks. Falls back to BM25 when no
    embeddings or API key are available."""
    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    if mode == "bm25" or not config.EMBEDDINGS_PATH.exists() or not config.GEMINI_API_KEY:
        return bm25
    from .embeddings import GeminiEmbedder

    dense = DenseRetriever(chunks, np.load(config.EMBEDDINGS_PATH), GeminiEmbedder())
    return dense if mode == "dense" else HybridRetriever([bm25, dense])
