"""Gemini embeddings (REST batchEmbedContents) with batching, retry and L2
normalisation.

Vectors are normalised because gemini-embedding-001 only returns unit-length
vectors at its full 3072 dimensions; at 768 dimensions we normalise ourselves
so a dot product equals cosine similarity."""
from __future__ import annotations

import numpy as np

from . import config
from .gemini_rest import post


class GeminiEmbedder:
    def __init__(self, api_key: str | None = None, model: str = config.EMBEDDING_MODEL,
                 dim: int = config.EMBEDDING_DIM, batch_size: int = 20):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model, self.dim, self.batch_size = model, dim, batch_size

    def _embed_batch(self, texts: list[str], task_type: str, retries: int = 5) -> list[list[float]]:
        payload = {"requests": [{
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": t}]},
            "taskType": task_type,
            "outputDimensionality": self.dim,
        } for t in texts]}
        data = post(f"models/{self.model}:batchEmbedContents", payload, self.api_key, retries)
        return [e["values"] for e in data["embeddings"]]

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i:i + self.batch_size], task_type))
            print(f"[embed] {min(i + self.batch_size, len(texts))}/{len(texts)} texts embedded")
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)

    def embed_query(self, query: str) -> np.ndarray:
        arr = np.asarray(self._embed_batch([query], "RETRIEVAL_QUERY")[0], dtype=np.float32)
        return arr / max(float(np.linalg.norm(arr)), 1e-12)
