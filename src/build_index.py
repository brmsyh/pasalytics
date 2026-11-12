"""Build the search index.

    python -m src.build_index            # chunks + Gemini embeddings
    python -m src.build_index --no-embed # chunks only (BM25 mode, no API key needed)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import config
from .chunking import chunk_corpus
from .ingest import load_corpus


def main(embed: bool = True) -> None:
    docs = load_corpus()
    chunks = chunk_corpus(docs)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    n_internal = sum(d.source_type == "internal" for d in docs)
    print(f"{len(docs)} documents ({n_internal} internal, {len(docs) - n_internal} regulations) "
          f"-> {len(chunks)} chunks")

    meta = {"n_chunks": len(chunks), "n_documents": len(docs), "built_at": time.strftime("%Y-%m-%d %H:%M")}
    if embed:
        from .embeddings import GeminiEmbedder

        vectors = GeminiEmbedder().embed([c.text_for_index for c in chunks])
        np.save(config.EMBEDDINGS_PATH, vectors)
        meta.update(embedding_model=config.EMBEDDING_MODEL, embedding_dim=int(vectors.shape[1]))
        print(f"Saved embeddings {vectors.shape} to {config.EMBEDDINGS_PATH.relative_to(config.ROOT)}")
    elif config.EMBEDDINGS_PATH.exists():
        config.EMBEDDINGS_PATH.unlink()  # stale vectors would no longer match the chunks
    config.INDEX_META_PATH.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", action="store_true", help="skip Gemini embeddings (BM25 only)")
    main(embed=not parser.parse_args().no_embed)
