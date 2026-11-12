"""Structure-aware chunking.

* Internal Markdown documents are split on their '## ' section headings, so a
  chunk never mixes two procedures. Long sections fall back to a sliding window.
* Regulations are split on article boundaries ('Pasal N'), then grouped into
  windows, keeping the page number for citations.
Each chunk carries a breadcrumb ("SOP-TM-05 > 2. Reminder Schedule") that is
prepended to the text before embedding; this markedly improves retrieval of
short sections.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from . import config
from .ingest import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source_type: str
    path: str
    page: int | None
    text: str

    @property
    def text_for_index(self) -> str:
        return f"{self.doc_id} {self.title} > {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["text_for_index"] = self.text_for_index
        return d


def _window(words: list[str], size: int, overlap: int) -> list[str]:
    if len(words) <= size:
        return [" ".join(words)]
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def chunk_markdown(doc: Document, max_words=config.MAX_CHUNK_WORDS,
                   overlap=config.CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    sections = re.split(r"\n(?=## )", doc.text)
    chunks: list[Chunk] = []
    for sec in sections:
        sec = sec.strip()
        if not sec or sec.startswith("# ") and "\n" not in sec:
            continue  # the document title line alone
        lines = sec.splitlines()
        heading = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else "Introduction"
        body = "\n".join(lines[1:]).strip() if lines[0].startswith("#") else sec
        if not body:
            continue
        for j, piece in enumerate(_window(body.split(), max_words, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{len(chunks):02d}",
                doc_id=doc.doc_id, title=doc.title,
                section=heading + (f" (part {j + 1})" if j else ""),
                source_type=doc.source_type, path=doc.path, page=None,
                text=piece if len(body.split()) > max_words else body,
            ))
    return chunks


_PASAL = re.compile(r"(?=\bPasal\s+\d+\b)")


def chunk_regulation(doc: Document, max_words=config.MAX_CHUNK_WORDS * 2,
                     overlap=config.CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[str] = []
    articles: list[str] = []  # article numbers contained in the current buffer
    buffer_page = 1

    def label() -> str:
        if not articles:
            return "Preamble / general provisions"
        return f"Pasal {articles[0]}" if len(articles) == 1 else f"Pasal {articles[0]}-{articles[-1]}"

    def flush():
        nonlocal buffer, articles
        if buffer:
            for piece in _window(buffer, max_words, overlap):
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::{len(chunks):04d}", doc_id=doc.doc_id,
                    title=doc.title, section=label(), source_type=doc.source_type,
                    path=doc.path, page=buffer_page, text=piece))
        buffer, articles = [], []

    for page_no, page_text in enumerate(doc.pages, start=1):
        for part in _PASAL.split(page_text):
            words = part.split()
            if not words:
                continue
            m = re.match(r"Pasal\s+(\d+)", part.strip())
            if m and len(buffer) >= max_words // 2:
                flush()  # start a new chunk at an article boundary
            if not buffer:
                buffer_page = page_no
            if m:
                articles.append(m.group(1))
            buffer.extend(words)
    flush()
    return chunks


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_markdown(doc) if doc.source_type == "internal" else chunk_regulation(doc))
    return chunks
