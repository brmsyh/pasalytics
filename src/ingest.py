"""Load the knowledge base: synthetic internal documents (Markdown) and
public Indonesian IP regulations (PDF, downloaded by the user)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: str  # "internal" or "regulation"
    path: str
    text: str
    pages: list[str] = field(default_factory=list)  # PDF text per page
    metadata: dict = field(default_factory=dict)


_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def load_internal_docs(folder: Path = config.INTERNAL_DOCS_DIR) -> list[Document]:
    docs = []
    for path in sorted(folder.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(
            Document(
                doc_id=meta.get("doc_id", path.stem),
                title=meta.get("title", path.stem),
                source_type="internal",
                path=str(path.relative_to(config.ROOT)),
                text=body.strip(),
                metadata=meta,
            )
        )
    return docs


def load_regulations(folder: Path = config.REGULATIONS_DIR) -> list[Document]:
    """Read every PDF in data/raw/regulations. Scanned PDFs without a text
    layer are skipped with a warning (they would need OCR)."""
    from pypdf import PdfReader

    docs = []
    for path in sorted(folder.glob("*.pdf")):
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages)
        if len(text.split()) < 200:
            print(f"[warn] {path.name}: little or no text layer, skipped (needs OCR)")
            continue
        docs.append(
            Document(
                doc_id=path.stem,
                title=path.stem.replace("_", " "),
                source_type="regulation",
                path=str(path.relative_to(config.ROOT)),
                text=text,
                pages=pages,
                metadata={"n_pages": len(pages)},
            )
        )
    return docs


def load_corpus() -> list[Document]:
    return load_internal_docs() + load_regulations()
