import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from utils import split_text, docs_to_vectorstore, DEFAULT_COLLECTION

# -------- Config --------
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RECURSIVE = os.environ.get("RECURSIVE", "true").lower() == "true"
ALLOWED_EXTS = {".pdf", ".txt", ".md"}
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)
SPLIT_STRATEGY = os.environ.get("SPLIT_STRATEGY", "auto")



def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".md":
        return "text/markdown"
    return "text/plain"


def _load_one(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        for i, d in enumerate(pages, start=1):
            d.metadata = {**(d.metadata or {}), "page": d.metadata.get("page", i)}
        return pages
    elif ext in {".txt", ".md"}:
        return TextLoader(str(path), autodetect_encoding=True).load()
    return []


def load_all_documents(root: Path, recursive: bool = True) -> List[Document]:
    if not root.exists():
        print(f"⚠️ Data folder not found: {root.resolve()}")
        return []
    glob = "**/*" if recursive else "*"
    docs: List[Document] = []
    for p in root.glob(glob):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTS:
            continue
        loaded = _load_one(p)
        for d in loaded:
            md = d.metadata or {}
            
            src = md.get("source") or str(p)
            d.metadata = {
                **md,
                "source_path": str(src),
                "file_name": p.name,
                "ext": p.suffix.lower(),
                "mime": _mime_for(p),
            }
        docs.extend(loaded)
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Uses the improved utils.split_text orchestrator per input doc.
    - Preserves/extends metadata
    - Stamps stable chunk indices per source_path
    """
    # Group by file
    by_src: Dict[str, List[Document]] = {}
    for d in docs:
        md = d.metadata or {}
        by_src.setdefault(md.get("source_path", "unknown"), []).append(d)

    out_chunks: List[Document] = []

    for src, items in by_src.items():
        # Assign a deterministic document id for this source
        doc_id = _hash(src)
        chunk_counter = 0

        for d in items:
            md = d.metadata or {}
            base_meta = {
                "source_path": src,
                "file_name": md.get("file_name"),
                "page": md.get("page"),
                "ext": md.get("ext"),
                "mime": md.get("mime"),
                "doc_id": doc_id,
            }
            
            chunks = split_text(
                d.page_content or "",
                base_metadata=base_meta,
                kind=SPLIT_STRATEGY,
                mime=base_meta.get("mime"),
            )
            # Stamp incremental indices per file
            stamped: List[Document] = []
            for c in chunks:
                c_md = {**(c.metadata or {})}
                c_md["chunk_index"] = chunk_counter
                c_md["chunk_id"] = f"{doc_id}::{chunk_counter}"
                stamped.append(Document(page_content=c.page_content.strip(), metadata=c_md))
                chunk_counter += 1

            out_chunks.extend(stamped)

    return out_chunks


def store_chunks(chunks: List[Document], collection_name: str = COLLECTION_NAME):
    vectordb = docs_to_vectorstore(chunks, collection_name=collection_name)
    print(f"✅ Stored {len(chunks)} chunks to Qdrant collection '{collection_name}'")
    return vectordb


def main():
    try:
        print("===== Loading documents =====")
        docs = load_all_documents(DATA_DIR, recursive=RECURSIVE)
        if not docs:
            print("Nothing to ingest. Put files under:", DATA_DIR.resolve())
            return

        print(f"Loaded {len(docs)} documents")
        print("===== Splitting =====")
        chunks = split_documents(docs)
        print(f"✅ Split into {len(chunks)} chunks")

        print("===== Storing Chunks in Qdrant =====")
        store_chunks(chunks, collection_name=COLLECTION_NAME)
        print("🎉 Done! Stored all chunks in Qdrant")

    except Exception as e:
        print(f"💥 Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
