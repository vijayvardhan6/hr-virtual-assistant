import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig,
    OptimizersConfigDiff, QuantizationConfig
)
from langchain_experimental.text_splitter import SemanticChunker  

load_dotenv()

# ----------------------------
# Defaults / ENV
# ----------------------------
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "hr_policies")

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


SPLIT_STRATEGY = os.environ.get("SPLIT_STRATEGY", "auto")
TOKEN_CHUNK_SIZE = int(os.environ.get("TOKEN_CHUNK_SIZE", "512"))
TOKEN_CHUNK_OVERLAP = int(os.environ.get("TOKEN_CHUNK_OVERLAP", "10"))
MIN_CHARS_PER_CHUNK = int(os.environ.get("MIN_CHARS_PER_CHUNK", "450"))
MAX_CHARS_PER_CHUNK = int(os.environ.get("MAX_CHARS_PER_CHUNK", "1000"))
CHAR_CHUNK_OVERLAP = int(os.environ.get("CHAR_CHUNK_OVERLAP", "200"))
SEM_BREAKPOINT_TYPE = os.environ.get("SEM_BREAKPOINT_TYPE", "percentile")  
SEM_BREAKPOINT_AMOUNT = float(os.environ.get("SEM_BREAKPOINT_AMOUNT", "85"))  
SEM_BUFFER = int(os.environ.get("SEM_BUFFER", "1"))     


# ----------------------------
# Splitters
# ----------------------------
def _mk_semantic_splitter(emb: HuggingFaceEmbeddings):
    """
    Compatible SemanticChunker factory — works across LangChain versions.
    Uses only supported args and applies size filtering afterward if needed.
    """
    t = SEM_BREAKPOINT_TYPE.lower()
    if t in {"std", "stdev", "standard_deviation"}:
        t = "standard_deviation"
    elif t in {"iqr", "interquartile"}:
        t = "interquartile"
    else:
        t = "percentile"

    
    return SemanticChunker(
        emb,
        breakpoint_threshold_type=t,
        breakpoint_threshold_amount=SEM_BREAKPOINT_AMOUNT,
        buffer_size=SEM_BUFFER,
        add_start_index=True,
    )


# ----------------------------
# Embeddings / Qdrant
# ----------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    QDRANT_LOCAL_PATH = os.environ.get("QDRANT_LOCAL_PATH", "./qdrant_local")
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    return QdrantClient(path=QDRANT_LOCAL_PATH)

def _ensure_collection(client: QdrantClient, collection_name: str, embeddings: HuggingFaceEmbeddings):
    dim = len(embeddings.embed_query("dimension probe"))
    try:
        if client.collection_exists(collection_name=collection_name):
            return
    except Exception:
        try:
            client.get_collection(collection_name)  
            return
        except Exception:
            pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=True,
        ),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type="int8",
                quantile=0.99,
                always_ram=False
            )
        ),
    )

    try:
        client.update_collection(
            collection_name=collection_name,
            quantization_config=QuantizationConfig(
                scalar=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(type="int8", quantile=0.99, always_ram=False)
                )
            )
        )
    except Exception:
        pass

# ----------------------------
# Utility: normalizers & helpers
# ----------------------------
_SOFT_BREAK = re.compile(r"(?<!\.)\n(?=[a-z])")
_HARD_HYPH = re.compile(r"(\w)-\n(\w)")
_MULTI_WS = re.compile(r"[\t\x0b\x0c]+")


def normalize_text(text: str) -> str:
    if not text:
        return text
    t = _HARD_HYPH.sub(r"\1\2", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _SOFT_BREAK.sub(" ", t)
    t = _MULTI_WS.sub(" ", t)
    return t


def ensure_model_limits(emb: HuggingFaceEmbeddings, desired_chunk: int, desired_overlap: int) -> Tuple[int, int]:
    max_len = None
    try:
        max_len = getattr(emb.client, "max_seq_length", None)
    except Exception:
        max_len = None

    if max_len is None:
        return desired_chunk, desired_overlap

    margin = 16
    safe = max(32, min(desired_chunk, max_len - margin))
    ovlp = max(0, min(desired_overlap, max(0, safe // 4)))
    return safe, ovlp



def _mk_markdown_header_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    )


def _mk_token_splitter(chunk_size: int, overlap: int) -> TokenTextSplitter:
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )


def _mk_char_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=[
            "\n```", 
            "```",   
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
        ],
        chunk_size=max(MIN_CHARS_PER_CHUNK, min(MAX_CHARS_PER_CHUNK, 1300)),
        chunk_overlap=max(100, min(350, CHAR_CHUNK_OVERLAP)),
        length_function=len,
        add_start_index=True,
    )


# ----------------------------
# Orchestrated splitting
# ----------------------------

def _coalesce_docs(docs: List[Document], min_chars: int) -> List[Document]:
    if not docs:
        return docs
    out: List[Document] = []
    buf: List[Document] = []
    run = 0

    def flush():
        nonlocal buf, run
        if not buf:
            return
        if len(buf) == 1:
            out.append(buf[0])
        else:
            txt = "".join(d.page_content for d in buf)
            meta = dict(buf[0].metadata)
            meta["merged"] = True
            meta["merged_count"] = len(buf)
            meta["child_spans"] = [(d.metadata.get("start_index", None), len(d.page_content)) for d in buf]
            out.append(Document(page_content=txt, metadata=meta))
        buf = []
        run = 0

    for d in docs:
        if run < min_chars:
            buf.append(d)
            run += len(d.page_content)
        else:
            flush()
            buf = [d]
            run = len(d.page_content)
    flush()
    return out


def _add_path_metadata(md_parts: Dict[str, str]) -> str:
    parts = [md_parts.get(k) for k in ("h1", "h2", "h3", "h4") if md_parts.get(k)]
    return " > ".join(parts)


def split_text(
    text: str,
    base_metadata: Optional[Dict] = None,
    kind: Optional[str] = None,
    mime: Optional[str] = None,
) -> List[Document]:
    base_metadata = base_metadata or {}
    strategy = (kind or SPLIT_STRATEGY).lower()

    if strategy == "auto":
        if mime and ("pdf" in mime or "html" in mime or "plain" in mime):
            strategy = "semantic"   
        else:
            strategy = "semantic_hybrid" if text.strip().startswith("#") else "semantic"

    text = normalize_text(text)

    emb = get_embeddings()
    safe_token_size, safe_token_overlap = ensure_model_limits(emb, TOKEN_CHUNK_SIZE, TOKEN_CHUNK_OVERLAP)

    header_splitter = _mk_markdown_header_splitter()
    token_splitter = _mk_token_splitter(safe_token_size, safe_token_overlap)
    char_splitter = _mk_char_splitter()
    semantic_splitter = _mk_semantic_splitter(emb)

    def as_docs(chunks: List[str], metadata_template: Dict) -> List[Document]:
        docs = []
        for i, c in enumerate(chunks):
            md = dict(metadata_template)
            md["chunk_index"] = i
            docs.append(Document(page_content=c, metadata=md))
        return docs

    # --- semantic strategies ---
    if strategy == "semantic":
        pieces = semantic_splitter.split_text(text)
        return _coalesce_docs(as_docs(pieces, dict(base_metadata)), MIN_CHARS_PER_CHUNK)


    md_sections = header_splitter.split_text(text)
    if not md_sections:
        pieces = char_splitter.split_text(text)
        return _coalesce_docs(as_docs(pieces, dict(base_metadata)), MIN_CHARS_PER_CHUNK)

    docs: List[Document] = []
    for s_idx, sec in enumerate(md_sections):
        md = {**base_metadata, **sec.metadata}
        md["section_index"] = s_idx
        md["path"] = _add_path_metadata(md)
        start_base = sec.metadata.get("start_index", None)
        sub_chunks = token_splitter.split_text(sec.page_content)
        docs.extend([
            Document(page_content=c, metadata={**md, "chunk_index": i, "start_index": start_base})
            for i, c in enumerate(sub_chunks)
        ])
    return _coalesce_docs(docs, MIN_CHARS_PER_CHUNK)


# ----------------------------
# Vector store ingestion
# ----------------------------

def docs_to_vectorstore(docs: List[Document], collection_name: str = DEFAULT_COLLECTION) -> QdrantVectorStore:
    embeddings = get_embeddings()
    client = get_qdrant()
    _ensure_collection(client, collection_name, embeddings)

    db = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    db.add_documents(docs)
    return db



def texts_to_docs(
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    strategy: Optional[str] = None,
    mimes: Optional[List[Optional[str]]] = None,
) -> List[Document]:
    
    all_docs: List[Document] = []
    metadatas = metadatas or [{} for _ in texts]
    mimes = mimes or [None for _ in texts]
    for t, md, mm in zip(texts, metadatas, mimes):
        all_docs.extend(split_text(t, base_metadata=md, kind=strategy, mime=mm))
    return all_docs


__all__ = [
    "split_text",
    "texts_to_docs",
    "docs_to_vectorstore",
    "get_embeddings",
    "get_qdrant",
]