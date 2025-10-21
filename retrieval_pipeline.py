import os
from typing import List
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from utils import get_embeddings, DEFAULT_COLLECTION, get_qdrant
from llm import _llm

# ----------------------------
# Environment knobs
# ----------------------------
DEFAULT_K = int(os.environ.get("RETRIEVE_K", "8"))
FETCH_K = int(os.environ.get("FETCH_K", "80"))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.4"))
CTX_MAX_CHARS = int(os.environ.get("CTX_MAX_CHARS", "3000"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.0"))
COLLECTION = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)

USE_MULTIQUERY = os.environ.get("USE_MULTIQUERY", "false").lower() == "true"
USE_RERANKER = os.environ.get("USE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

QDRANT_EF_SEARCH = int(os.environ.get("QDRANT_EF_SEARCH", "96"))


# Multi-query
_HAS_MQ = False
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    _HAS_MQ = True
except Exception:
    USE_MULTIQUERY = False

# Cross-encoder re-ranker
_cross = None
def _get_cross():
    global _cross
    if _cross is None and USE_RERANKER:
        try:
            from sentence_transformers import CrossEncoder
            _cross = CrossEncoder(RERANKER_MODEL)
        except Exception:
            _cross = False
    return _cross

# ----------------------------
# Vector store / retriever
# ----------------------------

def _fmt_source(md: dict) -> str:
    """Human-friendly source string."""
    src = md.get("source_path") or md.get("source_file") or md.get("source") or "document"
    page = md.get("page")
    header = md.get("header_path")
    parts = [src]
    if page is not None:
        parts.append(f"p.{page}")
    if header:
        parts.append(header)
    return " · ".join(str(p) for p in parts if p)


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    
    cut = text.rfind("\n\n", 0, max_chars)
    if cut == -1:
        cut = text.rfind("\n", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].rstrip() + " …"


def _unique_by_chunk_id(docs: List) -> List:
    seen = set()
    out = []
    for d in docs:
        cid = (d.metadata or {}).get("chunk_id")
        key = cid or (((d.metadata or {}).get("source_path"), (d.metadata or {}).get("chunk_index")))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def build_vectorstore() -> QdrantVectorStore:
    embeddings = get_embeddings()
    client: QdrantClient = get_qdrant()

    try:
        client.set_collection_params(
            collection_name=COLLECTION,
            hnsw_config={"ef_search": QDRANT_EF_SEARCH}
        )
    except Exception:
        pass

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )


_vectorstore = build_vectorstore()

retriever_base = _vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": max(DEFAULT_K, 1),
        "fetch_k": max(FETCH_K, DEFAULT_K),
        "lambda_mult": MMR_LAMBDA,
    },
)

if USE_MULTIQUERY and _HAS_MQ:
    try:
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever_base,
            llm=_llm(),
            include_original=True,
        )
    except Exception:
        retriever = retriever_base
else:
    retriever = retriever_base


def retrieve_context(query: str, k: int = DEFAULT_K) -> str:
    candidates = retriever.invoke(query)

    if not candidates:
        return ""

    # candidates = _score_filter(candidates, MIN_SCORE)

    # cross-encoder re-rank on the pool
    cross = _get_cross()
    if cross:
        pairs = [(query, d.page_content) for d in candidates]
        try:
            scores = cross.predict(pairs)  
            for d, s in zip(candidates, scores):
                (d.metadata or {}).update({"rerank_score": float(s)})
            candidates.sort(key=lambda d: (d.metadata or {}).get("rerank_score", 0.0), reverse=True)
        except Exception:
            pass 

    seeds = _unique_by_chunk_id(candidates)[:max(k, 1)]

    assembled = []

    for d in seeds:
        piece = f"{d.page_content}\n[Source: {_fmt_source(d.metadata or {})}]"
        assembled.append(piece)
        budget_left -= len(piece)
        if budget_left <= 0:
            break

    return "\n\n---\n\n".join(assembled)
