import time
from retrieval_pipeline import retrieve_context, retriever, DEFAULT_K, CTX_MAX_CHARS

try:
    from retrieval_pipeline import _fmt_source
except Exception:
    def _fmt_source(md: dict) -> str:
        src = md.get("source_path") or md.get("source_file") or md.get("source") or "document"
        page = md.get("page")
        header = md.get("header_path")
        parts = [src]
        if page is not None:
            parts.append(f"p.{page}")
        if header:
            parts.append(header)
        return " · ".join(str(p) for p in parts if p)


def test_retrieve(query: str, k: int = DEFAULT_K, show_raw: bool = False):
    """Quickly test retrieval with or without raw preview, including timing."""
    print(f"Query: {query}")
    print(f"Top-k: {k} | Context limit: {CTX_MAX_CHARS}\n")

    # --- Measure time for full retrieval pipeline ---
    start_time = time.time()
    context = retrieve_context(query, k=k)
    elapsed = time.time() - start_time
    print(f"⏱️ Retrieval time: {elapsed:.3f} seconds")

    print("\n==== Assembled Context ====")
    if not context.strip():
        print("(no context retrieved)")
    else:
        print(f"Length of context: {len(context)}")
        print(context)

    # --- Optional: measure time for raw retriever ---
    if show_raw:
        print("\n==== Raw Retrieved Chunks ====")
        start_raw = time.time()
        try:
            docs = retriever.invoke(query)
        except AttributeError:
            docs = retriever.get_relevant_documents(query)
        raw_elapsed = time.time() - start_raw
        print(f"⏱️ Raw retriever time: {raw_elapsed:.3f} seconds")

        for i, d in enumerate(docs[:k], start=1):
            md = d.metadata or {}
            print(f"\n[{i}] {_fmt_source(md)}")
            print((d.page_content or "").replace("\n", " "))


if __name__ == "__main__":
    query = "what is the dress code"
    test_retrieve(query, k=8, show_raw=True)
