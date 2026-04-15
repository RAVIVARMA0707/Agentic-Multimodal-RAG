from src.core.db import get_vector_store
from src.api.v1.tools.fts_search_tool import fts_search

def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
   Merge vector and FTS results using Reciprocal Rank Fusion (RRF).


   RRF score for a chunk = sum of 1/(rank + 60) across both result lists.
   Chunks appearing in both lists score higher than those in only one.
   The constant 60 prevents top-ranked outliers from dominating.
   """

    vector_store = get_vector_store()

    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search(query, k=k)

    rrf_scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunks[key] = {"content": doc.page_content, "metadata": doc.metadata}

    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunks[key] = {"content": item["content"], "metadata": item["metadata"]}

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[key] for key, _ in ranked[:k]]
