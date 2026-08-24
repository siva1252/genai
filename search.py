from bm25_search import search_bm25
from embeding import get_model
from vector_db import get_collection


def search_chunks(question, n_results=5):
    """
    Semantic-only search.
    Returns (context_text, best_distance).
    Lower distance = more relevant to the PDF.
    """
    ranked = search_chunks_ranked(question, n_results=n_results)
    if not ranked:
        return "", None

    context = "\n".join(text for _, text, _ in ranked)
    best_distance = ranked[0][2]
    return context, best_distance


def search_chunks_ranked(question, n_results=5):
    """
    Semantic branch for hybrid.
    Returns [(chunk_id, text, distance), ...] (lower distance = better).
    """
    collection = get_collection()

    if collection.count() == 0:
        raise ValueError(
            "Vector database is empty. Run vector_db.py first to store chunks."
        )

    model = get_model()
    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0] if results.get("distances") else []

    return list(zip(ids, documents, distances))


def hybrid_search(question, n_results=5, fetch_k=10):
    """
    Hybrid = Semantic (MiniLM + Chroma) + Keyword (BM25), fused with RRF.
    Returns (context_text, best_semantic_distance).
    """
    semantic = search_chunks_ranked(question, n_results=fetch_k)
    keyword = search_bm25(question, n_results=fetch_k)

    rrf_scores = {}
    texts = {}

    for rank, (chunk_id, text, _dist) in enumerate(semantic, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        texts[chunk_id] = text

    for rank, (chunk_id, text, _score) in enumerate(keyword, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        texts[chunk_id] = text

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
    context = "\n".join(texts[chunk_id] for chunk_id, _ in fused)

    best_distance = semantic[0][2] if semantic else None
    return context, best_distance


# How relevant is this question to the PDF?
# For MiniLM + Chroma L2, smaller = closer.
# Above this → probably not in the document.
DOC_DISTANCE_LIMIT = 1.4


def is_doc_relevant(best_distance):
    if best_distance is None:
        return False
    return best_distance <= DOC_DISTANCE_LIMIT
