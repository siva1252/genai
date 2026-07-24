from embeding import get_model
from vector_db import get_collection


def search_chunks(question, n_results=5):
    """
    Returns (context_text, best_distance).
    Lower distance = more relevant to the PDF.
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

    documents = results["documents"][0]
    distances = results["distances"][0] if results.get("distances") else []

    if not documents:
        return "", None

    context = "\n".join(documents)
    best_distance = distances[0] if distances else None
    return context, best_distance


# How relevant is this question to the PDF?
# For MiniLM + Chroma L2, smaller = closer.
# Above this → probably not in the document.
DOC_DISTANCE_LIMIT = 1.4


def is_doc_relevant(best_distance):
    if best_distance is None:
        return False
    return best_distance <= DOC_DISTANCE_LIMIT
