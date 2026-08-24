import re

from rank_bm25 import BM25Okapi

from pdf import get_chunks

_bm25 = None
_chunks = None


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25():
    """Build BM25 once from the same PDF chunks used by Chroma."""
    global _bm25, _chunks
    if _bm25 is None:
        _chunks, _, _ = get_chunks()
        tokenized = [tokenize(c) for c in _chunks]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _chunks


def search_bm25(question, n_results=5):
    """
    Keyword search.
    Returns [(chunk_id, text, score), ...] sorted by BM25 score (higher = better).
    """
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(tokenize(question))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = ranked[: min(n_results, len(ranked))]

    return [(f"chunk_{i}", chunks[i], float(score)) for i, score in top]
