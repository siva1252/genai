from sentence_transformers import SentenceTransformer

from pdf import get_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embeddings(chunks=None):
    if chunks is None:
        chunks, _, _ = get_chunks()

    model = get_model()
    embeddings = model.encode(chunks)
    return chunks, embeddings


if __name__ == "__main__":
    chunks, embeddings = get_embeddings()

    print(f"Total Chunks: {len(chunks)}")
    print(f"Total Embeddings: {len(embeddings)}")
    print(embeddings)
