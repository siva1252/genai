import time
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# 1. LOAD PDF
# =========================================================

pdf_path = "documents/company_policy.pdf"

reader = PdfReader(pdf_path)

document_text = ""

for page in reader.pages:
    document_text += page.extract_text() + "\n"


# =========================================================
# 2. CHUNK DOCUMENT
# =========================================================

def create_chunks(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = create_chunks(document_text)

print("Total chunks:", len(chunks))


# =========================================================
# 3. MODELS TO COMPARE
# =========================================================

models = {

    "model_a":
        "sentence-transformers/all-MiniLM-L6-v2",

    "model_b":
        "sentence-transformers/all-mpnet-base-v2",

    "model_c":
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
}


# =========================================================
# 4. TEST QUESTIONS
# =========================================================

questions = [

    "How many annual leaves can an employee take?",

    "What is the company's work from home policy?",

    "How many hours should employees work per day?",

    "What is the notice period for resignation?"
]


# =========================================================
# 5. CHROMA
# =========================================================

client = chromadb.PersistentClient(
    path="./embedding_comparison_db"
)


# =========================================================
# 6. COMPARE MODELS
# =========================================================

results = {}


for model_name, model_id in models.items():

    print("\n==============================")
    print("MODEL:", model_name)
    print("MODEL ID:", model_id)
    print("==============================")

    # Load model
    model = SentenceTransformer(model_id)


    # -----------------------------------------------------
    # Measure embedding time
    # -----------------------------------------------------

    start_time = time.time()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    embedding_time = time.time() - start_time


    print("Embedding time:",
          round(embedding_time, 3),
          "seconds")

    print("Embedding dimension:",
          embeddings.shape[1])


    # -----------------------------------------------------
    # Create separate collection
    # -----------------------------------------------------

    collection = client.get_or_create_collection(
        name=f"collection_{model_name}"
    )


    # -----------------------------------------------------
    # Store embeddings
    # -----------------------------------------------------

    ids = [
        f"{model_name}_chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )


    # -----------------------------------------------------
    # Retrieval testing
    # -----------------------------------------------------

    retrieval_times = []

    retrieved_results = []


    for question in questions:

        start_time = time.time()

        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        )

        result = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=3
        )

        retrieval_time = time.time() - start_time

        retrieval_times.append(retrieval_time)

        retrieved_results.append({
            "question": question,
            "results": result["documents"][0]
        })


    avg_retrieval_time = (
        sum(retrieval_times)
        / len(retrieval_times)
    )


    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results[model_name] = {

        "model_id": model_id,

        "embedding_dimension":
            embeddings.shape[1],

        "embedding_time":
            embedding_time,

        "average_retrieval_time":
            avg_retrieval_time,

        "retrieved_results":
            retrieved_results
    }


# =========================================================
# 7. FINAL COMPARISON
# =========================================================

print("\n\n===================================")
print("FINAL MODEL COMPARISON")
print("===================================")

for model_name, result in results.items():

    print("\nModel:", model_name)

    print(
        "Embedding dimension:",
        result["embedding_dimension"]
    )

    print(
        "Embedding time:",
        round(result["embedding_time"], 3),
        "seconds"
    )

    print(
        "Average retrieval time:",
        round(
            result["average_retrieval_time"],
            4
        ),
        "seconds"
    )