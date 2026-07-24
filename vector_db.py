from chromadb import PersistentClient

from embeding import get_embeddings

DB_PATH = "./vectordb"
COLLECTION_NAME = "project_docs"


def get_client():
    return PersistentClient(path=DB_PATH)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(force=False):
    collection = get_collection()

    if collection.count() > 0 and not force:
        print("Data already stored. Skipping.")
        print("Total Records:", collection.count())
        return collection

    if force and collection.count() > 0:
        get_client().delete_collection(name=COLLECTION_NAME)
        collection = get_collection()

    chunks, embeddings = get_embeddings()

    for i in range(len(chunks)):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunks[i]],
            embeddings=[embeddings[i].tolist()],
        )

    print("Data Stored Successfully")
    print("Total Records:", collection.count())
    return collection


if __name__ == "__main__":
    store_chunks()
