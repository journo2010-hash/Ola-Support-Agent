
from pathlib import Path
import re

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"


def load_documents():
    documents = {}

    for path in sorted(KB_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents[path.name] = content

    if len(documents) < 12:
        raise RuntimeError(
            f"Expected at least 12 KB documents; found {len(documents)} "
            f"in {KB_DIR}"
        )

    return documents


def fixed_chunks(text, size=80, overlap=15):
    words = text.split()
    chunks = []
    step = size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break

    return chunks


def sentence_chunks(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_chunks(documents, strategy):
    chunks = []

    for filename, content in documents.items():
        parts = (
            fixed_chunks(content)
            if strategy == "fixed"
            else sentence_chunks(content)
        )

        for index, part in enumerate(parts):
            chunks.append({
                "id": f"{filename}::{index}",
                "filename": filename,
                "text": part,
            })

    return chunks


def create_collection(client, name, chunks):
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB still stores and retrieves vectors. TF-IDF generates
    # the vectors locally for this temporary development workaround.
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"filename": chunk["filename"]}
            for chunk in chunks
        ],
        embeddings=matrix.toarray().astype(float).tolist(),
    )

    return collection, vectorizer


def retrieve(query, chunks, vectorizer, collection, top_k=5):
    query_vector = vectorizer.transform([query]).toarray()[0]

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, len(chunks)),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    seen_documents = set()

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        filename = metadata["filename"]

        if filename in seen_documents:
            continue

        seen_documents.add(filename)
        hits.append({
            "filename": filename,
            "text": text,
            "similarity": 1.0 - distance,
        })

    return hits


def main():
    documents = load_documents()
    client = chromadb.Client()

    for strategy in ("fixed", "sentence"):
        chunks = build_chunks(documents, strategy)
        collection_name = f"support_tfidf_{strategy}"

        collection, vectorizer = create_collection(
            client, collection_name, chunks
        )

        print(f"\n{'=' * 55}")
        print(f"Strategy: {strategy}")
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {len(chunks)}")
        print(f"Stored vectors: {collection.count()}")

        query = "What is the policy for escalating a support ticket?"
        hits = retrieve(query, chunks, vectorizer, collection)

        print(f"\nQuery: {query}")
        print("Retrieved documents:")

        for hit in hits:
            print(
                f"- {hit['filename']} "
                f"(similarity={hit['similarity']:.3f})"
            )
            print(f"  {hit['text'][:180]}")

        if not hits:
            print("No relevant documents retrieved.")

    print("\nTF-IDF retrieval smoke test completed.")
    print(
        "This is a temporary development fallback, not the "
        "required SentenceTransformers implementation."
    )


if __name__ == "__main__":
    main()
