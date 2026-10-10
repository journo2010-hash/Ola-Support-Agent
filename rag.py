
"""RAG core for the Ola Support Agent capstone.

Uses local SentenceTransformers embeddings and two separate ChromaDB
collections to compare fixed-overlap and sentence-based chunking.
"""

import re
from pathlib import Path
from statistics import median

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Five in-scope queries with their expected relevant documents.
EVALUATION_QUERIES = [
    {
        "query": "How are Critical and High tickets prioritized?",
        "expected": {"ticket_priority.md"},
    },
    {
        "query": "What are the initial response targets for each severity?",
        "expected": {"sla_policy.md"},
    },
    {
        "query": "When should a ticket be escalated and what details are required?",
        "expected": {"escalation_matrix.md"},
    },
    {
        "query": "How should agents handle refund requests?",
        "expected": {"refund_policy.md"},
    },
    {
        "query": "What should be communicated during a confirmed outage?",
        "expected": {"outage_communication.md"},
    },
]

OUT_OF_SCOPE_QUERIES = [
    "How do I bake sourdough bread at home?",
    "What is the capital of Japan?",
]


def load_documents():
    """Load Markdown documents from the knowledge_base directory."""
    files = sorted(KB_DIR.glob("*.md"))

    if len(files) < 12:
        raise ValueError(
            f"Expected at least 12 Markdown documents in {KB_DIR}; "
            f"found {len(files)}."
        )

    documents = {}
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty knowledge-base document: {file_path.name}")
        documents[file_path.name] = text

    return documents


def fixed_overlap_chunks(text, size=80, overlap=15):
    """Split text into word-based chunks with overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
        start += size - overlap

    return chunks


def sentence_chunks(text):
    """Split text into sentence-based chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def build_chunks(documents, strategy):
    """Return chunk text, IDs, and parent-document metadata."""
    rows = []

    for filename, text in documents.items():
        if strategy == "fixed":
            pieces = fixed_overlap_chunks(text)
        elif strategy == "sentence":
            pieces = sentence_chunks(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for index, piece in enumerate(pieces):
            rows.append({
                "id": f"{filename}::chunk-{index}",
                "text": piece,
                "parent": filename,
            })

    return rows


def build_collection(client, model, documents, strategy):
    """Embed chunks and upsert them into a dedicated ChromaDB collection."""
    collection_name = (
        "support_fixed_overlap"
        if strategy == "fixed"
        else "support_sentence_based"
    )

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    rows = build_chunks(documents, strategy)

    texts = [row["text"] for row in rows]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # Use upsert as required by the project brief.
    collection.upsert(
        ids=[row["id"] for row in rows],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"parent_doc": row["parent"], "strategy": strategy}
            for row in rows
        ],
    )

    print(f"{collection_name}: indexed {len(rows)} chunks")
    return collection


def retrieve(query, collection, model, top_k=TOP_K):
    """Retrieve chunks, deduplicate parent documents, and score similarity."""
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(collection.count(), max(top_k * 4, 8)),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    seen_documents = set()

    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        # Chroma cosine distance = 1 - cosine similarity.
        similarity = 1.0 - float(distance)
        parent = metadata["parent_doc"]

        hits.append({
            "text": text,
            "parent_doc": parent,
            "similarity": similarity,
        })

    # Preserve the best chunk for each parent document.
    unique_hits = []
    for hit in hits:
        if hit["parent_doc"] not in seen_documents:
            seen_documents.add(hit["parent_doc"])
            unique_hits.append(hit)
        if len(unique_hits) >= top_k:
            break

    top_similarity = hits[0]["similarity"] if hits else -1.0
    return unique_hits, top_similarity


def calibrate_threshold(collection, model):
    """Measure similarities and derive a threshold from observed query scores."""
    in_scores = []

    for item in EVALUATION_QUERIES:
        _, score = retrieve(item["query"], collection, model, top_k=1)
        in_scores.append(score)

    out_scores = []
    for query in OUT_OF_SCOPE_QUERIES:
        _, score = retrieve(query, collection, model, top_k=1)
        out_scores.append(score)

    lowest_in_scope = min(in_scores)
    highest_out_scope = max(out_scores)

    print("\nThreshold calibration")
    print("In-scope similarities:", [round(s, 4) for s in in_scores])
    print("Out-of-scope similarities:", [round(s, 4) for s in out_scores])

    if highest_out_scope < lowest_in_scope:
        threshold = (highest_out_scope + lowest_in_scope) / 2
        print("Observed score clusters are separated.")
    else:
        # This is a provisional threshold, not proof of separation.
        threshold = (median(in_scores) + median(out_scores)) / 2
        print(
            "WARNING: Similarity clusters overlap. The provisional "
            "threshold needs further query testing before deployment."
        )

    print(f"Selected threshold: {threshold:.4f}")
    return threshold


def grounded_answer(query, collection, model, threshold):
    """Produce an extractive mock answer using retrieved evidence only."""
    hits, top_similarity = retrieve(query, collection, model)

    if top_similarity < threshold or not hits:
        return {
            "answer": "I don't know based on the available support policies.",
            "fallback": True,
            "similarity": round(top_similarity, 4),
            "sources": [],
        }

    # MOCK_LLM behavior: use retrieved policy text, without inventing facts.
    answer = "Relevant policy information:\n" + "\n".join(
        f"- {hit['text']}" for hit in hits
    )

    return {
        "answer": answer,
        "fallback": False,
        "similarity": round(top_similarity, 4),
        "sources": [hit["parent_doc"] for hit in hits],
    }


def evaluate_collection(collection, model):
    """Calculate document-level precision and recall per query."""
    results = []

    for item in EVALUATION_QUERIES:
        hits, _ = retrieve(item["query"], collection, model)
        retrieved = {hit["parent_doc"] for hit in hits}
        expected = item["expected"]

        relevant_retrieved = retrieved & expected

        precision = (
            len(relevant_retrieved) / len(retrieved)
            if retrieved else 0.0
        )
        recall = (
            len(relevant_retrieved) / len(expected)
            if expected else 0.0
        )

        row = {
            "query": item["query"],
            "expected": sorted(expected),
            "retrieved": sorted(retrieved),
            "relevant_retrieved": sorted(relevant_retrieved),
            "precision": precision,
            "recall": recall,
        }
        results.append(row)

        print(f"\nQuery: {item['query']}")
        print(f"Expected documents: {sorted(expected)}")
        print(f"Retrieved documents: {sorted(retrieved)}")
        print(
            f"Precision = {len(relevant_retrieved)}/{len(retrieved)}"
            if retrieved else "Precision = 0/0 (no documents retrieved)"
        )
        print(
            f"Recall = {len(relevant_retrieved)}/{len(expected)}"
        )
        print(f"Precision: {precision:.3f}; Recall: {recall:.3f}")

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)

    print(f"\nMean precision: {avg_precision:.3f}")
    print(f"Mean recall: {avg_recall:.3f}")

    return results, avg_precision, avg_recall


def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} policy documents.")

    # The model is downloaded only if it is not already cached locally.
    model = SentenceTransformer(MODEL_NAME)

    # Ephemeral client: rebuild both collections on each run.
    client = chromadb.Client()

    fixed_collection = build_collection(
        client, model, documents, "fixed"
    )
    sentence_collection = build_collection(
        client, model, documents, "sentence"
    )

    fixed_threshold = calibrate_threshold(fixed_collection, model)
    sentence_threshold = calibrate_threshold(sentence_collection, model)

    print("\n=== Fixed-overlap evaluation ===")
    _, fixed_precision, fixed_recall = evaluate_collection(
        fixed_collection, model
    )

    print("\n=== Sentence-based evaluation ===")
    _, sentence_precision, sentence_recall = evaluate_collection(
        sentence_collection, model
    )

    print("\n=== Chunking comparison ===")
    print(
        f"Fixed-overlap: precision={fixed_precision:.3f}, "
        f"recall={fixed_recall:.3f}"
    )
    print(
        f"Sentence-based: precision={sentence_precision:.3f}, "
        f"recall={sentence_recall:.3f}"
    )

    if (
        fixed_precision + fixed_recall
        > sentence_precision + sentence_recall
    ):
        print("Preliminary recommendation: fixed-overlap chunking.")
    elif (
        sentence_precision + sentence_recall
        > fixed_precision + fixed_recall
    ):
        print("Preliminary recommendation: sentence-based chunking.")
    else:
        print("The two strategies tie on the current test set.")

    # Demonstrate supported answers and the out-of-scope fallback.
    print("\n=== Grounded-answer demonstration ===")
    for item in EVALUATION_QUERIES[:3]:
        result = grounded_answer(
            item["query"], sentence_collection, model, sentence_threshold
        )
        print(f"\nQuestion: {item['query']}")
        print(f"Fallback: {result['fallback']}")
        print(f"Sources: {result['sources']}")
        print(result["answer"])

    for query in OUT_OF_SCOPE_QUERIES:
        result = grounded_answer(
            query, sentence_collection, model, sentence_threshold
        )
        print(f"\nOut-of-scope question: {query}")
        print(f"Fallback: {result['fallback']}")
        print(result["answer"])


if __name__ == "__main__":
    main()

def initialize_support_rag():
    """Initialize the RAG system for reuse by the support agent."""
    documents = load_documents()
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.Client()

    sentence_collection = build_collection(
        client, model, documents, "sentence"
    )
    threshold = calibrate_threshold(sentence_collection, model)

    return sentence_collection, model, threshold
