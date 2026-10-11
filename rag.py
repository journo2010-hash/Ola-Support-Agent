
"""
RAG core for the Ola Support Agent capstone.

Uses local SentenceTransformers embeddings and two ChromaDB
collections to compare fixed-overlap and sentence-based chunking.

Hybrid ranking:
    75% semantic similarity
    25% lexical keyword overlap
"""

import os
import re
from pathlib import Path
from statistics import median

# Graded RAG runs must never attempt to download a model. The embedding
# model must be present in the local Hugging Face cache before offline grading.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 1

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do",
    "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "the", "to", "what", "when", "which", "with",
}


# --------------------------------------------------
# Evaluation queries
# --------------------------------------------------

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
        "query": (
            "When should a ticket be escalated and "
            "what details are required?"
        ),
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


# --------------------------------------------------
# Load Markdown documents
# --------------------------------------------------

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
            raise ValueError(
                f"Empty knowledge-base document: {file_path.name}"
            )

        documents[file_path.name] = text

    return documents


# --------------------------------------------------
# Chunking strategies
# --------------------------------------------------

def fixed_overlap_chunks(text, size=80, overlap=15):
    """Split text into word-based chunks with overlap."""

    if size <= 0:
        raise ValueError("size must be greater than zero")

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "overlap must be between zero and size - 1"
        )

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

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_chunks(documents, strategy):
    """Build chunks with unique IDs and parent document metadata."""

    rows = []

    for filename, text in documents.items():

        if strategy == "fixed":
            pieces = fixed_overlap_chunks(text)

        elif strategy == "sentence":
            pieces = sentence_chunks(text)

        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}"
            )

        for index, piece in enumerate(pieces):
            rows.append({
                "id": f"{filename}::chunk-{index}",
                "text": piece,
                "parent": filename,
            })

    return rows


# --------------------------------------------------
# Create ChromaDB collection
# --------------------------------------------------

def build_collection(client, model, documents, strategy):
    """Embed chunks and upsert them into a ChromaDB collection."""

    if strategy == "fixed":
        collection_name = "support_fixed_overlap"

    elif strategy == "sentence":
        collection_name = "support_sentence_based"

    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
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

    collection.upsert(
        ids=[row["id"] for row in rows],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "parent_doc": row["parent"],
                "strategy": strategy,
            }
            for row in rows
        ],
    )

    print(
        f"{collection_name}: indexed {len(rows)} chunks"
    )

    return collection


# --------------------------------------------------
# Keyword normalization
# --------------------------------------------------

def normalize_tokens(text):
    """Normalize words and add useful SLA-related keyword variants."""

    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    normalized = set(tokens)

    # Normalize common plural forms.
    for token in list(normalized):

        if len(token) > 4 and token.endswith("ies"):
            normalized.add(token[:-3] + "y")

        elif len(token) > 3 and token.endswith("s"):
            normalized.add(token[:-1])

    normalized -= STOP_WORDS

    # Normalize common first-response terminology.
    if "response" in normalized:
        normalized.add("first-response")

    if "first" in normalized and "response" in normalized:
        normalized.add("first-response")

    # Severity questions often refer to P1/P2/P3/P4 targets.
    if "severity" in normalized:
        normalized.update({"p1", "p2", "p3", "p4"})

    return normalized


# --------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------

def retrieve(query, collection, model, top_k=TOP_K):
    """
    Retrieve and rank unique parent documents.

    Ranking score:
        75% semantic similarity
        25% lexical keyword overlap

    Returns:
        ranked_hits, top_similarity
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection_count = collection.count()

    if collection_count == 0:
        return [], -1.0

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    candidate_count = min(
        collection_count,
        max(top_k * 4, 8),
    )

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    query_tokens = normalize_tokens(query)

    best_by_document = {}
    all_similarities = []

    retrieved_documents = result.get("documents", [[]])[0] or []
    retrieved_metadatas = result.get("metadatas", [[]])[0] or []
    retrieved_distances = result.get("distances", [[]])[0] or []

    for text, metadata, distance in zip(
        retrieved_documents,
        retrieved_metadatas,
        retrieved_distances,
    ):
        similarity = 1.0 - float(distance)
        all_similarities.append(similarity)

        parent = metadata["parent_doc"]
        document_tokens = normalize_tokens(text)

        lexical_score = (
            len(query_tokens & document_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )

        ranking_score = (
            0.75 * similarity
            + 0.25 * lexical_score
        )

        hit = {
            "text": text,
            "parent_doc": parent,
            "similarity": similarity,
            "lexical_score": lexical_score,
            "ranking_score": ranking_score,
        }

        previous = best_by_document.get(parent)

        if (
            previous is None
            or ranking_score > previous["ranking_score"]
        ):
            best_by_document[parent] = hit

    ranked_hits = sorted(
        best_by_document.values(),
        key=lambda hit: hit["ranking_score"],
        reverse=True,
    )

    top_similarity = (
        max(all_similarities)
        if all_similarities
        else -1.0
    )

    return ranked_hits[:top_k], top_similarity


# --------------------------------------------------
# Threshold calibration
# --------------------------------------------------

def calibrate_threshold(collection, model):
    """Estimate a fallback threshold from test-query similarities."""

    in_scores = []

    for item in EVALUATION_QUERIES:
        _, score = retrieve(
            item["query"],
            collection,
            model,
            top_k=1,
        )
        in_scores.append(score)

    out_scores = []

    for query in OUT_OF_SCOPE_QUERIES:
        _, score = retrieve(
            query,
            collection,
            model,
            top_k=1,
        )
        out_scores.append(score)

    lowest_in_scope = min(in_scores)
    highest_out_scope = max(out_scores)

    print("\nThreshold calibration")

    print(
        "In-scope similarities:",
        [round(score, 4) for score in in_scores],
    )

    print(
        "Out-of-scope similarities:",
        [round(score, 4) for score in out_scores],
    )

    if highest_out_scope < lowest_in_scope:
        threshold = (
            highest_out_scope + lowest_in_scope
        ) / 2

        print("Observed score clusters are separated.")

    else:
        threshold = (
            median(in_scores) + median(out_scores)
        ) / 2

        print(
            "WARNING: Similarity clusters overlap. "
            "Test more queries before relying on this threshold."
        )

    print(f"Selected threshold: {threshold:.4f}")

    return threshold


# --------------------------------------------------
# Grounded answer
# --------------------------------------------------

def grounded_answer(query, collection, model, threshold):
    """Return retrieved policy text or a safe fallback."""

    hits, top_similarity = retrieve(
        query,
        collection,
        model,
    )

    if top_similarity < threshold or not hits:
        return {
            "answer": (
                "I don't know based on the available support policies."
            ),
            "fallback": True,
            "similarity": round(top_similarity, 4),
            "sources": [],
        }

    answer = (
        "Relevant policy information:\n"
        + "\n".join(
            f"- {hit['text']}"
            for hit in hits
        )
    )

    return {
        "answer": answer,
        "fallback": False,
        "similarity": round(top_similarity, 4),
        "sources": [
            hit["parent_doc"]
            for hit in hits
        ],
    }


# --------------------------------------------------
# Precision and recall evaluation
# --------------------------------------------------

def evaluate_collection(collection, model):
    """Calculate document-level precision and recall per query."""

    results = []

    for item in EVALUATION_QUERIES:

        hits, _ = retrieve(
            item["query"],
            collection,
            model,
        )

        retrieved = {
            hit["parent_doc"]
            for hit in hits
        }

        expected = item["expected"]
        relevant_retrieved = retrieved & expected

        precision = (
            len(relevant_retrieved) / len(retrieved)
            if retrieved
            else 0.0
        )

        recall = (
            len(relevant_retrieved) / len(expected)
            if expected
            else 0.0
        )

        row = {
            "query": item["query"],
            "expected": sorted(expected),
            "retrieved": sorted(retrieved),
            "relevant_retrieved": sorted(
                relevant_retrieved
            ),
            "precision": precision,
            "recall": recall,
        }

        results.append(row)

        print(f"\nQuery: {item['query']}")
        print(f"Expected documents: {sorted(expected)}")
        print(f"Retrieved documents: {sorted(retrieved)}")
        print(
            "Relevant documents retrieved:",
            sorted(relevant_retrieved),
        )

        if retrieved:
            print(
                f"Precision = {len(relevant_retrieved)}"
                f"/{len(retrieved)}"
            )
        else:
            print("Precision = 0/0 (no documents retrieved)")

        print(
            f"Recall = {len(relevant_retrieved)}"
            f"/{len(expected)}"
        )

        print(
            f"Precision: {precision:.3f}; "
            f"Recall: {recall:.3f}"
        )

    if not results:
        return [], 0.0, 0.0

    avg_precision = (
        sum(row["precision"] for row in results)
        / len(results)
    )

    avg_recall = (
        sum(row["recall"] for row in results)
        / len(results)
    )

    print(f"\nMean precision: {avg_precision:.3f}")
    print(f"Mean recall: {avg_recall:.3f}")

    return results, avg_precision, avg_recall


# --------------------------------------------------
# Initialize RAG for CrewAI or other modules
# --------------------------------------------------

def initialize_support_rag():
    """
    Initialize sentence-based RAG for reuse.

    Returns:
        sentence_collection, model, threshold
    """

    documents = load_documents()
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.Client()

    sentence_collection = build_collection(
        client,
        model,
        documents,
        "sentence",
    )

    threshold = calibrate_threshold(
        sentence_collection,
        model,
    )

    return sentence_collection, model, threshold


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    documents = load_documents()

    print(
        f"Loaded {len(documents)} policy documents."
    )

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.Client()

    fixed_collection = build_collection(
        client,
        model,
        documents,
        "fixed",
    )

    sentence_collection = build_collection(
        client,
        model,
        documents,
        "sentence",
    )

    fixed_threshold = calibrate_threshold(
        fixed_collection,
        model,
    )

    sentence_threshold = calibrate_threshold(
        sentence_collection,
        model,
    )

    print("\n=== Fixed-overlap evaluation ===")

    _, fixed_precision, fixed_recall = evaluate_collection(
        fixed_collection,
        model,
    )

    print("\n=== Sentence-based evaluation ===")

    _, sentence_precision, sentence_recall = evaluate_collection(
        sentence_collection,
        model,
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
        print(
            "Preliminary recommendation: "
            "fixed-overlap chunking."
        )

    elif (
        sentence_precision + sentence_recall
        > fixed_precision + fixed_recall
    ):
        print(
            "Preliminary recommendation: "
            "sentence-based chunking."
        )

    else:
        print(
            "The two strategies tie on the current test set."
        )

    print("\n=== Grounded-answer demonstration ===")

    for item in EVALUATION_QUERIES[:3]:

        result = grounded_answer(
            item["query"],
            sentence_collection,
            model,
            sentence_threshold,
        )

        print(f"\nQuestion: {item['query']}")
        print(f"Fallback: {result['fallback']}")
        print(f"Similarity: {result['similarity']}")
        print(f"Sources: {result['sources']}")
        print(result["answer"])

    print("\n=== Out-of-scope fallback demonstration ===")

    for query in OUT_OF_SCOPE_QUERIES:

        result = grounded_answer(
            query,
            sentence_collection,
            model,
            sentence_threshold,
        )

        print(f"\nQuestion: {query}")
        print(f"Fallback: {result['fallback']}")
        print(f"Similarity: {result['similarity']}")
        print(result["answer"])


if __name__ == "__main__":
    main()
