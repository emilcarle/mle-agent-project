
from pathlib import Path
import re
from my_solution.embedding import embed_texts
import numpy as np
        

def load_documents(data_dir: Path) -> list[dict]:
    """Load approved policy documents from Markdown files."""

    documents: list[dict] = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next(
            (
                line.removeprefix("#").strip()
                for line in text.splitlines()
                if line.startswith("# ")
            ),
            path.stem,
        )
        documents.append(
            {
                "source_id": path.stem,
                "title": title,
                "text": text,
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split policy files into citable paragraphs while preserving source IDs."""

    chunks: list[dict] = []
    for document in documents:
        source_id = document["source_id"]
        text = document["text"]
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip() and not paragraph.startswith("# ") and not paragraph.startswith("Source ID:")
        ]
        for position, paragraph in enumerate(paragraphs):
            chunks.append(
                {
                    "chunk_id": f"{source_id}-{position}",
                    "source_id": source_id,
                    "title": document["title"],
                    "position": position,
                    "text": paragraph,
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed each chunk's text into a vector and store it in the chunk dictionary."""

    for chunk in chunks:
        chunk["vector"] = embed_texts(chunk["text"])[0]
    return chunks


def retrieve_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Return the top_k chunks most similar to the query, each with its similarity score."""

    query_vector = np.array(embed_texts(query)[0])
    # Vectors are normalized, so the dot product equals the cosine similarity.
    # {**chunk, ...} copies the chunk, so the shared chunk list is not modified.
    scored = [
        {**chunk, "score": float(np.dot(query_vector, chunk["vector"]))}
        for chunk in chunks
    ]
    return sorted(scored, key=lambda chunk: chunk["score"], reverse=True)[:top_k]


def determine_retrieval_threshold(cases: list[dict], chunks: list[dict], top_k: int = 3) -> float:
    """Return a similarity threshold halfway between typical relevant and typical irrelevant chunk scores."""

    relevant_scores = []
    irrelevant_scores = []
    for case in cases:
        for chunk in retrieve_chunks(case["prompt"], chunks, top_k=top_k):
            # A chunk is relevant if its source is one the case expects as evidence
            if chunk["source_id"] in case["expected_sources"]:
                relevant_scores.append(chunk["score"])
            else:
                irrelevant_scores.append(chunk["score"])

    # Halfway between the average relevant score and the average irrelevant score
    threshold = (np.mean(relevant_scores) + np.mean(irrelevant_scores)) / 2
    return round(float(threshold), 2)
