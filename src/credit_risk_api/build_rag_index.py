"""Build the retrieval index for regulation-aware explanations (offline, run locally).

Steps:
1. Extract the text of each PDF in data/regulations/, page by page.
2. Split it into overlapping chunks that remember their source and page.
3. Embed every chunk with a small local embedding model (no API, no key).
4. For each model feature, retrieve the most relevant chunks once and save
   the result, so the API never needs an embedding model at runtime.

Run from the project root:
    uv run python -m credit_risk_api.build_rag_index
"""
import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

# Only the guidelines on loan origination and their annexes are indexed. The executive
# summary, background and public-consultation feedback are not rules, and section 8
# (monitoring of existing loans, pp. 60-67) is not about deciding on a new application.
SOURCES = {
    "eba_gl_2020_06.pdf": {
        "title": "EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06)",
        "pages": [(12, 59), (68, 76)],
    },
}
# Running page header repeated on every page; it adds noise to every chunk.
PAGE_HEADER = re.compile(r"FINAL REPORT\s*[–-]\s*GUIDELINES ON LOAN ORIGINATION AND MONITORING\s*\d*")
INPUT_DIR = Path("data/regulations")
OUTPUT_DIR = Path("models/rag")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters shared by consecutive chunks, so no idea is cut in half
TOP_K = 2  # chunks kept per feature

# One retrieval query per regulatory topic, worded for consumer lending (the
# document also covers businesses, real estate and shipping). Features map to
# topics, and every explanation also gets the topic on automated models.
TOPIC_QUERIES = {
    "external_scores": "creditworthiness assessment of consumers using credit scoring, credit registers and credit bureau data",
    "income": "verifying the consumer's income, its sources and stability in the creditworthiness assessment",
    "loan_amount": "the consumer's ability to repay the loan amount and meet the obligations of the credit agreement",
    "affordability": "consumer debt service relative to income: periodic loan payments, income and living expenses",
    "collateral": "loan-to-value and valuation of movable property financed by the loan",
    "employment": "the consumer's employment status, employment history and occupation in the creditworthiness assessment",
    "product": "consumer credit agreements and types of credit products, such as cash loans and revolving credit",
    "personal_data": "collection and use of consumer data, data protection and avoiding discrimination in credit decisions",
    "automated_models": "use of automated models for creditworthiness assessment and credit decisions",
}
FEATURE_TOPICS = {
    "ext_source_1": "external_scores",
    "ext_source_2": "external_scores",
    "ext_source_3": "external_scores",
    "amt_income_total": "income",
    "amt_credit": "loan_amount",
    "amt_annuity": "affordability",
    "amt_goods_price": "collateral",
    "days_employed": "employment",
    "occupation_type": "employment",
    "name_contract_type": "product",
    "name_education_type": "personal_data",
}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return PAGE_HEADER.sub("", text).strip()


def section_starts(reader: PdfReader) -> list[tuple[int, str]]:
    """(first page, title) of the document's sections and subsections, from the PDF bookmarks."""
    starts = []

    def walk(items, depth):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            elif depth <= 1:
                starts.append((reader.get_destination_page_number(item) + 1, item.title.strip()))

    walk(reader.outline, 0)
    return sorted(starts, key=lambda start: start[0])  # stable: keeps bookmark order within a page


def section_for(page: int, starts: list[tuple[int, str]]) -> str:
    titles = [title for first_page, title in starts if first_page <= page]
    return titles[-1] if titles else ""


def extract_pages(pdf_path: Path, first: int, last: int) -> list[tuple[int, str, str]]:
    """(page number, section title, cleaned text) for the pages in [first, last]."""
    reader = PdfReader(pdf_path)
    starts = section_starts(reader)
    pages = []
    for number in range(first, last + 1):
        text = clean_text(reader.pages[number - 1].extract_text() or "")
        if text:
            pages.append((number, section_for(number, starts), text))
    return pages


def chunk_pages(pages: list[tuple[int, str, str]], source: str) -> list[dict]:
    """Split each page into overlapping chunks, cutting at sentence ends when possible."""
    chunks = []
    for page, section, text in pages:
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                sentence_end = text.rfind(". ", start + CHUNK_SIZE // 2, end)
                if sentence_end != -1:
                    end = sentence_end + 1
            chunk = text[start:end].strip()
            if len(chunk) > 100:  # skip fragments such as headers and page numbers
                chunks.append(
                    {"id": len(chunks), "source": source, "section": section, "page": page, "text": chunk}
                )
            if end == len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def top_chunks(query_vector: np.ndarray, chunk_vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices and cosine similarities of the k chunks most similar to the query."""
    scores = chunk_vectors @ query_vector  # vectors are normalised, so this is cosine similarity
    best = np.argsort(scores)[::-1][:k]
    return [(int(i), round(float(scores[i]), 4)) for i in best]


def normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main() -> None:
    chunks = []
    for filename, source in SOURCES.items():
        pages = [p for first, last in source["pages"] for p in extract_pages(INPUT_DIR / filename, first, last)]
        for chunk in chunk_pages(pages, source["title"]):
            chunk["id"] = len(chunks)
            chunks.append(chunk)
    print(f"{len(chunks)} chunks from {len(SOURCES)} document(s)")

    model = TextEmbedding(EMBEDDING_MODEL, cache_dir="data/embedding_model")
    chunk_vectors = normalise(np.array(list(model.passage_embed([c["text"] for c in chunks]))))
    topics = list(TOPIC_QUERIES)
    query_vectors = normalise(np.array(list(model.query_embed([TOPIC_QUERIES[t] for t in topics]))))

    retrieval = {
        topic: [{"chunk_id": i, "score": s} for i, s in top_chunks(vector, chunk_vectors, TOP_K)]
        for topic, vector in zip(topics, query_vectors)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "retrieval.json").write_text(
        json.dumps({"feature_topics": FEATURE_TOPICS, "topics": retrieval}, indent=1) + "\n"
    )
    np.save(OUTPUT_DIR / "embeddings.npy", chunk_vectors.astype(np.float32))

    # Print what was retrieved, so a human can judge whether it makes sense.
    for topic in topics:
        print(f"\n=== {topic}: {TOPIC_QUERIES[topic]}")
        for hit in retrieval[topic]:
            chunk = chunks[hit["chunk_id"]]
            print(f"  [score {hit['score']:.3f}] p.{chunk['page']} | {chunk['section']}")
            print(f"      {chunk['text'][:150]}...")


if __name__ == "__main__":
    main()
