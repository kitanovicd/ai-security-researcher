import argparse
import os
from pathlib import Path

import psycopg
import voyageai
from dotenv import load_dotenv

from chunk import chunk_text
from extract import AUDIT_DIR, extract_text

# Load ANTHROPIC_API_KEY / VOYAGE_API_KEY / DATABASE_URL from .env into the process env.
load_dotenv()

EMBED_MODEL = "voyage-3-large"
# Voyage caps batches at 128 documents per embed call.
VOYAGE_BATCH_SIZE = 128


def _already_embedded(conn: psycopg.Connection, source: str) -> bool:
    # Cheap existence probe — any row with this source means we've embedded it before.
    cur = conn.execute("SELECT 1 FROM chunks WHERE source = %s LIMIT 1", (source,))
    return cur.fetchone() is not None


def _vector_literal(values: list[float]) -> str:
    # pgvector accepts a "[v1,v2,...]" text literal which we cast with ::vector on insert.
    return "[" + ",".join(repr(v) for v in values) + "]"


def embed_pdf(pdf_path: str) -> int:
    source = Path(pdf_path).name
    database_url = os.environ["DATABASE_URL"]

    # Skip if this PDF has already been embedded (idempotent re-runs of --all).
    with psycopg.connect(database_url) as conn:
        if _already_embedded(conn, source):
            print(f"Skipping {source}: already embedded")
            return 0

    # Step 1: extract text from the PDF, then split into token-sized chunks.
    text = extract_text(pdf_path)
    chunks = chunk_text(text, source=source)

    # Step 2: embed all chunks via Voyage, batched to respect the 128-doc-per-call limit.
    # input_type="document" tells Voyage to use the indexing-side embedding (queries use "query").
    vo = voyageai.Client()
    embeddings: list[list[float]] = []
    total_tokens = 0

    for i in range(0, len(chunks), VOYAGE_BATCH_SIZE):
        batch = chunks[i:i + VOYAGE_BATCH_SIZE]

        result = vo.embed(
            [c["text"] for c in batch],
            model=EMBED_MODEL,
            input_type="document",
        )

        embeddings.extend(result.embeddings)
        total_tokens += result.total_tokens

    # Step 3: build (chunk_metadata, vector) tuples ready for INSERT.
    rows = [
        (
            c["source"],
            c["chunk_index"],
            c["text"],
            c["char_start"],
            c["char_end"],
            c["token_count"],
            _vector_literal(emb),
        )
        for c, emb in zip(chunks, embeddings)
    ]

    # Step 4: bulk insert via executemany; the `with` block commits on success / rolls back on error.
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (source, chunk_index, text, char_start, char_end, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                """,
                rows,
            )

    print(f"Embedding {source}: {len(chunks)} chunks created, {total_tokens} tokens used")
    return len(chunks)


def main() -> None:
    # CLI: --file foo.pdf OR --all (mutually exclusive, one is required).
    parser = argparse.ArgumentParser(description="Embed audit-report PDFs into pgvector.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="PDF filename inside audit-reports/")
    group.add_argument("--all", action="store_true", help="Embed every PDF in audit-reports/")
    args = parser.parse_args()

    # Build the list of PDFs to process.
    if args.all:
        pdf_paths = sorted(AUDIT_DIR.glob("*.pdf"))
    else:
        pdf_paths = [AUDIT_DIR / args.file]

    # embed_pdf handles its own skip-if-already-embedded check.
    for path in pdf_paths:
        embed_pdf(str(path))


if __name__ == "__main__":
    main()
