import argparse
import os
from pathlib import Path

import psycopg
import voyageai
from dotenv import load_dotenv

from chunk import chunk_text
from extract import AUDIT_DIR, extract_text

load_dotenv()

EMBED_MODEL = "voyage-3-large"
VOYAGE_BATCH_SIZE = 128


def _already_embedded(conn: psycopg.Connection, source: str) -> bool:
    cur = conn.execute("SELECT 1 FROM chunks WHERE source = %s LIMIT 1", (source,))
    return cur.fetchone() is not None


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(v) for v in values) + "]"


def embed_pdf(pdf_path: str) -> int:
    source = Path(pdf_path).name
    database_url = os.environ["DATABASE_URL"]

    with psycopg.connect(database_url) as conn:
        if _already_embedded(conn, source):
            print(f"Skipping {source}: already embedded")
            return 0

    text = extract_text(pdf_path)
    chunks = chunk_text(text, source=source)

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
    parser = argparse.ArgumentParser(description="Embed audit-report PDFs into pgvector.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="PDF filename inside audit-reports/")
    group.add_argument("--all", action="store_true", help="Embed every PDF in audit-reports/")
    args = parser.parse_args()

    if args.all:
        pdf_paths = sorted(AUDIT_DIR.glob("*.pdf"))
    else:
        pdf_paths = [AUDIT_DIR / args.file]

    for path in pdf_paths:
        embed_pdf(str(path))


if __name__ == "__main__":
    main()
