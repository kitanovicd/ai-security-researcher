import argparse
import os

import anthropic
import psycopg
import voyageai
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY / VOYAGE_API_KEY / DATABASE_URL from .env into the process env.
load_dotenv()

EMBED_MODEL = "voyage-3-large"
CLAUDE_MODEL = "claude-sonnet-4-6"

# Pinning Claude to the retrieved context — no outside knowledge, mandatory chunk citations.
SYSTEM_PROMPT = (
    "You are a security expert answering questions about smart contract audits. "
    "Answer ONLY using the provided context chunks. "
    "Cite which chunk number you used for each claim like [chunk 1], [chunk 2]. "
    "If the context doesn't contain the answer, say 'I don't have enough information to answer this.' "
    "Do not use outside knowledge."
)


def _vector_literal(values: list[float]) -> str:
    # pgvector accepts a "[v1,v2,...]" text literal which we cast with ::vector in the SQL.
    return "[" + ",".join(repr(v) for v in values) + "]"


def search(question: str, k: int = 5) -> list[dict]:
    # input_type="query" produces a query-side embedding (chunks were indexed with "document").
    # Using the wrong input_type silently hurts retrieval quality.
    vo = voyageai.Client()
    result = vo.embed([question], model=EMBED_MODEL, input_type="query")
    qvec = _vector_literal(result.embeddings[0])

    # `<=>` is pgvector's cosine-distance operator; ORDER BY ... LIMIT k gives top-k nearest chunks.
    database_url = os.environ["DATABASE_URL"]
    with psycopg.connect(database_url) as conn:
        cur = conn.execute(
            """
            SELECT text, source, chunk_index, embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        )

        rows = cur.fetchall()

    return [
        {"text": text, "source": source, "chunk_index": chunk_index, "distance": float(distance)}
        for text, source, chunk_index, distance in rows
    ]


def _ask_claude(question: str, chunks: list[dict]) -> str:
    # Format retrieved chunks as a numbered context block so the model can cite [chunk N].
    context = "\n\n".join(
        f"[chunk {i+1}] source: {c['source']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    user_msg = f"Question: {question}\n\nContext chunks:\n\n{context}"

    # Single non-streaming Sonnet 4.6 call; reads ANTHROPIC_API_KEY from env.
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Extract the first text block from the response (ignores any other block types).
    return next(b.text for b in response.content if b.type == "text")


def answer(question: str, k: int = 5) -> str:
    return _ask_claude(question, search(question, k=k))


def main() -> None:
    # CLI: positional question, optional -k to control how many chunks to retrieve.
    parser = argparse.ArgumentParser(description="Ask a question about embedded audit reports.")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    # Retrieve once, then reuse the same chunks for both the LLM call and the displayed sources.
    chunks = search(args.question, k=args.k)

    print(_ask_claude(args.question, chunks))
    print()
    print("--- Sources ---")

    for i, c in enumerate(chunks):
        print(f"[chunk {i+1}] source={c['source']}, chunk_index={c['chunk_index']}, distance={c['distance']:.4f}")


if __name__ == "__main__":
    main()
