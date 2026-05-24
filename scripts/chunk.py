import argparse

import tiktoken

from extract import AUDIT_DIR, extract_text


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    # cl100k_base is OpenAI's tokenizer; good enough as a generic token-count proxy here.
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks: list[dict] = []
    # Sliding window: each chunk advances by `step`, leaving `overlap` shared with the previous.
    step = chunk_size - overlap

    i = 0
    chunk_index = 0

    while i < len(tokens):
        # j = end of this window (clamped to the final token).
        j = min(i + chunk_size, len(tokens))

        chunk_tokens = tokens[i:j]
        chunk_str = enc.decode(chunk_tokens)
        # Decode the prefix [0:i] to get the char offset where this chunk starts in the
        # original text. O(N) per chunk, but the doc sizes here make it negligible.
        char_start = len(enc.decode(tokens[:i]))
        char_end = char_start + len(chunk_str)

        chunks.append({
            "text": chunk_str,
            "source": source,
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": len(chunk_tokens),
        })

        chunk_index += 1

        # Stop once the window has reached the end (don't advance past it).
        if j == len(tokens):
            break

        i += step

    return chunks

def main() -> None:
    # CLI: chunking params + display selector (--show INDEX or --show-all).
    parser = argparse.ArgumentParser(description="Chunk extracted PDF text with tiktoken.")
    parser.add_argument("--file", required=True, help="PDF filename inside the audit-reports/ folder")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--show", type=int, default=0, help="Chunk index to print (default 0)")
    parser.add_argument("--show-all", action="store_true", help="Print every chunk with metadata")
    args = parser.parse_args()

    # Extract → chunk in one pass.
    pdf_path = AUDIT_DIR / args.file
    text = extract_text(str(pdf_path))
    chunks = chunk_text(text, source=args.file, chunk_size=args.chunk_size, overlap=args.overlap)

    print(f"Total chunks created: {len(chunks)}")
    print("---")

    META_KEYS = ("source", "chunk_index", "char_start", "char_end", "token_count")

    # Helper: print one chunk's metadata + text.
    def print_chunk(c: dict) -> None:
        print(f"Chunk #{c['chunk_index']} metadata:")

        for k in META_KEYS:
            print(f"  {k}: {c[k]}")

        print(f"Chunk #{c['chunk_index']} text:")
        print(c["text"])

    # --show-all: dump every chunk, separated by '---'.
    if args.show_all:
        for c in chunks:
            print_chunk(c)
            print("---")

        return

    # Bounds-check --show; print a friendly error instead of crashing on IndexError.
    if args.show < 0 or args.show >= len(chunks):
        print(f"Error: --show {args.show} is out of range (valid: 0 to {len(chunks) - 1})")
        return

    print_chunk(chunks[args.show])


if __name__ == "__main__":
    main()
