import argparse
from pathlib import Path

import pymupdf

AUDIT_DIR = Path("audit-reports")


def extract_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return (
        text
        .replace("​", "")
        .replace("\xa0", "")
        .replace(",!", "")
        .replace("\x00", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from an audit-report PDF.")
    parser.add_argument("--file", required=True, help="PDF filename inside the audit-reports/ folder")
    parser.add_argument("--start", type=int, default=0, help="Start character index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End character index (exclusive)")
    args = parser.parse_args()

    pdf_path = AUDIT_DIR / args.file
    text = extract_text(str(pdf_path))
    end = args.end if args.end is not None else len(text)

    print(text[args.start:end])
    print("---")
    print(f"Total characters: {len(text)}")

    doc = pymupdf.open(pdf_path)
    print(f"Total pages: {doc.page_count}")
    doc.close()


if __name__ == "__main__":
    main()
