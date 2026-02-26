#!/usr/bin/env python3
"""
Process all Purple Book section PDFs: convert to markdown, extract terms, aggregate.

Pipeline:
  1. PDFs in purple_book/sections/ -> markdown in purple_book/sections_md/
  2. Markdown -> term-definition pairs in purple_book/terms/{name}.json
  3. Aggregate all terms/*.json -> purple_book_terms.json in project root

Usage:
  python process_purple_book_sections.py
  python process_purple_book_sections.py --from-md   # process existing markdown only
"""

import argparse
import json
import sys
from pathlib import Path

from extract_purple_book_terms import extract_term_definition_pairs, pdf_to_markdown

PROJECT_ROOT = Path(__file__).parent
SECTIONS_PDF = PROJECT_ROOT / "purple_book" / "sections"
SECTIONS_MD = PROJECT_ROOT / "purple_book" / "sections_md"
TERMS_DIR = PROJECT_ROOT / "purple_book" / "terms"
AGGREGATE_OUTPUT = PROJECT_ROOT / "purple_book_terms.json"


def process_section_pdf(pdf_path: Path, verbose: bool = True) -> list[dict]:
    """Convert PDF to markdown, extract terms, return pairs."""
    stem = pdf_path.stem
    md_path = SECTIONS_MD / f"{stem}.md"
    terms_path = TERMS_DIR / f"{stem}.json"

    if verbose:
        print(f"  {pdf_path.name} -> markdown...", file=sys.stderr)
    md_text = pdf_to_markdown(pdf_path)
    SECTIONS_MD.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")

    if verbose:
        print(f"  {pdf_path.name} -> terms...", file=sys.stderr)
    pairs = extract_term_definition_pairs(md_text)
    TERMS_DIR.mkdir(parents=True, exist_ok=True)
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    return pairs


def process_section_md(md_path: Path, verbose: bool = True) -> list[dict]:
    """Extract terms from markdown, return pairs."""
    stem = md_path.stem
    terms_path = TERMS_DIR / f"{stem}.json"

    if verbose:
        print(f"  {md_path.name} -> terms...", file=sys.stderr)
    text = md_path.read_text(encoding="utf-8")
    pairs = extract_term_definition_pairs(text)
    TERMS_DIR.mkdir(parents=True, exist_ok=True)
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    return pairs


def aggregate_terms(verbose: bool = True) -> list[dict]:
    """Merge all purple_book/terms/*.json, deduplicate by term (first wins)."""
    seen: set[str] = set()
    aggregated: list[dict] = []

    json_files = sorted(TERMS_DIR.glob("*.json"))
    if not json_files:
        if verbose:
            print("No JSON files in purple_book/terms/", file=sys.stderr)
        return []

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            pairs = json.load(f)
        for p in pairs:
            term = p.get("term", "")
            key = term.strip().lower().replace(" ", " ")
            if key and key not in seen:
                seen.add(key)
                # Add section file to source for traceability
                p["source"] = p.get("source", "") + f" (from {jf.stem})"
                aggregated.append(p)
        if verbose:
            print(f"  Merged {jf.name} ({len(pairs)} terms)", file=sys.stderr)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Process Purple Book sections: PDF->MD->terms->aggregate"
    )
    parser.add_argument(
        "--from-md",
        action="store_true",
        help="Process existing markdown in sections_md/ (skip PDF conversion)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    if args.from_md:
        SECTIONS_MD.mkdir(parents=True, exist_ok=True)
        md_files = sorted(SECTIONS_MD.glob("*.md"))
        if not md_files:
            print(f"No markdown files in {SECTIONS_MD}", file=sys.stderr)
            sys.exit(1)
        print(f"Processing {len(md_files)} markdown section(s)...", file=sys.stderr)
        for md_path in md_files:
            print(f"\n{md_path.name}:", file=sys.stderr)
            process_section_md(md_path, verbose)
    else:
        SECTIONS_PDF.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(SECTIONS_PDF.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files in {SECTIONS_PDF}", file=sys.stderr)
            print("Add section PDFs (e.g. terminology.pdf) or use --from-md for existing markdown.", file=sys.stderr)
            sys.exit(1)
        print(f"Processing {len(pdf_files)} section(s)...", file=sys.stderr)
        for pdf_path in pdf_files:
            print(f"\n{pdf_path.name}:", file=sys.stderr)
            process_section_pdf(pdf_path, verbose)

    print(f"\nAggregating terms from {TERMS_DIR}...", file=sys.stderr)
    aggregated = aggregate_terms(verbose)
    with open(AGGREGATE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nDone. Wrote {len(aggregated)} unique terms to {AGGREGATE_OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
