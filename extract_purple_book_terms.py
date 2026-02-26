#!/usr/bin/env python3
"""
Extract term-definition pairs from the IUPAC Purple Book (Compendium of Polymer
Terminology and Nomenclature).

Supports two input modes:
  1. Markdown: parses pre-cleaned .md with format "## n.m term \\n definition"
  2. PDF: extracts selected pages (PyMuPDF), converts to markdown (Docling), then parses

Usage:
  python extract_purple_book_terms.py purple_book/sections_md/terminology.md -o terms.json
  python extract_purple_book_terms.py purple_book/sections/terminology.pdf -o terms.json
  python extract_purple_book_terms.py ONLINE-IUPAC-PB2-Online-June2014.pdf --pages 23-42 -o terms.json
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

# Section headers to skip (not term definitions)
SKIP_HEADINGS = {
    "contents", "preamble", "glossary of basic terms in polymer science",
    "terminology", "alphabetical index of terms",
}


def keep_pages(input_pdf: Path, output_pdf: Path, keep: list[int]) -> None:
    """Extract selected pages from PDF using PyMuPDF (fitz)."""
    try:
        import fitz
    except ImportError:
        print("Install PyMuPDF: pip install pymupdf", file=sys.stderr)
        sys.exit(1)

    src = fitz.open(str(input_pdf))
    dst = fitz.open()
    for i in keep:
        dst.insert_pdf(src, from_page=i, to_page=i)
    dst.save(str(output_pdf))
    dst.close()
    src.close()


def pdf_to_markdown(pdf_path: Path) -> str:
    """Convert PDF to markdown using Docling."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("Install docling: pip install docling", file=sys.stderr)
        sys.exit(1)

    converter = DocumentConverter()
    doc = converter.convert(str(pdf_path)).document
    return doc.export_to_markdown()


def parse_pages_range(spec: str) -> list[int]:
    """Parse '23-42' or '23,24,25' into list of 0-based page indices."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a.strip()), int(b.strip())
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return pages


def extract_term_definition_pairs(text: str) -> list[dict]:
    """
    Parse markdown with format: ## n.m term \\n definition

    Structure:
    - ## 1.1 macromolecule  -> section 1.1, term "macromolecule"
    - ## polymer molecule   -> alternate term (same definition)
    - ## 1.8 monomeric unit monomer unit  -> terms "monomeric unit", "monomer unit"
    - ## mer  -> alternate term
    - ## 2.13 \\n ## uniform polymer \\n ## monodisperse polymer \\n Definition
      -> terms "uniform polymer", "monodisperse polymer"
    """
    pairs: list[dict] = []
    seen_terms: set[str] = set()
    section_re = re.compile(r"^(\d+(?:\.\d+)*)\s*$")  # "2.13" only
    numbered_term_re = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")  # "1.1 macromolecule"

    def normalize_term(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    def add_pair(term: str, definition: str, section: str):
        term = term.strip()
        definition = re.sub(r"\s+", " ", definition.strip()).strip()
        if not term or len(term) < 2:
            return
        if not definition or len(definition) < 10:
            return
        key = normalize_term(term)
        if key in seen_terms:
            return
        if key in SKIP_HEADINGS:
            return
        seen_terms.add(key)
        pairs.append({"term": term, "definition": definition, "source": f"section:{section}"})

    def extract_terms_from_heading(heading: str) -> list[str]:
        """Extract term(s) from heading. '1.1 macromolecule' -> ['macromolecule']."""
        heading = heading.strip()
        if not heading:
            return []
        # "1.1 macromolecule" or "1.8 monomeric unit monomer unit"
        m = numbered_term_re.match(heading)
        if m:
            rest = m.group(2).strip()
            if rest.lower() in SKIP_HEADINGS:
                return []
            # "monomeric unit monomer unit" -> ["monomeric unit", "monomer unit"]
            # Find multiple terms when same suffix repeats (unit, molecule, etc.)
            sub_matches = re.findall(
                r"(\w+(?:\s+\w+)*?)\s+(unit|molecule|polymer|chain)\b",
                rest,
                re.IGNORECASE,
            )
            if len(sub_matches) >= 2:
                return [f"{a} {b}".strip() for a, b in sub_matches]
            return [rest] if rest else []
        # "polymer molecule" or "mer" - no number, whole thing is term
        if heading.lower() in SKIP_HEADINGS:
            return []
        if section_re.match(heading):  # "2.13" only
            return []
        return [heading]

    lines = text.split("\n")
    i = 0
    current_section = ""
    current_terms: list[str] = []
    current_definition_parts: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("## "):
            heading_content = stripped[3:].strip()  # after "## "

            # Flush previous block if we have definition
            if current_definition_parts and current_terms:
                definition = " ".join(current_definition_parts)
                for term in current_terms:
                    add_pair(term, definition, current_section)

            # Parse new heading
            current_definition_parts = []

            if section_re.match(heading_content):
                # "2.13" - section only, terms follow
                current_section = heading_content
                current_terms = []
            else:
                terms = extract_terms_from_heading(heading_content)
                if terms:
                    m = numbered_term_re.match(heading_content)
                    if m:
                        current_section = m.group(1)
                        current_terms = terms
                    else:
                        # Alternate term (e.g. "polymer molecule" after "1.1 macromolecule")
                        current_terms = list(current_terms) + terms
                else:
                    current_terms = []
            i += 1
            continue

        # Non-heading line - definition body
        if stripped and current_terms:
            current_definition_parts.append(stripped)
        i += 1

    # Flush last block
    if current_definition_parts and current_terms:
        definition = " ".join(current_definition_parts)
        for term in current_terms:
            add_pair(term, definition, current_section)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract term-definition pairs from IUPAC Purple Book"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to Purple Book markdown or PDF file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: purple_book/terms/<input_stem>.json)",
    )
    parser.add_argument(
        "--pages",
        metavar="RANGE",
        help="Page range for PDF input, e.g. '23-42' or '23,24,25' (0-based indices)",
    )
    parser.add_argument(
        "--save-md",
        metavar="PATH",
        type=Path,
        help="Save intermediate markdown to file (PDF input only)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "txt", "terms-only"],
        default="json",
        help="Output format: json (term+definition), txt (readable), terms-only (one term per line)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        ext = "json" if args.format == "json" else "txt"
        output_path = Path(__file__).parent / "purple_book" / "terms" / f"{input_path.stem}.{ext}"

    if input_path.suffix.lower() == ".md":
        if not args.quiet:
            print(f"Extracting from {input_path}...", file=sys.stderr)
        text = input_path.read_text(encoding="utf-8")
    else:
        # PDF input: optionally extract pages, then convert to markdown
        pdf_to_convert = input_path
        tmp_pdf: Path | None = None
        if args.pages:
            keep = parse_pages_range(args.pages)
            if not args.quiet:
                print(f"Extracting pages {args.pages} from {input_path}...", file=sys.stderr)
            tmp_pdf = Path(tempfile.mktemp(suffix=".pdf"))
            keep_pages(input_path, tmp_pdf, keep)
            pdf_to_convert = tmp_pdf
        if not args.quiet:
            print(f"Converting {pdf_to_convert} to markdown (Docling)...", file=sys.stderr)
        text = pdf_to_markdown(pdf_to_convert)
        if tmp_pdf and tmp_pdf.exists():
            tmp_pdf.unlink()
        if args.save_md:
            args.save_md.parent.mkdir(parents=True, exist_ok=True)
            args.save_md.write_text(text, encoding="utf-8")
            if not args.quiet:
                print(f"Saved markdown to {args.save_md}", file=sys.stderr)

    if not args.quiet:
        print(f"Parsing term-definition pairs...", file=sys.stderr)
    pairs = extract_term_definition_pairs(text)

    if not args.quiet:
        print(f"Found {len(pairs)} term-definition pairs", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
    elif args.format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(f"{p['term']}\n  {p['definition']}\n\n")
    else:  # terms-only
        with open(output_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(f"{p['term']}\n")

    if not args.quiet:
        print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
