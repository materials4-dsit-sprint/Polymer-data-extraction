#!/usr/bin/env python3
"""
Filter papers by semantic similarity to example polymer papers.

Uses SPECTER (sentence-transformers/allenai-specter), a dense embedding model
trained on scientific papers (title + abstract). Encodes example papers from
corpus/examples, then keeps papers from corpus/2019/papers_passing that are
close to the example centroid or to any example.

Usage:
  python -m filters.filter_by_embedding --percentile 30
  python -m filters.filter_by_embedding --papers corpus/2019/papers_onto_passing -o out.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT
from parsers.rsc_html_parser import RSCSectionParser

EXAMPLES_DIR = PROJECT_ROOT / "corpus" / "examples"
PAPERS_DIR = PROJECT_ROOT / "corpus" / "2019" / "papers_passing"
DEFAULT_OUTPUT = PROJECT_ROOT / "corpus" / "2019" / "papers_embedding_filtered.txt"


def get_paper_text(parser_result: dict, include_intro: bool = False) -> str:
    """Extract title + abstract (+ optional intro) for embedding."""
    meta = parser_result.get("meta", {})
    sections = parser_result.get("sections", {})
    title = meta.get("title", "")
    abstract = sections.get("Abstract", "")
    parts = [title, abstract]
    if include_intro:
        intro = sections.get("Introduction", "") or sections.get("1. Introduction", "")
        if intro:
            parts.append(intro)
    return " [SEP] ".join(p for p in parts if p.strip())


def load_model(model_name: str = "sentence-transformers/allenai-specter"):
    """Load SPECTER embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)
    return SentenceTransformer(model_name)


def encode_papers(
    html_paths: list[Path],
    model,
    include_intro: bool = False,
    verbose: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Parse papers, extract text, encode. Returns (paths_as_str, embeddings)."""
    texts: list[str] = []
    valid_paths: list[Path] = []

    for i, path in enumerate(html_paths):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Parsed {i + 1}/{len(html_paths)}...", file=sys.stderr)
        try:
            parser = RSCSectionParser(path, include_tables=False)
            result = parser.to_dict()
            text = get_paper_text(result, include_intro=include_intro)
            if not text or len(text.strip()) < 50:
                continue
            texts.append(text)
            valid_paths.append(path)
        except Exception as e:
            if verbose:
                print(f"  Skip {path.name}: {e}", file=sys.stderr)
            continue

    if not texts:
        return [], np.array([])

    if verbose:
        print(f"  Encoding {len(texts)} papers...", file=sys.stderr)
    embeddings = model.encode(texts, show_progress_bar=verbose)

    return [str(p) for p in valid_paths], embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def main():
    parser = argparse.ArgumentParser(
        description="Filter papers by similarity to example polymer papers (SPECTER embeddings)"
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=EXAMPLES_DIR,
        help=f"Folder with example papers (default: corpus/examples)",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=PAPERS_DIR,
        help="Folder with papers to filter",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file with passing paper paths",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Min cosine similarity (default: 0.6)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        metavar="P",
        help="Keep top P%% most similar papers (e.g. 30). Overrides threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Alternatively, keep top K most similar papers (overrides threshold)",
    )
    parser.add_argument(
        "--strategy",
        choices=["centroid", "max"],
        default="max",
        help="centroid: similarity to mean of examples. max: max similarity to any example (default)",
    )
    parser.add_argument(
        "--include-intro",
        action="store_true",
        help="Include introduction in text for embedding (longer, slower)",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Write paper -> similarity score to this file",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if not args.examples.exists():
        print(f"Examples folder not found: {args.examples}", file=sys.stderr)
        sys.exit(1)
    if not args.papers.exists():
        print(f"Papers folder not found: {args.papers}", file=sys.stderr)
        sys.exit(1)

    example_files = sorted(args.examples.glob("*.html"))
    paper_files = sorted(args.papers.glob("*.html"))

    if not example_files:
        print(f"No HTML files in {args.examples}", file=sys.stderr)
        sys.exit(1)
    if not paper_files:
        print(f"No HTML files in {args.papers}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Loading SPECTER model...", file=sys.stderr)
    model = load_model()

    if verbose:
        print(f"Encoding {len(example_files)} example papers...", file=sys.stderr)
    example_paths, example_emb = encode_papers(
        example_files, model, include_intro=args.include_intro, verbose=verbose
    )
    if len(example_emb) == 0:
        print("No valid example papers.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Encoding {len(paper_files)} candidate papers...", file=sys.stderr)
    paper_paths, paper_emb = encode_papers(
        paper_files, model, include_intro=args.include_intro, verbose=verbose
    )
    if len(paper_emb) == 0:
        print("No valid papers to filter.", file=sys.stderr)
        sys.exit(1)

    paper_emb_np = np.array(paper_emb)
    if args.strategy == "centroid":
        centroid = np.mean(example_emb, axis=0)
        similarities = np.array([
            cosine_similarity(centroid, paper_emb_np[i]) for i in range(len(paper_emb_np))
        ])
    else:
        example_emb_np = np.array(example_emb)
        ex_norm = example_emb_np / (np.linalg.norm(example_emb_np, axis=1, keepdims=True) + 1e-9)
        pa_norm = paper_emb_np / (np.linalg.norm(paper_emb_np, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.dot(pa_norm, ex_norm.T)
        similarities = np.max(sim_matrix, axis=1)

    if args.percentile is not None:
        k = max(1, int(len(paper_paths) * args.percentile / 100))
        top_indices = np.argsort(similarities)[::-1][:k]
        passing = [paper_paths[i] for i in top_indices]
        passing_scores = [(paper_paths[i], similarities[i]) for i in top_indices]
    elif args.top_k is not None:
        k = min(args.top_k, len(paper_paths))
        top_indices = np.argsort(similarities)[::-1][:k]
        passing = [paper_paths[i] for i in top_indices]
        passing_scores = [(paper_paths[i], similarities[i]) for i in top_indices]
    else:
        passing = [paper_paths[i] for i in range(len(paper_paths)) if similarities[i] >= args.threshold]
        passing_scores = [(paper_paths[i], similarities[i]) for i in range(len(paper_paths)) if similarities[i] >= args.threshold]
        passing_scores.sort(key=lambda x: x[1], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in passing:
            f.write(f"{Path(p).name}\n")

    if verbose:
        print(f"\nSimilarity range: min={similarities.min():.3f} max={similarities.max():.3f} median={np.median(similarities):.3f}", file=sys.stderr)
        print(f"Passed: {len(passing)} / {len(paper_paths)}", file=sys.stderr)
        if passing_scores:
            print(f"Top 5 scores: {[f'{s:.3f}' for _, s in passing_scores[:5]]}", file=sys.stderr)
        if len(passing) == len(paper_paths) and args.percentile is None and args.top_k is None:
            print("Tip: All papers passed. Try --percentile 30 or --threshold 0.7 to filter.", file=sys.stderr)
        print(f"Wrote {args.output}", file=sys.stderr)

    if args.scores:
        args.scores.parent.mkdir(parents=True, exist_ok=True)
        all_scores = [(paper_paths[i], float(similarities[i])) for i in range(len(paper_paths))]
        all_scores.sort(key=lambda x: x[1], reverse=True)
        with open(args.scores, "w") as f:
            for p, s in all_scores:
                f.write(f"{Path(p).name}\t{s:.4f}\n")
        if verbose:
            print(f"Wrote all scores to {args.scores}", file=sys.stderr)


if __name__ == "__main__":
    main()
