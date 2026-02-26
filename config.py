"""Project configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Corpus
CORPUS_DIR = PROJECT_ROOT / "corpus"
DEFAULT_PAPERS_FOLDER = CORPUS_DIR / "2019" / "papers_all_filters_passing"
DEFAULT_PAPERS_PASSING = CORPUS_DIR / "2019" / "papers_passing"
DEFAULT_PAPERS_INPUT = CORPUS_DIR / "2019"  # Raw papers for filter_polymer_papers

# Purple Book
PURPLE_BOOK_TERMS_FILE = PROJECT_ROOT / "purple_book_terms.json"
PURPLE_BOOK_SECTIONS_MD = PROJECT_ROOT / "purple_book" / "sections_md"

# Extraction output
DEFAULT_EXTRACTION_OUTPUT = PROJECT_ROOT / "corpus" / "polymer_properties_extracted.json"
