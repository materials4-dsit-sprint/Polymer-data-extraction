# Polymer Paper Selection & Property Extraction

Extract polymer-related data from RSC (Royal Society of Chemistry) HTML papers. Pipeline: filter papers by term + ontology + embedding, then extract structured polymer properties (KnowMat-style schema) using OpenAI or Anthropic.

## Overview

1. **Filtering** – RSC HTML papers → term filter → ontology filter → embedding filter → intersection
2. **Extraction** – Parse HTML → subfield classification (experimental only) → LLM extraction from full text → JSON

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in project root:

```
OPENAI_API_KEY=sk-...          # For OpenAI extraction (gpt-4o, gpt-5.2)
ANTHROPIC_API_KEY=sk-...       # For Claude extraction (optional)
```

For subfield classification (experimental vs computational), run Ollama locally:

```bash
ollama pull deepseek-r1:8b
```

## Usage

### Filter papers

```bash
# Run all filters (term + ontology + embedding, intersection)
python run_all_filters.py corpus/2019 -o corpus/2019/papers_all_filters_passing --copy

# Individual filters (run as modules)
python -m filters.filter_polymer_papers corpus/2019/papers -o passing.txt
python -m filters.filter_by_ontology corpus/2019/papers_passing -o onto_passing.txt
python -m filters.filter_by_embedding --papers corpus/2019/papers_onto_passing -o embedding_passing.txt
```

### Extract properties

```bash
# OpenAI (recommended)
python extract_polymer_properties.py corpus/2019/papers_all_filters_passing -o extracted.json --extraction-model gpt-5.2

# Cheaper: gpt-4o, claude-3-5-sonnet, claude-haiku4.5
python extract_polymer_properties.py papers_folder -o out.json --extraction-model gpt-4o

# Note: Ollama models (llama, deepseek, qwen) are not recommended for extraction; use OpenAI or Anthropic.

# Extract only specified properties
python extract_polymer_properties.py papers_folder -o out.json --properties-list Tg Tm Mw Mn

# Skip subfield filter (faster, no Ollama)
python extract_polymer_properties.py papers_folder -o out.json --skip-subfield

# Limit papers for testing
python extract_polymer_properties.py papers_folder -o out.json --limit 5

# Resume interrupted extraction (skips already-processed papers)
python extract_polymer_properties.py papers_folder -o out.json --resume
```

## Project layout

```
├── config.py                    # Paths
├── extract_polymer_properties.py # Main extraction
├── run_all_filters.py           # Filter pipeline
├── filters/                     # Paper filters
│   ├── filter_polymer_papers.py # Term-based
│   ├── filter_by_ontology.py    # Ontology triples
│   └── filter_by_embedding.py   # SPECTER similarity
├── extraction/                  # TrustCall schemas, prompts
├── parsers/                     # RSC HTML parser
├── corpus/                      # Papers
├── purple_book/                 # IUPAC Purple Book terms
└── purple_book_terms.json
```

## Output schema

Each paper yields: `compositions` (polymer names), `processing_conditions`, `characterisation`, `properties_of_composition` with:

- `property_name`, `property_symbol`, `value`, `value_numeric`, `value_type`, `unit`
- `measurement_condition`, `additional_information`
- `confidence` (0–1): per-property heuristic score for downstream filtering
- `property_name_original` (when aligned): original name before ontology mapping
- `validation_warning`, `unit_warning` (when validation issues): optional post-processing flags

Post-processing includes: property ontology alignment (Purple Book mapping), validation of `value_numeric` and units (pint), and confidence scores.

## License

MIT
