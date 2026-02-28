#!/usr/bin/env python3
"""
Aggregate extracted JSON into a wide-format DataFrame: one row per composition.

Columns: paper metadata, composition, processing_conditions, then for each property
(Tg, Tm, Mn, Mw, etc.): value, unit, confidence, measurement_condition, additional_information.

When a composition has multiple values for the same property (e.g. Tg for different blocks),
the first is used. Use aggregate_extracted.py for long format with all property instances.

Usage:
  python aggregate_compositions_wide.py extracted.json -o compositions_wide.csv
  python aggregate_compositions_wide.py extracted.json -o compositions_wide.parquet --properties Tg Tm Mn Mw
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def _sanitize_col(name: str) -> str:
    """Sanitize property name for column (e.g. 'glass transition temperature' -> 'Tg' when we use symbol)."""
    return re.sub(r"[^\w]", "_", str(name)).strip("_") or "prop"


def load_extracted(path: Path) -> list[dict]:
    """Load extracted JSON (list of paper results)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def aggregate_to_wide_dataframe(
    data: list[dict],
    properties: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build wide-format DataFrame: one row per composition.
    For each property: value, unit, confidence, measurement_condition, additional_information.
    """
    # Property symbol/name -> short column prefix (Tg, Tm, Mn, Mw, etc.)
    SYMBOL_MAP = {
        "glass transition temperature": "Tg",
        "melting temperature": "Tm",
        "number-average molecular weight": "Mn",
        "weight-average molecular weight": "Mw",
        "dispersity": "PDI",
        "degree of crystallinity": "crystallinity",
        "Young's modulus": "E",
        "tensile strength": "tensile_strength",
        "viscosity": "viscosity",
        "thermal conductivity": "thermal_conductivity",
    }

    def _prop_prefix(prop: dict) -> str:
        sym = (prop.get("property_symbol") or "").strip()
        if sym:
            return sym.replace("/", "_")
        name = (prop.get("property_name") or "").lower()
        return SYMBOL_MAP.get(name, _sanitize_col(name))[:20]

    rows = []
    for paper in data:
        paper_meta = {
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "source_file": paper.get("source_file"),
            "subfield": paper.get("subfield"),
        }
        for comp in paper.get("compositions", []):
            row = {
                **paper_meta,
                "composition": comp.get("composition"),
                "composition_standard": comp.get("composition_standard"),
                "processing_conditions": comp.get("processing_conditions"),
            }
            # Collect properties for this composition
            props_by_prefix: dict[str, dict] = {}
            for prop in comp.get("properties_of_composition", []):
                prefix = _prop_prefix(prop)
                props_ok = [p.strip() for p in properties] if properties else None
                if props_ok and prefix not in props_ok:
                    continue
                if prefix not in props_by_prefix:
                    props_by_prefix[prefix] = prop
                # Keep first occurrence when multiple values for same property

            for prefix, prop in props_by_prefix.items():
                row[f"{prefix}"] = prop.get("value_numeric") or prop.get("value")
                row[f"{prefix}_unit"] = prop.get("unit")
                row[f"{prefix}_confidence"] = prop.get("confidence")
                row[f"{prefix}_measurement_condition"] = prop.get("measurement_condition")
                row[f"{prefix}_additional_information"] = prop.get("additional_information")
                row[f"{prefix}_value_original"] = prop.get("value")
                row[f"{prefix}_value_error"] = prop.get("value_error")

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Reorder: meta first, then property columns (Tg, Tg_unit, ... Tm, Tm_unit, ...)
    meta_cols = ["doi", "title", "source_file", "subfield", "composition", "composition_standard", "processing_conditions"]
    priority_prefixes = ["Tg", "Tm", "Mn", "Mw", "PDI", "crystallinity", "E", "tensile_strength", "viscosity", "thermal_conductivity"]
    suffixes = ["", "_unit", "_confidence", "_measurement_condition", "_additional_information", "_value_original", "_value_error"]
    prop_cols = [c for c in df.columns if c not in meta_cols]
    ordered_props = []
    for prefix in priority_prefixes:
        for suf in suffixes:
            c = prefix + suf if suf else prefix
            if c in prop_cols and c not in ordered_props:
                ordered_props.append(c)
    for c in sorted(prop_cols):
        if c not in ordered_props:
            ordered_props.append(c)
    final_cols = [c for c in meta_cols if c in df.columns] + ordered_props
    return df[[c for c in final_cols if c in df.columns]]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate extracted JSON into wide-format DataFrame (one row per composition)"
    )
    parser.add_argument(
        "input",
        type=Path,
        default=Path("extracted.json"),
        nargs="?",
        help="Input JSON file (default: extracted.json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (CSV or Parquet). If omitted, print to stdout.",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=None,
        help="Only include these properties (e.g. Tg Tm Mn Mw). Default: all.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        help="Output format (inferred from -o extension if not set)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    data = load_extracted(args.input)
    df = aggregate_to_wide_dataframe(data, properties=args.properties)

    if args.output:
        fmt = args.format or args.output.suffix.lstrip(".").lower()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"Wrote {len(df)} rows to {args.output}", file=sys.stderr)
    else:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
