"""
Post-processing for extracted polymer data.

- Property ontology alignment: map to controlled vocabulary (Purple Book, KnowMat)
- Validation: value_numeric consistency, unit validation (pint)
- Confidence scores: heuristic per-property confidence
"""

import json
import re
from pathlib import Path

PROPERTY_ONTOLOGY_PATH = Path(__file__).parent / "property_ontology.json"

# Unit normalization: common variants -> canonical (before pint fallback)
UNIT_ALIASES = {
    "°c": "°C", "deg c": "°C", "celsius": "°C",
    "k": "K", "kelvin": "K",
    "g/mol": "g/mol", "da": "Da", "kda": "kDa", "kg/mol": "kg/mol",
    "mpa": "MPa", "gpa": "GPa", "pa": "Pa",
    "wt%": "wt%", "mol%": "mol%", "%": "%",
}


def _load_ontology() -> dict:
    """Load property ontology mappings."""
    if not PROPERTY_ONTOLOGY_PATH.exists():
        return {"mappings": {}}
    with open(PROPERTY_ONTOLOGY_PATH, encoding="utf-8") as f:
        return json.load(f)


def align_property_name(name: str, symbol: str | None, ontology: dict) -> str:
    """Map property name/symbol to standard vocabulary (Purple Book, KnowMat)."""
    mappings = ontology.get("mappings", {})
    name_str = (name or "").strip()
    name_lower = name_str.lower()
    symbol_str = (symbol or "").strip()
    # Try symbol first (often more reliable)
    if symbol_str and symbol_str in mappings:
        return mappings[symbol_str]
    # Try exact match on name (case-insensitive)
    for k, v in mappings.items():
        if k.lower() == name_lower:
            return v
    # Try partial match (e.g. "glass transition" in "glass transition temperature")
    for k, v in mappings.items():
        if k.lower() in name_lower or name_lower in k.lower():
            return v
    return name_str or ""  # Keep original if no match


def validate_value_numeric(value: str | None, value_numeric: float | None, value_type: str) -> tuple[float | None, str | None]:
    """
    Validate value_numeric against value. Returns (corrected_value_numeric, error_msg).
    """
    if value is None or value_type == "missing":
        return (None, None) if value_numeric is None else (None, "value_numeric should be null when value is missing")
    if value_type == "qualitative":
        return (0.0, None) if value_numeric in (None, 0.0) else (value_numeric, "qualitative typically maps to 0.0")
    # Parse numeric from value
    try:
        if value_type == "exact":
            num = float(re.sub(r"[^\d.eE+-]", "", value) or "0")
            if value_numeric is not None and abs(num - value_numeric) > 0.01:
                return (num, f"value_numeric {value_numeric} inconsistent with value '{value}'")
            return (num if value_numeric is None else value_numeric, None)
        if value_type == "lower_bound":
            m = re.search(r">\s*([\d.]+)", value)
            bound = float(m.group(1)) if m else (value_numeric or 0)
            return (bound, None)
        if value_type == "upper_bound":
            m = re.search(r"<\s*([\d.]+)", value)
            bound = float(m.group(1)) if m else (value_numeric or 0)
            return (bound, None)
        if value_type == "range":
            parts = re.findall(r"[\d.]+", value)
            if len(parts) >= 2:
                mid = (float(parts[0]) + float(parts[1])) / 2
                return (mid, None)
    except (ValueError, TypeError):
        pass
    return (value_numeric, None)


def validate_unit(unit: str | None) -> tuple[str | None, str | None]:
    """Normalize unit and check validity. Returns (normalized_unit, error_msg)."""
    if not unit or not str(unit).strip():
        return (None, None)
    u = str(unit).strip()
    u_lower = u.lower()
    if u_lower in UNIT_ALIASES:
        return (UNIT_ALIASES[u_lower], None)
    # Try pint if available
    try:
        import pint
        ureg = pint.UnitRegistry()
        parsed = ureg(u)
        return (str(parsed.units), None)
    except ImportError:
        return (u, None)
    except Exception:
        return (u, f"unit '{u}' may be invalid")


def compute_confidence(prop: dict) -> float:
    """
    Heuristic confidence score 0-1 for a property.
    - Has value + unit + value_numeric: higher
    - Has value_type exact: higher
    - Missing value or qualitative: lower
    """
    score = 0.5  # base
    if prop.get("value") and prop.get("value_type") != "missing":
        score += 0.2
    if prop.get("unit"):
        score += 0.1
    if prop.get("value_numeric") is not None and prop.get("value_type") == "exact":
        score += 0.2
    if prop.get("measurement_condition"):
        score += 0.05
    if prop.get("property_symbol"):
        score += 0.05
    return min(1.0, score)


def post_process_compositions(data: dict, align_ontology: bool = True, validate: bool = True, add_confidence: bool = True) -> dict:
    """Apply all post-processing to extracted compositions."""
    ontology = _load_ontology() if align_ontology else {}
    for comp in data.get("compositions", []):
        for prop in comp.get("properties_of_composition", []):
            # Ontology alignment
            if align_ontology and ontology.get("mappings"):
                original = prop.get("property_name", "")
                aligned = align_property_name(original, prop.get("property_symbol"), ontology)
                if aligned != original:
                    prop["property_name_original"] = original
                prop["property_name"] = aligned
            # Validation
            if validate:
                vn, err = validate_value_numeric(
                    prop.get("value"),
                    prop.get("value_numeric"),
                    prop.get("value_type", "exact"),
                )
                if vn is not None:
                    prop["value_numeric"] = vn
                if err:
                    prop["validation_warning"] = err
                u, uerr = validate_unit(prop.get("unit"))
                if u is not None:
                    prop["unit"] = u
                if uerr:
                    prop["unit_warning"] = uerr
            # Confidence
            if add_confidence:
                prop["confidence"] = round(compute_confidence(prop), 2)
    return data
