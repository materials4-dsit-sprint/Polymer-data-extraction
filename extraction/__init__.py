"""Extraction module: TrustCall schemas and prompts (KnowMat-style)."""

from extraction.extractors import (
    CompositionList,
    CompositionProperties,
    Property,
    extract_with_trustcall,
)

__all__ = [
    "CompositionList",
    "CompositionProperties",
    "Property",
    "extract_with_trustcall",
]
