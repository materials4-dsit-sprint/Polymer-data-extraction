"""
Parser for RSC (Royal Society of Chemistry) HTML papers.
Extracts all text organised into sections: Abstract, Introduction, Experimental, etc.

Requires: lxml (pip install lxml)
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional


def _normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


class _RSCHTMLParser(HTMLParser):
    """
    Standard library HTML parser for RSC structure.
    Collects sections by tracking h2/h3 headings and content between them.
    """

    def __init__(self):
        super().__init__()
        self.sections: dict[str, list[str]] = {}
        self.meta: dict[str, str] = {}
        self._current_section: Optional[str] = None
        self._current_content: list[str] = []
        self._in_abstract_div = False
        self._abstract_parts: list[str] = []
        self._in_h2_span = False
        self._in_h3_span = False
        self._heading_class: Optional[str] = None
        self._in_h1 = False
        self._in_article_info = False
        self._past_abstract = False
        self._in_table_caption = False
        self._in_table = False
        self._skip_until_close = 0  # Skip script/style content

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in ("script", "style", "noscript"):
            self._skip_until_close = 1
        if self._skip_until_close:
            return

        if tag == "div":
            cls = attrs_dict.get("class", "")
            if cls == "abstract":
                self._in_abstract_div = True
                self._abstract_parts = []
            elif "table_caption" in cls:
                self._in_table_caption = True
        elif tag == "table":
            self._in_table = True
        elif tag == "h1":
            self._in_h1 = True
        elif tag == "h2":
            self._in_h2_span = True
        elif tag == "h3":
            self._in_h3_span = True
        elif tag == "span":
            cls = attrs_dict.get("class", "")
            if cls == "a_heading" and self._in_h2_span:
                self._heading_class = "a"
            elif cls == "b_heading" and self._in_h3_span:
                self._heading_class = "b"

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip_until_close = max(0, self._skip_until_close - 1)
        if self._skip_until_close:
            return

        if tag == "div":
            if self._in_abstract_div:
                self._in_abstract_div = False
                if self._abstract_parts:
                    text = _normalize_text(" ".join(self._abstract_parts))
                    text = re.sub(r"^Abstract\s*", "", text, flags=re.I).strip()
                    if text:
                        self.sections["Abstract"] = [text]
                self._past_abstract = True
            elif self._in_table_caption:
                self._in_table_caption = False
        elif tag == "table":
            self._in_table = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "h2":
            self._in_h2_span = False
            self._heading_class = None
        elif tag == "h3":
            self._in_h3_span = False
            self._heading_class = None

    def handle_data(self, data):
        if self._skip_until_close:
            return
        text = data.strip()
        if not text:
            return

        if self._in_abstract_div:
            self._abstract_parts.append(text)
            return

        if self._in_h1:
            self.meta["title"] = text
            return

        if self._heading_class and (self._in_h2_span or self._in_h3_span):
            if text.lower() != "abstract":
                self._current_section = text
                self._current_content = []
            self._heading_class = None
            return

        if self._current_section and self._past_abstract:
            if not self._in_table_caption and not self._in_table and len(text) > 15:
                self._current_content.append(text)
                if self._current_section not in self.sections:
                    self.sections[self._current_section] = []
                self.sections[self._current_section] = self._current_content.copy()


def _parse_with_stdlib(html_content: str) -> tuple[dict[str, str], dict[str, str], list[dict]]:
    """Parse using standard library HTMLParser."""
    parser = _RSCHTMLParser()
    # Only parse body content - skip head to avoid script issues
    body_match = re.search(r"<body[^>]*>(.*)</body>", html_content, re.DOTALL | re.I)
    if body_match:
        content = body_match.group(1)
    else:
        content = html_content
    parser.feed(content)

    # Flatten section content
    sections = {}
    for name, parts in parser.sections.items():
        text = _normalize_text(" ".join(parts))
        if text:
            sections[name] = text

    # Extract tables via regex (stdlib has no table parsing)
    tables = _extract_tables_regex(content)
    return parser.meta, sections, tables


def _extract_tables_regex(html_content: str) -> list[dict]:
    """Extract tables and captions using regex for stdlib fallback."""
    tables = []
    # Match table_caption div followed by table (possibly with wrapper divs in between)
    caption_pattern = re.compile(
        r'<div[^>]*class="[^"]*table_caption[^"]*"[^>]*>(.*?)</div>',
        re.DOTALL | re.I,
    )
    table_pattern = re.compile(
        r'<table[^>]*class="[^"]*tgroup[^"]*"[^>]*>(.*?)</table>',
        re.DOTALL | re.I,
    )

    # Find all captions and tables in order
    captions = [(m.start(), _strip_html(m.group(1))) for m in caption_pattern.finditer(html_content)]
    table_matches = list(table_pattern.finditer(html_content))

    for i, table_m in enumerate(table_matches):
        caption = ""
        for cap_pos, cap_text in captions:
            if cap_pos < table_m.start() and cap_text:
                caption = cap_text
            elif cap_pos > table_m.start():
                break
        content = _strip_html(table_m.group(1))
        content = re.sub(r"\s+", " ", content).strip()
        tables.append({"caption": caption, "content": content})

    return tables


def _strip_html(html_str: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", html_str)
    return _normalize_text(text)


def _extract_table_content(table_elem, get_text) -> str:
    """Extract text from table cells (th, td), preserving row structure."""
    rows = []
    for tr in table_elem.xpath(".//tr"):
        cells = []
        for cell in tr.xpath("./th | ./td"):
            text = get_text(cell)
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _parse_with_lxml(
    html_content: bytes,
) -> tuple[dict[str, str], dict[str, str], list[dict]]:
    """Parse using lxml (preferred, more robust)."""
    from lxml import html as lxml_html
    from lxml import etree

    tree = lxml_html.fromstring(html_content)
    meta = {}
    sections = {}
    tables: list[dict] = []

    def get_text(elem):
        if elem is None:
            return ""
        t = etree.tostring(elem, method="text", encoding="unicode")
        return _normalize_text(t)

    # Meta
    for h1 in tree.xpath("//h1"):
        meta["title"] = get_text(h1)
        break
    for a in tree.xpath('//div[@class="article_info"]//span[@class="italic"]/a'):
        meta["journal"] = get_text(a)
        break
    for m in tree.xpath('//meta[@name="citation_doi"]/@content'):
        meta["doi"] = m
        break

    # Abstract
    for div in tree.xpath('//div[@class="abstract"]'):
        text = get_text(div)
        text = re.sub(r"^Abstract\s*", "", text, flags=re.I).strip()
        if text:
            sections["Abstract"] = text
        break

    # Tables: find all tables with class tgroup, get caption from preceding table_caption div
    for table in tree.xpath('//table[contains(@class, "tgroup")]'):
        caption = ""
        # Caption precedes table in document order - use XPath for robustness across RSC variants
        caption_divs = table.xpath(
            'preceding::div[contains(@class, "table_caption")][1]'
        )
        if caption_divs:
            caption = get_text(caption_divs[0])

        content = _extract_table_content(table, get_text)
        if caption or content:
            tables.append({"caption": caption, "content": content})

    # Body sections (exclude table captions from section text)
    headings = tree.xpath(
        '//div[@id="wrapper"]//h2[span[@class="a_heading"]] | '
        '//div[@id="wrapper"]//h3[span[@class="b_heading"]]'
    )

    for heading in headings:
        span = heading.find('.//span[@class="a_heading"]')
        if span is None:
            span = heading.find('.//span[@class="b_heading"]')
        if span is None:
            continue
        section_name = get_text(span).strip()
        if not section_name or section_name.lower() == "abstract":
            continue

        content_parts = []
        for sibling in heading.itersiblings():
            if sibling.tag in ("h2", "h3"):
                break
            if sibling.tag == "p":
                text = get_text(sibling)
                if len(text) > 10:
                    content_parts.append(text)
            elif sibling.tag == "span":
                text = get_text(sibling)
                if len(text) > 30:
                    content_parts.append(text)
            elif sibling.tag == "div":
                cls = sibling.get("class") or ""
                # Skip table_caption and table wrappers - captions belong to tables
                if "table_caption" in cls or "rtable" in cls or "image_table" in cls:
                    continue
                text = get_text(sibling)
                if len(text) > 50:
                    content_parts.append(text)

        text = _normalize_text(" ".join(content_parts))
        if text:
            sections[section_name] = text

    return meta, sections, tables


class RSCSectionParser:
    """
    Parser for RSC HTML papers that extracts text organised by sections.

    RSC HTML structure:
    - Abstract: <div class="abstract"> with <h2>Abstract</h2>
    - Main sections: <h2> with <span class="a_heading"> (Introduction, Experimental, etc.)
    - Subsections: <h3> with <span class="b_heading">
    - Content: <p class="otherpara">, <span>, etc.
    - Tables: separate sections with caption and content
    """

    def __init__(self, filepath: str | Path, include_tables: bool = True):
        self.filepath = Path(filepath)
        self.include_tables = include_tables  # If False, tables list will be empty
        self.sections: dict[str, str] = {}
        self.tables: list[dict] = []
        self.meta: dict[str, str] = {}
        self._use_lxml: Optional[bool] = None

    def parse(self) -> dict[str, str]:
        """Parse the HTML and return a dict of section_name -> text."""
        content = self.filepath.read_bytes()
        html_str = content.decode("utf-8", errors="replace")

        if self._use_lxml is None:
            try:
                import lxml  # noqa: F401
                self._use_lxml = True
            except ImportError:
                self._use_lxml = False

        if self._use_lxml:
            self.meta, self.sections, tables = _parse_with_lxml(content)
            self.tables = tables if self.include_tables else []
        else:
            self.meta, self.sections, tables = _parse_with_stdlib(html_str)
            self.tables = tables if self.include_tables else []

        return self.sections

    def get_sections(self) -> dict[str, str]:
        """Return parsed sections. Calls parse() if not yet parsed."""
        if not self.sections:
            self.parse()
        return self.sections

    def get_tables(self) -> list[dict]:
        """Return parsed tables (list of {caption, content}). Calls parse() if not yet parsed."""
        if not self.tables and not self.sections:
            self.parse()
        return self.tables

    def get_meta(self) -> dict[str, str]:
        """Return metadata."""
        if not self.sections and not self.meta:
            self.parse()
        return self.meta

    def to_dict(self) -> dict:
        """Return full parsed result with meta, sections, and tables."""
        self.get_sections()
        return {
            "meta": self.meta,
            "sections": self.sections,
            "tables": self.tables,
        }

    def __repr__(self) -> str:
        return f"RSCSectionParser({self.filepath})"


def parse_rsc_html(filepath: str | Path, include_tables: bool = True) -> dict[str, str]:
    """
    Convenience function to parse an RSC HTML file and return sections.

    Args:
        filepath: Path to the HTML file
        include_tables: Whether to extract tables (default True). Tables are returned
            separately via parser.get_tables() or parser.to_dict()['tables'].

    Returns:
        Dict mapping section names to their text content. Use RSCSectionParser.to_dict()
        for full output including tables and metadata.
    """
    parser = RSCSectionParser(filepath, include_tables=include_tables)
    return parser.parse()


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rsc_html_parser.py <path_to_rsc_html>")
        sys.exit(1)

    path = sys.argv[1]
    parser = RSCSectionParser(path)
    result = parser.to_dict()

    print(json.dumps(result, indent=2, ensure_ascii=False))
