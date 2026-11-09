"""HTML-to-text conversion and chunking for Copom documents."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "tr", "table", "ul", "ol"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts).replace("​", "").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    paragraphs = [line for line in lines if line]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, max_chars: int = 1200, overlap_paragraphs: int = 1) -> list[str]:
    """Group paragraphs into chunks of up to ``max_chars``.

    Consecutive chunks share ``overlap_paragraphs`` paragraphs so an idea split across a
    boundary is still retrievable. A single paragraph longer than ``max_chars`` is split on
    sentence boundaries.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    paragraphs: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) <= max_chars:
            paragraphs.append(para)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.;!?])\s+", para):
            if current and len(current) + 1 + len(sentence) > max_chars:
                paragraphs.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            paragraphs.append(current)

    chunks: list[str] = []
    window: list[str] = []
    for para in paragraphs:
        candidate = "\n\n".join([*window, para])
        if window and len(candidate) > max_chars:
            chunks.append("\n\n".join(window))
            window = window[-overlap_paragraphs:] if overlap_paragraphs else []
            if window and len("\n\n".join([*window, para])) > max_chars:
                window = []
        window.append(para)
    if window:
        chunks.append("\n\n".join(window))
    return chunks
