"""Split documents into overlapping word-window chunks, keeping heading context."""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    section: str


_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def _sections(text: str):
    """Yield (section_title, section_text) pairs from markdown-ish text."""
    title, buf = "Introduction", []
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            if "".join(buf).strip():
                yield title, "\n".join(buf)
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        yield title, "\n".join(buf)


def chunk_document(text: str, source: str, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    chunks: list[Chunk] = []
    step = chunk_words - overlap
    for section, body in _sections(text):
        words = body.split()
        for start in range(0, max(len(words), 1), step):
            window = words[start : start + chunk_words]
            if not window:
                break
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{source}#{idx}",
                    # prefix the section title so both BM25 and embeddings see it
                    text=f"{section}: " + " ".join(window),
                    source=source,
                    chunk_index=idx,
                    section=section,
                )
            )
            if start + chunk_words >= len(words):
                break
    return chunks
