"""
DocIQ — Intelligent Chunker
Token-aware + Section-aware + Semantic chunking strategy.

Design Philosophy:
  1. Section-Aware: Never split across section boundaries if possible.
  2. Token-Aware: Respect LLM context window limits using tiktoken.
  3. Overlap: Sliding window overlap preserves cross-chunk context.
  4. Metadata-Rich: Each chunk carries section, position, and overlap info.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from core.document_processor import ParsedDocument, DocumentSection

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# DATA MODEL
# ──────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A semantically meaningful unit of text ready for embedding."""
    chunk_id: str                   # SHA256 hash of content
    text: str                       # raw chunk text
    section_title: str              # parent section heading
    section_level: int              # heading depth
    section_index: int              # position in document sections
    chunk_index: int                # position within section
    start_char: int                 # character offset in raw document
    end_char: int                   # character offset end
    token_count: int                # approximate token count
    is_overlap: bool = False        # True if this chunk is an overlap chunk
    doc_filename: str = ""
    doc_format: str = ""
    metadata: dict = field(default_factory=dict)

    # Contextual prefix injected during retrieval (not used for embedding)
    context_prefix: str = ""

    @property
    def retrieval_text(self) -> str:
        """Text used for retrieval context (includes section prefix)."""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.text}"
        return self.text

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "section_title": self.section_title,
            "section_level": self.section_level,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "doc_filename": self.doc_filename,
        }


# ──────────────────────────────────────────────
# TOKEN COUNTER
# ──────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Estimate token count. Uses tiktoken if available (accurate),
    otherwise falls back to a reliable heuristic (words * 1.33).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Heuristic: ~0.75 tokens per character on average
        return int(len(text) * 0.75)


def split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Splits text into token-bounded chunks with sliding window overlap.
    Uses sentence boundary awareness to avoid mid-sentence cuts.
    """
    # Sentence tokenization
    sentence_pattern = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n|(?<=:)\n"
    )
    sentences = sentence_pattern.split(text)
    # Restore split points
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences = []
    current_tokens = 0
    overlap_buffer = []

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # If a single sentence exceeds max_tokens, hard-split it
        if sent_tokens > max_tokens:
            # Flush current buffer first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            # Hard split the long sentence by words
            words = sent.split()
            sub_chunk = []
            sub_tokens = 0
            for word in words:
                wt = count_tokens(word)
                if sub_tokens + wt > max_tokens:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_tokens = wt
                else:
                    sub_chunk.append(word)
                    sub_tokens += wt
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            current_sentences = []
            current_tokens = 0
            continue

        if current_tokens + sent_tokens > max_tokens:
            # Emit current chunk
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            # Seed next chunk with overlap
            overlap_buffer = []
            overlap_t = 0
            for s in reversed(current_sentences):
                st = count_tokens(s)
                if overlap_t + st <= overlap_tokens:
                    overlap_buffer.insert(0, s)
                    overlap_t += st
                else:
                    break
            current_sentences = overlap_buffer + [sent]
            current_tokens = overlap_t + sent_tokens
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c for c in chunks if c.strip()]


# ──────────────────────────────────────────────
# CHUNKER
# ──────────────────────────────────────────────

class DocumentChunker:
    """
    Multi-strategy document chunker.

    Strategy:
      Phase 1: Split document into logical sections (from DocumentProcessor).
      Phase 2: Within each section, apply token-aware sliding window chunking.
      Phase 3: Enrich each chunk with section metadata and positional context.
      Phase 4: Generate deterministic chunk IDs for deduplication.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_tokens: int = 30,
        include_section_header: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.include_section_header = include_section_header

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Main entry point: returns all chunks for a parsed document."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        for section in doc.sections:
            section_chunks = self._chunk_section(
                section=section,
                doc=doc,
                global_chunk_index=global_chunk_index,
            )
            all_chunks.extend(section_chunks)
            global_chunk_index += len(section_chunks)

        logger.info(f"Chunked '{doc.filename}': {len(doc.sections)} sections → "
                    f"{len(all_chunks)} chunks "
                    f"(avg {len(doc.raw_text)/max(1,len(all_chunks)):.0f} chars/chunk)")
        return all_chunks

    def _chunk_section(
        self,
        section: DocumentSection,
        doc: ParsedDocument,
        global_chunk_index: int,
    ) -> List[DocumentChunk]:
        """Chunk a single document section with token awareness."""
        chunks: List[DocumentChunk] = []
        section_text = section.content

        # Remove the header line itself from content to avoid repetition
        first_line = section_text.split("\n")[0]
        if first_line.strip() == section.title.strip():
            section_text = "\n".join(section_text.split("\n")[1:]).strip()

        if not section_text:
            return chunks

        # Build context prefix for retrieval (injected at query time, not embedding time)
        context_prefix = (
            f"[Section: {section.title}]" if self.include_section_header else ""
        )

        raw_chunks = split_by_tokens(
            text=section_text,
            max_tokens=self.chunk_size,
            overlap_tokens=self.chunk_overlap,
        )

        for idx, chunk_text in enumerate(raw_chunks):
            token_count = count_tokens(chunk_text)

            # Discard trivially small chunks
            if token_count < self.min_chunk_tokens:
                continue

            # Embedding text: optionally prepend section header for richer embeddings
            embed_text = (
                f"{section.title}: {chunk_text}"
                if self.include_section_header
                else chunk_text
            )

            # Deterministic ID: hash of filename + chunk text
            chunk_id = hashlib.sha256(
                f"{doc.filename}::{chunk_text[:200]}".encode()
            ).hexdigest()[:16]

            # Estimate char positions (approximate)
            start_char = section.start_char + section_text.find(chunk_text[:50])
            end_char = start_char + len(chunk_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=embed_text,          # used for embedding
                section_title=section.title,
                section_level=section.level,
                section_index=section.section_index,
                chunk_index=global_chunk_index + idx,
                start_char=max(0, start_char),
                end_char=min(len(doc.raw_text), end_char),
                token_count=token_count,
                doc_filename=doc.filename,
                doc_format=doc.format,
                context_prefix=context_prefix,
                metadata={
                    "section_word_count": section.word_count,
                    "raw_chunk_text": chunk_text,  # original without prefix
                },
            )
            chunks.append(chunk)

        return chunks

    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> dict:
        if not chunks:
            return {}
        token_counts = [c.token_count for c in chunks]
        sections = list({c.section_title for c in chunks})
        return {
            "total_chunks": len(chunks),
            "total_sections": len(sections),
            "avg_tokens_per_chunk": f"{sum(token_counts)/len(token_counts):.0f}",
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "sections": sections,
        }
