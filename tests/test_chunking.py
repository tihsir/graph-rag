"""Tests for chunking module."""

import pytest
from graphrag.app.data.chunking import Chunker, Chunk


def test_chunker_init():
    """Test chunker initialization."""
    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 50


def test_chunk_simple_text():
    """Test chunking simple text."""
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    text = "This is a test document. " * 20
    chunks = chunker.split_by_tokens(text, "doc1")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "doc1" for c in chunks)


def test_chunk_with_headings():
    """Test chunking text with markdown headings."""
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    text = """# Introduction
This is the introduction section.

## Subsection
This is a subsection.

# Conclusion
This is the conclusion."""
    chunks = chunker.split_by_headings(text, "doc1")
    assert len(chunks) > 0
    # Check that sections are preserved
    sections = [c.section for c in chunks if c.section]
    assert len(sections) > 0

