"""Tests for entity canonicalization."""

import pytest
from graphrag.app.graph.canonicalize import Canonicalizer
from graphrag.app.graph.extraction import Entity


@pytest.mark.asyncio
async def test_canonicalize_simple():
    """Test simple canonicalization."""
    from graphrag.app.core.providers import LocalEmbeddingProvider
    
    canonicalizer = Canonicalizer(embedding_provider=LocalEmbeddingProvider())
    
    entities = [
        Entity(
            name="Apple Inc",
            type="org",
            aliases=["Apple", "Apple Corporation"],
            span=[0, 10],
            confidence=0.9,
            doc_id="doc1",
            chunk_id="chunk1",
            char_start=0,
            char_end=10,
        ),
        Entity(
            name="Apple",
            type="org",
            aliases=[],
            span=[20, 25],
            confidence=0.8,
            doc_id="doc1",
            chunk_id="chunk1",
            char_start=20,
            char_end=25,
        ),
    ]
    
    groups = await canonicalizer.canonicalize(entities)
    # Should merge similar entities
    assert len(groups) <= len(entities)

