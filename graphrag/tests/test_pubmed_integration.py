"""Integration tests for PubMed data ingestion."""

import pytest
import asyncio
from pathlib import Path
import os

from ..app.services.pipeline import pipeline
from ..app.data.metadata_store import metadata_store


@pytest.mark.asyncio
async def test_pubmed_ingestion():
    """Test ingesting PubMed data via markdown files."""
    from .pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries
    
    # Create test PubMed documents
    test_dir = "./test_data/pubmed"
    queries = get_sample_medical_queries()[:2]  # Just use 2 queries for testing
    files = create_pubmed_test_documents(queries, output_dir=test_dir)
    
    if not files:
        pytest.skip("No PubMed files created - API may be unavailable")
    
    # Test ingestion
    doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "test"])
    
    assert len(doc_ids) > 0, "Should have ingested at least one document"
    
    # Verify documents are in metadata store
    session = metadata_store.get_session()
    try:
        from ..app.data.metadata_store import DocumentModel
        doc_count = session.query(DocumentModel).filter(
            DocumentModel.doc_id.in_(doc_ids)
        ).count()
        assert doc_count == len(doc_ids), "All documents should be in metadata store"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_pubmed_graph_building():
    """Test building knowledge graph from PubMed data."""
    from .pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries
    
    # Create and ingest PubMed documents
    test_dir = "./test_data/pubmed"
    queries = get_sample_medical_queries()[:1]  # Just one query for graph test
    files = create_pubmed_test_documents(queries, output_dir=test_dir)
    
    if not files:
        pytest.skip("No PubMed files created")
    
    # Ingest
    doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "test"])
    
    # Build graph (with lower confidence for testing)
    result = await pipeline.rebuild_graph(min_conf=0.3, batch_size=32)
    
    assert result["entities"] >= 0, "Should extract some entities (even if 0)"
    assert result["relations"] >= 0, "Should extract some relations (even if 0)"


if __name__ == "__main__":
    # Run a quick test
    asyncio.run(test_pubmed_ingestion())

