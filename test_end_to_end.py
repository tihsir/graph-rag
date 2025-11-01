#!/usr/bin/env python3
"""
End-to-end test script using PubMed data.
This script demonstrates the complete GraphRAG workflow.
"""

import asyncio
import glob
from pathlib import Path
from graphrag.tests.pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries
from graphrag.app.services.pipeline import pipeline
from graphrag.app.api.routers import query
from graphrag.app.api.schemas import QueryRequest


async def main():
    print("=" * 70)
    print("GraphRAG End-to-End Test with PubMed Data")
    print("=" * 70)
    
    # Step 1: Setup PubMed data
    print("\n[Step 1/5] Fetching PubMed data...")
    test_dir = "./test_data/pubmed"
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    existing_files = glob.glob(f"{test_dir}/pubmed_*.md")
    if existing_files:
        print(f"  Using {len(existing_files)} existing PubMed files")
        files = existing_files
    else:
        queries = get_sample_medical_queries()[:2]  # Use 2 queries for quick test
        files = create_pubmed_test_documents(queries, output_dir=test_dir)
        print(f"  Created {len(files)} PubMed documents")
    
    # Step 2: Ingest
    print("\n[Step 2/5] Ingesting documents...")
    doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "test"])
    print(f"  ✓ Ingested {len(doc_ids)} documents")
    
    # Step 3: Build graph
    print("\n[Step 3/5] Building knowledge graph...")
    result = await pipeline.rebuild_graph(min_conf=0.3, batch_size=32)  # Lower threshold for testing
    print(f"  ✓ Extracted {result['entities']} entities and {result['relations']} relations")
    
    # Step 4: Test queries
    print("\n[Step 4/5] Testing queries...")
    test_queries = [
        "What are the treatments for asthma?",
        "How is diabetes managed?",
    ]
    
    for q in test_queries:
        print(f"\n  Query: {q}")
        request = QueryRequest(query=q, mode="auto", k=5, rerank=True)
        try:
            response = await query(request)
            print(f"    Route: {response.route}")
            print(f"    Answer (first 150 chars): {response.answer[:150]}...")
            print(f"    Citations: {len(response.citations)}")
        except Exception as e:
            print(f"    Error: {str(e)}")
    
    # Step 5: Status
    print("\n[Step 5/5] Final status...")
    from graphrag.app.data.metadata_store import DocumentModel, ChunkModel, metadata_store
    session = metadata_store.get_session()
    try:
        doc_count = session.query(DocumentModel).count()
        chunk_count = session.query(ChunkModel).count()
        print(f"  Documents in store: {doc_count}")
        print(f"  Chunks in store: {chunk_count}")
    finally:
        session.close()
    
    print("\n" + "=" * 70)
    print("End-to-end test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

