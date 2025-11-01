#!/usr/bin/env python3
"""
Quick setup script to prepare PubMed test data for GraphRAG.
This fetches PubMed abstracts and makes them ready for ingestion.
"""

import asyncio
import glob
from pathlib import Path
from graphrag.tests.pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries
from graphrag.app.services.pipeline import pipeline

async def main():
    print("=" * 60)
    print("GraphRAG Test Data Setup")
    print("=" * 60)
    
    # Step 1: Fetch PubMed data
    print("\n[1/3] Fetching PubMed data...")
    test_dir = "./test_data/pubmed"
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    existing_files = glob.glob(f"{test_dir}/pubmed_*.md")
    if existing_files:
        print(f"  Found {len(existing_files)} existing PubMed files")
        response = input("  Fetch new data anyway? (y/N): ").strip().lower()
        if response != 'y':
            files = existing_files
            print(f"  Using existing files")
        else:
            queries = get_sample_medical_queries()
            files = create_pubmed_test_documents(queries, output_dir=test_dir)
            print(f"  Created {len(files)} new PubMed documents")
    else:
        queries = get_sample_medical_queries()
        files = create_pubmed_test_documents(queries, output_dir=test_dir)
        print(f"  Created {len(files)} PubMed documents")
    
    # Step 2: Ingest documents
    print("\n[2/3] Ingesting documents...")
    doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "test"])
    print(f"  ✓ Ingested {len(doc_ids)} documents")
    
    # Step 3: Build graph (optional)
    print("\n[3/3] Building knowledge graph...")
    response = input("  Build knowledge graph now? This may take a few minutes (y/N): ").strip().lower()
    if response == 'y':
        result = await pipeline.rebuild_graph(min_conf=0.4, batch_size=64)
        print(f"  ✓ Graph built: {result['entities']} entities, {result['relations']} relations")
    else:
        print("  Skipped. Run 'grag build-graph' when ready.")
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now:")
    print("  - Query: grag query 'What are treatments for asthma?'")
    print("  - Check status: grag status")
    print("  - Start API: uvicorn graphrag.app.api.routers:app --reload")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

