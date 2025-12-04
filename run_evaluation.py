#!/usr/bin/env python3
"""
Run comprehensive evaluation comparing GraphRAG vs Vanilla RAG.

This script:
1. Ingests PubMed test data (if not already done)
2. Builds the knowledge graph (if not already done)
3. Runs evaluation on both GraphRAG and Vanilla RAG
4. Outputs detailed comparison metrics

Usage:
    python run_evaluation.py                    # Full evaluation
    python run_evaluation.py --quick            # Quick test with 3 queries
    python run_evaluation.py --no-ingest        # Skip ingestion (use existing data)
    python run_evaluation.py --no-graph         # Skip graph building
    python run_evaluation.py --no-llm-judge     # Skip LLM-based answer judging
    python run_evaluation.py --vanilla-only     # Only run vanilla RAG evaluation
    python run_evaluation.py --graphrag-only    # Only run GraphRAG evaluation
"""

import asyncio
import argparse
import glob
import sys
from pathlib import Path
from datetime import datetime


def check_data_status():
    """Check current data status in the database."""
    from graphrag.app.data.metadata_store import DocumentModel, ChunkModel, metadata_store
    
    session = metadata_store.get_session()
    try:
        doc_count = session.query(DocumentModel).count()
        chunk_count = session.query(ChunkModel).count()
    finally:
        session.close()
    
    return doc_count, chunk_count


async def setup_data(skip_ingest: bool = False, skip_graph: bool = False):
    """Setup test data by ingesting documents and building graph."""
    from graphrag.app.services.pipeline import pipeline
    from graphrag.app.data.metadata_store import DocumentModel, ChunkModel, metadata_store
    
    test_dir = "./test_data/pubmed"
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Check existing data
    doc_count, chunk_count = check_data_status()
    
    print(f"\n📊 Current data status:")
    print(f"   Documents: {doc_count}")
    print(f"   Chunks: {chunk_count}")
    
    # Ingest if needed
    if not skip_ingest and (doc_count == 0 or chunk_count == 0):
        print("\n📥 Setting up test data...")
        
        # Check for existing files
        existing_files = glob.glob(f"{test_dir}/pubmed_*.md")
        if existing_files:
            print(f"   Found {len(existing_files)} existing PubMed files")
            files = existing_files
        else:
            # Try to create documents from pubmed
            try:
                from graphrag.tests.pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries
                print("   Fetching PubMed articles...")
                queries = get_sample_medical_queries()[:2]  # Use 2 queries
                files = create_pubmed_test_documents(queries, output_dir=test_dir)
                print(f"   Created {len(files)} PubMed documents")
            except ImportError:
                print("   ⚠️  PubMed test data module not available")
                print("   Please add test documents manually to ./test_data/pubmed/")
                return False
        
        if not files:
            print("   ⚠️  No files to ingest")
            return False
        
        print("\n📝 Ingesting documents...")
        try:
            doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "eval"])
            print(f"   ✓ Ingested {len(doc_ids)} documents")
        except Exception as e:
            print(f"   ❌ Ingestion failed: {e}")
            return False
    elif skip_ingest:
        print("\n⏭️  Skipping ingestion (--no-ingest)")
    else:
        print("\n✓ Data already exists, skipping ingestion")
    
    # Build graph if needed
    if not skip_graph:
        # Check if graph exists
        from graphrag.app.services.pipeline import get_graph_store
        try:
            graph_store = get_graph_store()
            # Try to get node count
            node_count = 0
            if hasattr(graph_store, 'graph') and hasattr(graph_store.graph, 'number_of_nodes'):
                node_count = graph_store.graph.number_of_nodes()
            
            if node_count == 0:
                print("\n🔗 Building knowledge graph...")
                result = await pipeline.rebuild_graph(min_conf=0.3, batch_size=32)
                
                if "error" in result:
                    print(f"   ⚠️  Graph building skipped: {result.get('message', result['error'])}")
                    print("   (GraphRAG will fall back to vector search)")
                else:
                    print(f"   ✓ Extracted {result['entities']} entities and {result['relations']} relations")
            else:
                print(f"\n✓ Graph already exists ({node_count} nodes)")
        except Exception as e:
            print(f"\n⚠️  Graph building failed: {e}")
            print("   (GraphRAG will fall back to vector search)")
    else:
        print("\n⏭️  Skipping graph building (--no-graph)")
    
    return True


async def run_full_evaluation(
    quick: bool = False,
    use_llm_judge: bool = True,
    output_dir: str = "./eval_results",
    num_queries: int = None,
    k: int = 10,
    rerank: bool = True
):
    """Run the full evaluation pipeline."""
    from graphrag.app.eval import (
        run_evaluation,
        run_quick_eval,
        create_pubmed_evaluation_dataset
    )
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("🧪 RUNNING EVALUATION: GraphRAG vs Vanilla RAG")
    print("=" * 70)
    
    try:
        if quick:
            print("\n⚡ Quick mode: evaluating 3 queries")
            report = await run_quick_eval(num_queries=3)
        else:
            dataset = create_pubmed_evaluation_dataset()
            
            # Limit queries if requested
            if num_queries is not None and num_queries < len(dataset.qa_pairs):
                dataset.qa_pairs = dataset.qa_pairs[:num_queries]
                print(f"\n📋 Dataset: {dataset.name} (limited to {num_queries} queries)")
            else:
                print(f"\n📋 Dataset: {dataset.name}")
            
            print(f"   Queries: {len(dataset.qa_pairs)}")
            print(f"   Domain: {dataset.domain}")
            print(f"   k={k}, rerank={rerank}, llm_judge={use_llm_judge}")
            
            print("\n🔄 Running evaluation (this may take a few minutes)...")
            report = await run_evaluation(
                dataset=dataset,
                k=k,
                rerank=rerank,
                use_llm_judge=use_llm_judge,
                output_path=f"{output_dir}/eval_report_{timestamp}.json",
                update_gold_from_chunks=True
            )
        
        # Print summary
        report.print_summary()
        
        # Save report
        report_path = f"{output_dir}/eval_report_{timestamp}.json"
        report.save(report_path)
        print(f"\n💾 Full report saved to: {report_path}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    parser = argparse.ArgumentParser(description="Run GraphRAG vs Vanilla RAG evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick test with 3 queries")
    parser.add_argument("--no-ingest", action="store_true", help="Skip data ingestion")
    parser.add_argument("--no-graph", action="store_true", help="Skip graph building")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-based answer judging")
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory for reports")
    parser.add_argument("--num-queries", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GraphRAG Evaluation Suite")
    print("=" * 70)
    
    # Setup data
    success = await setup_data(
        skip_ingest=args.no_ingest,
        skip_graph=args.no_graph
    )
    
    if not success:
        print("\n❌ Setup failed. Cannot run evaluation.")
        return 1
    
    # Check if we have data
    doc_count, chunk_count = check_data_status()
    if chunk_count == 0:
        print("\n❌ No chunks found. Please ingest data first.")
        print("   Run: python run_evaluation.py (without --no-ingest)")
        return 1
    
    # Run evaluation
    report = await run_full_evaluation(
        quick=args.quick,
        use_llm_judge=not args.no_llm_judge,
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        k=args.k,
        rerank=not args.no_rerank
    )
    
    if report:
        print("\n✅ Evaluation complete!")
        return 0
    else:
        print("\n❌ Evaluation failed!")
        return 1


if __name__ == "__main__":
    asyncio.run(main())

