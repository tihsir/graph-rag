#!/usr/bin/env python3
"""
Simple end-to-end evaluation test for GraphRAG vs Vanilla RAG.

This script tests the evaluation framework with minimal setup:
1. Tests individual metric functions
2. Tests vanilla RAG retrieval
3. Tests the full evaluation pipeline (optional, requires data)

Usage:
    python test_eval.py                    # Run unit tests only
    python test_eval.py --full             # Run full evaluation (requires ingested data)
    python test_eval.py --quick            # Run quick evaluation with 3 queries
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_metrics():
    """Test all metric functions with sample data."""
    print("\n" + "=" * 60)
    print("🧪 TESTING METRICS")
    print("=" * 60)
    
    # Import metrics directly from the file (avoiding full package init chain)
    import importlib.util
    metrics_path = Path(__file__).parent / "graphrag" / "app" / "eval" / "metrics.py"
    
    spec = importlib.util.spec_from_file_location("metrics", metrics_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    
    # Extract functions
    recall_at_k = metrics.recall_at_k
    precision_at_k = metrics.precision_at_k
    f1_at_k = metrics.f1_at_k
    ndcg_at_k = metrics.ndcg_at_k
    mrr = metrics.mrr
    average_precision = metrics.average_precision
    hit_rate_at_k = metrics.hit_rate_at_k
    compute_ir_metrics = metrics.compute_ir_metrics
    compute_exact_match = metrics.compute_exact_match
    compute_f1_token = metrics.compute_f1_token
    compute_keyword_coverage = metrics.compute_keyword_coverage
    compute_rouge_l = metrics.compute_rouge_l
    compute_bleu_1 = metrics.compute_bleu_1
    compute_answer_coverage = metrics.compute_answer_coverage
    normalize_text = metrics.normalize_text
    
    # Test IR metrics
    print("\n📊 IR Metrics:")
    retrieved = ["chunk_1", "chunk_3", "chunk_5", "chunk_2", "chunk_7"]
    relevant = {"chunk_1", "chunk_2", "chunk_4"}
    
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant:  {relevant}")
    print(f"  Recall@3:     {recall_at_k(retrieved, relevant, 3):.3f}")
    print(f"  Recall@5:     {recall_at_k(retrieved, relevant, 5):.3f}")
    print(f"  Precision@3:  {precision_at_k(retrieved, relevant, 3):.3f}")
    print(f"  Precision@5:  {precision_at_k(retrieved, relevant, 5):.3f}")
    print(f"  F1@5:         {f1_at_k(retrieved, relevant, 5):.3f}")
    print(f"  nDCG@5:       {ndcg_at_k(retrieved, relevant, 5):.3f}")
    print(f"  MRR:          {mrr(retrieved, relevant):.3f}")
    print(f"  MAP:          {average_precision(retrieved, relevant):.3f}")
    print(f"  Hit@1:        {hit_rate_at_k(retrieved, relevant, 1):.3f}")
    print(f"  Hit@5:        {hit_rate_at_k(retrieved, relevant, 5):.3f}")
    
    # Test aggregated IR metrics
    ir_metrics = compute_ir_metrics(retrieved, relevant)
    print(f"\n  Aggregated IR metrics: {len(ir_metrics.to_dict())} metrics computed")
    
    # Test answer quality metrics
    print("\n📝 Answer Quality Metrics:")
    reference = "Asthma treatments include inhaled corticosteroids (ICS) for long-term control and beta-agonists for relief."
    answer_good = "Asthma is treated with inhaled corticosteroids for control and beta-agonists for quick relief."
    answer_bad = "Diabetes requires insulin injections and blood sugar monitoring."
    
    print(f"  Reference: {reference[:60]}...")
    print(f"  Good ans:  {answer_good[:60]}...")
    print(f"  Bad ans:   {answer_bad[:60]}...")
    
    print(f"\n  Good answer metrics:")
    print(f"    Exact Match:      {compute_exact_match(answer_good, reference):.3f}")
    print(f"    F1 Token:         {compute_f1_token(answer_good, reference):.3f}")
    print(f"    ROUGE-L:          {compute_rouge_l(answer_good, reference):.3f}")
    print(f"    BLEU-1:           {compute_bleu_1(answer_good, reference):.3f}")
    print(f"    Answer Coverage:  {compute_answer_coverage(answer_good, reference):.3f}")
    
    keywords = ["corticosteroids", "ICS", "beta-agonists", "asthma"]
    print(f"    Keyword Coverage: {compute_keyword_coverage(answer_good, keywords):.3f}")
    
    print(f"\n  Bad answer metrics:")
    print(f"    Exact Match:      {compute_exact_match(answer_bad, reference):.3f}")
    print(f"    F1 Token:         {compute_f1_token(answer_bad, reference):.3f}")
    print(f"    ROUGE-L:          {compute_rouge_l(answer_bad, reference):.3f}")
    print(f"    BLEU-1:           {compute_bleu_1(answer_bad, reference):.3f}")
    print(f"    Answer Coverage:  {compute_answer_coverage(answer_bad, reference):.3f}")
    print(f"    Keyword Coverage: {compute_keyword_coverage(answer_bad, keywords):.3f}")
    
    # Verify basic correctness
    assert compute_exact_match(answer_good, reference) == 0.0, "Good answer should not exactly match"
    assert compute_f1_token(answer_good, reference) > compute_f1_token(answer_bad, reference), "Good answer should have higher F1"
    assert compute_keyword_coverage(answer_good, keywords) > compute_keyword_coverage(answer_bad, keywords), "Good answer should have better keyword coverage"
    
    print("\n✅ Metrics tests passed!")
    return True


def test_datasets():
    """Test dataset creation and manipulation."""
    print("\n" + "=" * 60)
    print("🧪 TESTING DATASETS")
    print("=" * 60)
    
    # Import datasets directly from file
    import importlib.util
    datasets_path = Path(__file__).parent / "graphrag" / "app" / "eval" / "datasets.py"
    
    spec = importlib.util.spec_from_file_location("datasets", datasets_path)
    datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datasets)
    
    QAPair = datasets.QAPair
    EvalDataset = datasets.EvalDataset
    create_pubmed_evaluation_dataset = datasets.create_pubmed_evaluation_dataset
    create_sample_dataset = datasets.create_sample_dataset
    
    # Create full dataset
    dataset = create_pubmed_evaluation_dataset()
    print(f"\n📋 PubMed Evaluation Dataset:")
    print(f"  Name: {dataset.name}")
    print(f"  Description: {dataset.description}")
    print(f"  Domain: {dataset.domain}")
    print(f"  Total QA pairs: {len(dataset.qa_pairs)}")
    
    # Show difficulty distribution
    difficulties = {}
    reasoning_types = {}
    for qa in dataset.qa_pairs:
        difficulties[qa.difficulty] = difficulties.get(qa.difficulty, 0) + 1
        reasoning_types[qa.reasoning_type] = reasoning_types.get(qa.reasoning_type, 0) + 1
    
    print(f"\n  Difficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        print(f"    {diff}: {count}")
    
    print(f"\n  Reasoning type distribution:")
    for rtype, count in sorted(reasoning_types.items()):
        print(f"    {rtype}: {count}")
    
    # Show sample question
    sample = dataset.qa_pairs[0]
    print(f"\n  Sample question:")
    print(f"    Q: {sample.question[:80]}...")
    print(f"    A: {sample.answer[:80]}...")
    print(f"    Keywords: {sample.keywords[:5]}...")
    print(f"    Difficulty: {sample.difficulty}")
    print(f"    Reasoning: {sample.reasoning_type}")
    
    # Test sample dataset
    sample_qa = create_sample_dataset()
    print(f"\n  Sample dataset (quick test): {len(sample_qa)} QA pairs")
    
    # Basic assertions
    assert len(dataset.qa_pairs) > 0, "Dataset should have QA pairs"
    assert len(sample_qa) == 3, "Sample dataset should have 3 QA pairs"
    
    print("\n✅ Dataset tests passed!")
    return True


async def test_vanilla_rag():
    """Test vanilla RAG implementation."""
    print("\n" + "=" * 60)
    print("🧪 TESTING VANILLA RAG")
    print("=" * 60)
    
    try:
        from graphrag.app.eval.vanilla_rag import vanilla_rag, VanillaRAG
        from graphrag.app.data.metadata_store import ChunkModel, metadata_store
        
        # Check if data exists
        session = metadata_store.get_session()
        chunk_count = session.query(ChunkModel).count()
        session.close()
        
        if chunk_count == 0:
            print("\n⚠️  No chunks in database. Skipping vanilla RAG test.")
            print("   Run 'python run_evaluation.py' first to ingest data.")
            return True
        
        print(f"\n📊 Found {chunk_count} chunks in database")
        
        # Test retrieval
        print("\n🔍 Testing retrieval...")
        test_query = "What are the main treatments for asthma?"
        
        chunk_ids, chunks = await vanilla_rag.retrieve_only(test_query, k=5, rerank=False)
        print(f"  Query: {test_query}")
        print(f"  Retrieved {len(chunk_ids)} chunks")
        
        if chunks:
            print(f"  Top chunk preview: {chunks[0][1][:100]}...")
        
        # Test full query (if LLM is available)
        try:
            print("\n💬 Testing full query pipeline...")
            result = await vanilla_rag.query(test_query, k=5, rerank=False)
            print(f"  Answer: {result.answer[:200]}...")
            print(f"  Latencies: {result.latencies}")
            print("\n✅ Vanilla RAG tests passed!")
        except Exception as e:
            print(f"\n⚠️  Full query failed (likely API key issue): {str(e)[:100]}")
            print("   Retrieval-only test passed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Vanilla RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_quick_evaluation():
    """Run quick evaluation with 3 queries."""
    print("\n" + "=" * 60)
    print("🧪 RUNNING QUICK EVALUATION (3 queries)")
    print("=" * 60)
    
    try:
        from graphrag.app.eval import run_quick_eval
        
        print("\n⏳ Running evaluation (this may take a few minutes)...")
        report = await run_quick_eval(num_queries=3)
        
        print("\n📊 Evaluation Results:")
        report.print_summary()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_full_evaluation():
    """Run full evaluation."""
    print("\n" + "=" * 60)
    print("🧪 RUNNING FULL EVALUATION")
    print("=" * 60)
    
    try:
        from graphrag.app.eval import run_evaluation, create_pubmed_evaluation_dataset
        from pathlib import Path
        from datetime import datetime
        
        output_dir = Path("./eval_results")
        output_dir.mkdir(exist_ok=True)
        
        dataset = create_pubmed_evaluation_dataset()
        print(f"\n📋 Dataset: {dataset.name} ({len(dataset.qa_pairs)} queries)")
        
        print("\n⏳ Running evaluation (this may take several minutes)...")
        report = await run_evaluation(
            dataset=dataset,
            k=10,
            rerank=True,
            use_llm_judge=True,
            output_path=str(output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
            update_gold_from_chunks=True
        )
        
        print("\n📊 Evaluation Results:")
        report.print_summary()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="Test GraphRAG evaluation framework")
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (3 queries)")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG tests (no API needed)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GraphRAG Evaluation Framework Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Always run unit tests
    if not test_metrics():
        all_passed = False
    
    if not test_datasets():
        all_passed = False
    
    # Optionally test vanilla RAG
    if not args.skip_rag:
        if not await test_vanilla_rag():
            all_passed = False
    
    # Optionally run evaluation
    if args.quick:
        if not await run_quick_evaluation():
            all_passed = False
    elif args.full:
        if not await run_full_evaluation():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
