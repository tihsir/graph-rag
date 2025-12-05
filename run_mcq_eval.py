#!/usr/bin/env python3
"""
Run MCQ Evaluation comparing GraphRAG, Vanilla RAG, and Direct LLM.

This script:
1. Loads biomedical MCQ dataset
2. Runs each question through different RAG methods
3. Grades answers using LLM-as-a-judge
4. Generates comparison report

Usage:
    python run_mcq_eval.py                    # Full evaluation
    python run_mcq_eval.py --quick            # Quick test (5 questions)
    python run_mcq_eval.py --num-questions 10 # Limit to 10 questions
    python run_mcq_eval.py --no-graphrag      # Skip GraphRAG (faster)
    python run_mcq_eval.py --no-llm-judge     # Simple grading (faster)
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime


async def main():
    parser = argparse.ArgumentParser(description="Run MCQ evaluation comparing RAG methods")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 questions")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--no-graphrag", action="store_true", help="Skip GraphRAG evaluation")
    parser.add_argument("--no-llm-judge", action="store_true", help="Use simple grading instead of LLM judge")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory for reports")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MCQ Evaluation: GraphRAG vs Vanilla RAG vs Direct LLM")
    print("=" * 70)
    
    # Import here to avoid loading models at startup
    from graphrag.app.eval.mcq_eval_runner import (
        run_mcq_evaluation,
        run_quick_mcq_eval,
        create_biomedical_mcq_dataset
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine methods to run
    methods = ["vanilla_rag", "direct_llm"]
    if not args.no_graphrag:
        methods.insert(0, "graphrag")
    
    print(f"\n📋 Configuration:")
    print(f"   Methods: {', '.join(methods)}")
    print(f"   LLM Judge: {'No' if args.no_llm_judge else 'Yes'}")
    print(f"   Retrieval k: {args.k}")
    
    if args.quick:
        print(f"\n⚡ Quick mode: 5 questions")
        report = await run_quick_mcq_eval(
            num_questions=5,
            methods=methods,
            use_llm_judge=not args.no_llm_judge
        )
    else:
        dataset = create_biomedical_mcq_dataset()
        num_q = args.num_questions or len(dataset.questions)
        print(f"\n📊 Dataset: {dataset.name}")
        print(f"   Total questions: {len(dataset.questions)}")
        print(f"   Evaluating: {num_q} questions")
        
        output_path = str(output_dir / f"mcq_eval_{timestamp}.json")
        
        report = await run_mcq_evaluation(
            dataset=dataset,
            methods=methods,
            k=args.k,
            use_llm_judge=not args.no_llm_judge,
            output_path=output_path,
            num_questions=args.num_questions
        )
    
    # Print summary
    report.print_summary()
    
    # Save report
    report_path = output_dir / f"mcq_eval_{timestamp}.json"
    report.save(str(report_path))
    print(f"\n💾 Full report saved to: {report_path}")
    
    print("\n✅ MCQ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())

