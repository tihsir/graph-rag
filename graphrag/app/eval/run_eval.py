"""
Comprehensive evaluation runner comparing GraphRAG vs Vanilla RAG.

This module runs systematic evaluations across multiple dimensions:
- Information Retrieval quality (Recall, Precision, nDCG, MRR, etc.)
- Answer quality (ROUGE, BLEU, semantic similarity, LLM judge)
- Latency comparison
"""

import asyncio
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .datasets import QAPair, EvalDataset, create_pubmed_evaluation_dataset, update_gold_passages_from_chunks, get_chunk_texts_from_store
from .metrics import (
    compute_ir_metrics, 
    compute_answer_metrics, 
    aggregate_metrics,
    IRMetrics,
    AnswerMetrics
)
from .judge import llm_judge
from .vanilla_rag import vanilla_rag
from ..api.routers import query as graphrag_query
from ..api.schemas import QueryRequest

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SingleQueryResult:
    """Results for a single query evaluation."""
    question: str
    reference_answer: str
    difficulty: str
    reasoning_type: str
    
    # GraphRAG results
    graphrag_answer: str = ""
    graphrag_route: str = ""
    graphrag_retrieved_ids: List[str] = field(default_factory=list)
    graphrag_ir_metrics: Dict[str, float] = field(default_factory=dict)
    graphrag_answer_metrics: Dict[str, float] = field(default_factory=dict)
    graphrag_latencies: Dict[str, float] = field(default_factory=dict)
    
    # Vanilla RAG results
    vanilla_answer: str = ""
    vanilla_retrieved_ids: List[str] = field(default_factory=list)
    vanilla_ir_metrics: Dict[str, float] = field(default_factory=dict)
    vanilla_answer_metrics: Dict[str, float] = field(default_factory=dict)
    vanilla_latencies: Dict[str, float] = field(default_factory=dict)
    
    # Gold data
    gold_passages: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass  
class EvaluationReport:
    """Full evaluation report comparing GraphRAG and Vanilla RAG."""
    dataset_name: str
    num_queries: int
    timestamp: str
    
    # Aggregated metrics
    graphrag_ir_metrics: Dict[str, float] = field(default_factory=dict)
    graphrag_answer_metrics: Dict[str, float] = field(default_factory=dict)
    graphrag_avg_latency: Dict[str, float] = field(default_factory=dict)
    
    vanilla_ir_metrics: Dict[str, float] = field(default_factory=dict)
    vanilla_answer_metrics: Dict[str, float] = field(default_factory=dict)
    vanilla_avg_latency: Dict[str, float] = field(default_factory=dict)
    
    # Metric deltas (GraphRAG - Vanilla)
    ir_metric_deltas: Dict[str, float] = field(default_factory=dict)
    answer_metric_deltas: Dict[str, float] = field(default_factory=dict)
    
    # Per-query results
    query_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Breakdown by difficulty
    metrics_by_difficulty: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Breakdown by reasoning type  
    metrics_by_reasoning: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print(f"EVALUATION REPORT: {self.dataset_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Queries evaluated: {self.num_queries}")
        print("=" * 80)
        
        print("\n📊 INFORMATION RETRIEVAL METRICS")
        print("-" * 60)
        print(f"{'Metric':<20} {'GraphRAG':>12} {'Vanilla':>12} {'Delta':>12}")
        print("-" * 60)
        
        for metric in ["recall@5", "recall@10", "precision@5", "ndcg@10", "mrr", "hit_rate@5"]:
            g_val = self.graphrag_ir_metrics.get(metric, 0)
            v_val = self.vanilla_ir_metrics.get(metric, 0)
            delta = self.ir_metric_deltas.get(metric, 0)
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            print(f"{metric:<20} {g_val:>12.3f} {v_val:>12.3f} {delta_str:>12}")
        
        print("\n📝 ANSWER QUALITY METRICS")
        print("-" * 60)
        print(f"{'Metric':<20} {'GraphRAG':>12} {'Vanilla':>12} {'Delta':>12}")
        print("-" * 60)
        
        for metric in ["keyword_coverage", "semantic_similarity", "rouge_l", "f1_token", "answer_coverage", "llm_accuracy"]:
            g_val = self.graphrag_answer_metrics.get(metric, 0)
            v_val = self.vanilla_answer_metrics.get(metric, 0)
            delta = self.answer_metric_deltas.get(metric, 0)
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            print(f"{metric:<20} {g_val:>12.3f} {v_val:>12.3f} {delta_str:>12}")
        
        print("\n⏱️  LATENCY (seconds)")
        print("-" * 60)
        print(f"{'Stage':<20} {'GraphRAG':>12} {'Vanilla':>12}")
        print("-" * 60)
        
        for stage in ["retrieval", "generation", "total"]:
            g_val = self.graphrag_avg_latency.get(stage, 0)
            v_val = self.vanilla_avg_latency.get(stage, 0)
            print(f"{stage:<20} {g_val:>12.3f} {v_val:>12.3f}")
        
        print("\n" + "=" * 80)
        
        # Winner summary
        g_wins = sum(1 for d in self.ir_metric_deltas.values() if d > 0)
        g_wins += sum(1 for d in self.answer_metric_deltas.values() if d > 0)
        total_metrics = len(self.ir_metric_deltas) + len(self.answer_metric_deltas)
        
        if g_wins > total_metrics / 2:
            print("🏆 Overall Winner: GraphRAG")
        elif g_wins < total_metrics / 2:
            print("🏆 Overall Winner: Vanilla RAG")
        else:
            print("🤝 Result: Tie")
        
        print("=" * 80 + "\n")


async def run_single_query_eval(
    qa: QAPair,
    k: int = 10,
    rerank: bool = True,
    use_llm_judge: bool = True,
    embedding_provider=None
) -> SingleQueryResult:
    """
    Run evaluation for a single query on both systems.
    
    Args:
        qa: QA pair with question and ground truth
        k: Number of chunks to retrieve
        rerank: Whether to use reranking
        use_llm_judge: Whether to use LLM for answer quality judging
        embedding_provider: Embedding provider for semantic similarity
    
    Returns:
        SingleQueryResult with all metrics
    """
    result = SingleQueryResult(
        question=qa.question,
        reference_answer=qa.answer,
        difficulty=qa.difficulty,
        reasoning_type=qa.reasoning_type,
        gold_passages=qa.gold_passages,
        keywords=qa.keywords
    )
    
    gold_set = set(qa.gold_passages)
    
    # ========== GraphRAG Evaluation ==========
    try:
        request = QueryRequest(
            query=qa.question,
            mode="auto",
            k=k,
            rerank=rerank
        )
        graphrag_response = await graphrag_query(request)
        
        result.graphrag_answer = graphrag_response.answer
        result.graphrag_route = graphrag_response.route
        result.graphrag_retrieved_ids = [c.chunk_id for c in graphrag_response.citations]
        result.graphrag_latencies = graphrag_response.latencies
        
        # Compute IR metrics
        if gold_set:
            ir_metrics = compute_ir_metrics(result.graphrag_retrieved_ids, gold_set)
            result.graphrag_ir_metrics = ir_metrics.to_dict()
        
        # Compute answer metrics
        llm_scores = None
        if use_llm_judge:
            try:
                llm_scores = await llm_judge(result.graphrag_answer, qa.answer)
            except Exception as e:
                logger.warning("LLM judge failed for GraphRAG", error=str(e))
        
        answer_metrics = await compute_answer_metrics(
            result.graphrag_answer,
            qa.answer,
            qa.keywords,
            llm_scores=llm_scores,
            embedding_provider=embedding_provider
        )
        result.graphrag_answer_metrics = answer_metrics.to_dict()
        
    except Exception as e:
        logger.error("GraphRAG evaluation failed", question=qa.question[:50], error=str(e))
    
    # ========== Vanilla RAG Evaluation ==========
    try:
        vanilla_response = await vanilla_rag.query(
            query=qa.question,
            k=k,
            rerank=rerank
        )
        
        result.vanilla_answer = vanilla_response.answer
        result.vanilla_retrieved_ids = [c[0] for c in vanilla_response.retrieved_chunks]
        result.vanilla_latencies = vanilla_response.latencies
        
        # Compute IR metrics
        if gold_set:
            ir_metrics = compute_ir_metrics(result.vanilla_retrieved_ids, gold_set)
            result.vanilla_ir_metrics = ir_metrics.to_dict()
        
        # Compute answer metrics
        llm_scores = None
        if use_llm_judge:
            try:
                llm_scores = await llm_judge(result.vanilla_answer, qa.answer)
            except Exception as e:
                logger.warning("LLM judge failed for Vanilla RAG", error=str(e))
        
        answer_metrics = await compute_answer_metrics(
            result.vanilla_answer,
            qa.answer,
            qa.keywords,
            llm_scores=llm_scores,
            embedding_provider=embedding_provider
        )
        result.vanilla_answer_metrics = answer_metrics.to_dict()
        
    except Exception as e:
        logger.error("Vanilla RAG evaluation failed", question=qa.question[:50], error=str(e))
    
    return result


async def run_evaluation(
    dataset: Optional[EvalDataset] = None,
    modes: List[str] = ["graph", "rag"],  # Kept for backward compatibility
    metrics: List[str] = ["recall@k", "ndcg", "llm_judge"],  # Kept for backward compatibility
    k: int = 10,
    rerank: bool = True,
    use_llm_judge: bool = True,
    output_path: Optional[str] = None,
    update_gold_from_chunks: bool = True
) -> EvaluationReport:
    """
    Run comprehensive evaluation comparing GraphRAG vs Vanilla RAG.
    
    Args:
        dataset: Evaluation dataset (uses default PubMed dataset if None)
        modes: Deprecated, kept for backward compatibility
        metrics: Deprecated, kept for backward compatibility  
        k: Number of chunks to retrieve
        rerank: Whether to use reranking
        use_llm_judge: Whether to use LLM for answer quality
        output_path: Path to save report JSON
        update_gold_from_chunks: Whether to update gold passages from actual chunks
    
    Returns:
        EvaluationReport with all metrics and comparisons
    """
    # Load dataset
    if dataset is None:
        dataset = create_pubmed_evaluation_dataset()
    
    # Update gold passages from actual chunks if requested
    if update_gold_from_chunks:
        chunk_texts = get_chunk_texts_from_store()
        if chunk_texts:
            dataset = update_gold_passages_from_chunks(dataset, chunk_texts)
            logger.info("Updated gold passages from chunks", num_chunks=len(chunk_texts))
        else:
            logger.warning("No chunks found in store, using keyword-based gold passages")
    
    # Initialize embedding provider for semantic similarity
    embedding_provider = None
    try:
        from ..core.providers import get_embedding_provider
        embedding_provider = get_embedding_provider()
    except Exception as e:
        logger.warning("Could not load embedding provider for semantic similarity", error=str(e))
    
    # Run evaluations
    query_results: List[SingleQueryResult] = []
    
    for i, qa in enumerate(dataset.qa_pairs):
        logger.info(f"Evaluating query {i+1}/{len(dataset.qa_pairs)}", question=qa.question[:50])
        
        result = await run_single_query_eval(
            qa=qa,
            k=k,
            rerank=rerank,
            use_llm_judge=use_llm_judge,
            embedding_provider=embedding_provider
        )
        query_results.append(result)
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)
    
    # Aggregate metrics
    graphrag_ir = aggregate_metrics([r.graphrag_ir_metrics for r in query_results if r.graphrag_ir_metrics])
    graphrag_answer = aggregate_metrics([r.graphrag_answer_metrics for r in query_results if r.graphrag_answer_metrics])
    graphrag_latency = aggregate_metrics([r.graphrag_latencies for r in query_results if r.graphrag_latencies])
    
    vanilla_ir = aggregate_metrics([r.vanilla_ir_metrics for r in query_results if r.vanilla_ir_metrics])
    vanilla_answer = aggregate_metrics([r.vanilla_answer_metrics for r in query_results if r.vanilla_answer_metrics])
    vanilla_latency = aggregate_metrics([r.vanilla_latencies for r in query_results if r.vanilla_latencies])
    
    # Compute deltas
    ir_deltas = {}
    for key in graphrag_ir:
        if key in vanilla_ir:
            ir_deltas[key] = graphrag_ir[key] - vanilla_ir[key]
    
    answer_deltas = {}
    for key in graphrag_answer:
        if key in vanilla_answer:
            answer_deltas[key] = graphrag_answer[key] - vanilla_answer[key]
    
    # Breakdown by difficulty
    metrics_by_difficulty = {}
    for difficulty in ["easy", "medium", "hard"]:
        diff_results = [r for r in query_results if r.difficulty == difficulty]
        if diff_results:
            metrics_by_difficulty[difficulty] = {
                "count": len(diff_results),
                "graphrag_ir": aggregate_metrics([r.graphrag_ir_metrics for r in diff_results if r.graphrag_ir_metrics]),
                "vanilla_ir": aggregate_metrics([r.vanilla_ir_metrics for r in diff_results if r.vanilla_ir_metrics]),
                "graphrag_answer": aggregate_metrics([r.graphrag_answer_metrics for r in diff_results if r.graphrag_answer_metrics]),
                "vanilla_answer": aggregate_metrics([r.vanilla_answer_metrics for r in diff_results if r.vanilla_answer_metrics]),
            }
    
    # Breakdown by reasoning type
    metrics_by_reasoning = {}
    for rtype in ["factual", "multi-hop", "comparative", "temporal"]:
        type_results = [r for r in query_results if r.reasoning_type == rtype]
        if type_results:
            metrics_by_reasoning[rtype] = {
                "count": len(type_results),
                "graphrag_ir": aggregate_metrics([r.graphrag_ir_metrics for r in type_results if r.graphrag_ir_metrics]),
                "vanilla_ir": aggregate_metrics([r.vanilla_ir_metrics for r in type_results if r.vanilla_ir_metrics]),
                "graphrag_answer": aggregate_metrics([r.graphrag_answer_metrics for r in type_results if r.graphrag_answer_metrics]),
                "vanilla_answer": aggregate_metrics([r.vanilla_answer_metrics for r in type_results if r.vanilla_answer_metrics]),
            }
    
    # Build report
    report = EvaluationReport(
        dataset_name=dataset.name,
        num_queries=len(query_results),
        timestamp=datetime.now().isoformat(),
        graphrag_ir_metrics=graphrag_ir,
        graphrag_answer_metrics=graphrag_answer,
        graphrag_avg_latency=graphrag_latency,
        vanilla_ir_metrics=vanilla_ir,
        vanilla_answer_metrics=vanilla_answer,
        vanilla_avg_latency=vanilla_latency,
        ir_metric_deltas=ir_deltas,
        answer_metric_deltas=answer_deltas,
        query_results=[r.to_dict() for r in query_results],
        metrics_by_difficulty=metrics_by_difficulty,
        metrics_by_reasoning=metrics_by_reasoning,
    )
    
    # Save if path provided
    if output_path:
        report.save(output_path)
        logger.info("Saved evaluation report", path=output_path)
    
    return report


async def run_quick_eval(num_queries: int = 3) -> EvaluationReport:
    """
    Run a quick evaluation with a subset of queries.
    Useful for testing the evaluation pipeline.
    """
    dataset = create_pubmed_evaluation_dataset()
    dataset.qa_pairs = dataset.qa_pairs[:num_queries]
    
    return await run_evaluation(
        dataset=dataset,
        use_llm_judge=False,  # Skip LLM judge for speed
        update_gold_from_chunks=True
    )


# Backward compatibility
async def run_evaluation_legacy(
    dataset: List[QAPair],
    modes: List[str] = ["graph", "rag"],
    metrics: List[str] = ["recall@k", "ndcg", "llm_judge"],
) -> Dict[str, Dict[str, float]]:
    """
    Legacy evaluation function for backward compatibility.
    Returns simplified dict format.
    """
    eval_dataset = EvalDataset(
        name="legacy",
        description="Legacy evaluation",
        qa_pairs=dataset
    )
    
    report = await run_evaluation(
        dataset=eval_dataset,
        use_llm_judge="llm_judge" in metrics
    )
    
    return {
        "graph": {**report.graphrag_ir_metrics, **report.graphrag_answer_metrics},
        "rag": {**report.vanilla_ir_metrics, **report.vanilla_answer_metrics},
    }
