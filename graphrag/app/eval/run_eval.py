"""Evaluation runner comparing GraphRAG vs vanilla RAG."""

from typing import List, Dict
from .datasets import QAPair
from .metrics import recall_at_k, ndcg_at_k
from .judge import llm_judge
from ..api.routers import query
from ..api.schemas import QueryRequest

from ..core.logging import get_logger

logger = get_logger(__name__)


async def run_evaluation(
    dataset: List[QAPair],
    modes: List[str] = ["graph", "rag"],
    metrics: List[str] = ["recall@k", "ndcg", "llm_judge"],
) -> Dict[str, Dict[str, float]]:
    """Run evaluation comparing different modes."""
    results = {mode: {} for mode in modes}
    
    for mode in modes:
        recall_scores = []
        ndcg_scores = []
        judge_scores = {"accuracy": [], "completeness": [], "relevance": []}
        
        for qa in dataset:
            # Query system
            request = QueryRequest(query=qa.question, mode=mode, k=10, rerank=True)
            response = await query(request)
            
            # IR metrics
            retrieved_chunk_ids = [c.chunk_id for c in response.citations]
            gold_passages = set(qa.gold_passages)
            
            if "recall@k" in metrics:
                recall_10 = recall_at_k(retrieved_chunk_ids, gold_passages, k=10)
                recall_scores.append(recall_10)
            
            if "ndcg" in metrics:
                ndcg_10 = ndcg_at_k(retrieved_chunk_ids, gold_passages, k=10)
                ndcg_scores.append(ndcg_10)
            
            # LLM judge
            if "llm_judge" in metrics:
                scores = await llm_judge(response.answer, qa.answer)
                for key in judge_scores:
                    judge_scores[key].append(scores[key])
        
        # Aggregate results
        if recall_scores:
            results[mode]["recall@10"] = sum(recall_scores) / len(recall_scores)
        if ndcg_scores:
            results[mode]["ndcg@10"] = sum(ndcg_scores) / len(ndcg_scores)
        if judge_scores["accuracy"]:
            results[mode]["judge_accuracy"] = sum(judge_scores["accuracy"]) / len(judge_scores["accuracy"])
            results[mode]["judge_completeness"] = sum(judge_scores["completeness"]) / len(judge_scores["completeness"])
            results[mode]["judge_relevance"] = sum(judge_scores["relevance"]) / len(judge_scores["relevance"])
    
    return results

