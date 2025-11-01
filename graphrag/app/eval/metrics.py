"""IR metrics for evaluation: Recall@k, nDCG."""

from typing import List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute nDCG@k (simplified binary relevance)."""
    import math
    
    if not relevant:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance = 1.0
            rank = i + 1
            dcg += relevance / math.log2(rank + 1)
    
    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    
    return dcg / idcg if idcg > 0 else 0.0

