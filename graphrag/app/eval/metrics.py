"""Comprehensive IR and answer quality metrics for evaluation."""

from typing import List, Set, Dict, Tuple, Optional
import math
import re
from dataclasses import dataclass
from collections import Counter


# ============================================================================
# Information Retrieval Metrics
# ============================================================================

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Recall@k: fraction of relevant items retrieved in top-k.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
        k: Number of top results to consider
    
    Returns:
        Recall score between 0 and 1
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Precision@k: fraction of top-k that are relevant.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
        k: Number of top results to consider
    
    Returns:
        Precision score between 0 and 1
    """
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute F1@k: harmonic mean of precision and recall at k.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
        k: Number of top results to consider
    
    Returns:
        F1 score between 0 and 1
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute nDCG@k (Normalized Discounted Cumulative Gain) with binary relevance.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
        k: Number of top results to consider
    
    Returns:
        nDCG score between 0 and 1
    """
    if not relevant:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            relevance = 1.0
            rank = i + 1
            dcg += relevance / math.log2(rank + 1)
    
    # Calculate ideal DCG (all relevant items at top)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Compute Mean Reciprocal Rank: 1/rank of first relevant item.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
    
    Returns:
        MRR score between 0 and 1
    """
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Compute Average Precision (AP) for a single query.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
    
    Returns:
        AP score between 0 and 1
    """
    if not relevant:
        return 0.0
    
    num_relevant_seen = 0
    precision_sum = 0.0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant)


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Hit Rate@k: 1 if any relevant item in top-k, else 0.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
        k: Number of top results to consider
    
    Returns:
        1.0 if hit, 0.0 if miss
    """
    retrieved_k = set(retrieved[:k])
    return 1.0 if len(retrieved_k & relevant) > 0 else 0.0


# ============================================================================
# Answer Quality Metrics
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, remove punctuation, normalize whitespace.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def compute_exact_match(answer: str, reference: str) -> float:
    """
    Compute Exact Match (EM): 1 if normalized answer equals normalized reference, else 0.
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if normalize_text(answer) == normalize_text(reference) else 0.0


def compute_f1_token(answer: str, reference: str) -> float:
    """
    Compute token-level F1 score between answer and reference.
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
    
    Returns:
        F1 score between 0 and 1
    """
    answer_tokens = normalize_text(answer).split()
    reference_tokens = normalize_text(reference).split()
    
    if not answer_tokens or not reference_tokens:
        return 0.0
    
    answer_counter = Counter(answer_tokens)
    reference_counter = Counter(reference_tokens)
    
    # Count common tokens
    common = answer_counter & reference_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(answer_tokens)
    recall = num_common / len(reference_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_context_relevance(
    answer: str, 
    context_chunks: List[str],
    keywords: List[str] = None
) -> float:
    """
    Compute how relevant the retrieved context is to the answer.
    Measures overlap between answer content and retrieved chunks.
    
    Args:
        answer: Generated answer text
        context_chunks: List of retrieved chunk texts
        keywords: Optional keywords to check for coverage
    
    Returns:
        Relevance score between 0 and 1
    """
    if not context_chunks:
        return 0.0
    
    answer_lower = answer.lower()
    combined_context = ' '.join(context_chunks).lower()
    
    # Method 1: Token overlap
    answer_tokens = set(normalize_text(answer).split())
    context_tokens = set(normalize_text(combined_context).split())
    
    if not answer_tokens:
        return 0.0
    
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    
    # Method 2: Keyword presence (if keywords provided)
    keyword_score = 1.0
    if keywords:
        keywords_in_context = sum(1 for kw in keywords if kw.lower() in combined_context)
        keyword_score = keywords_in_context / len(keywords) if keywords else 1.0
    
    # Combine scores
    return 0.7 * overlap + 0.3 * keyword_score


def compute_answer_coverage(answer: str, reference: str) -> float:
    """
    Compute how much of the reference answer is covered by the generated answer.
    Higher score means the answer captures more of the expected content.
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
    
    Returns:
        Coverage score between 0 and 1
    """
    answer_tokens = set(normalize_text(answer).split())
    reference_tokens = set(normalize_text(reference).split())
    
    if not reference_tokens:
        return 1.0 if not answer_tokens else 0.0
    
    covered = len(answer_tokens & reference_tokens)
    return covered / len(reference_tokens)


def compute_keyword_coverage(answer: str, keywords: List[str]) -> float:
    """
    Compute fraction of expected keywords present in the answer.
    
    Args:
        answer: Generated answer text
        keywords: List of expected keywords
    
    Returns:
        Coverage score between 0 and 1
    """
    if not keywords:
        return 1.0
    
    answer_lower = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matches / len(keywords)


def compute_answer_length_ratio(answer: str, reference: str) -> float:
    """
    Compute ratio of answer length to reference length.
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
    
    Returns:
        Length ratio (1.0 means same length)
    """
    if not reference:
        return 0.0
    return len(answer.split()) / len(reference.split())


async def compute_semantic_similarity(
    answer: str, 
    reference: str,
    embedding_provider=None
) -> float:
    """
    Compute semantic similarity between answer and reference using embeddings.
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
        embedding_provider: Embedding provider instance
    
    Returns:
        Cosine similarity between 0 and 1
    """
    if embedding_provider is None:
        try:
            from ..core.providers import get_embedding_provider
            embedding_provider = get_embedding_provider()
        except Exception:
            return 0.0
    
    try:
        embeddings = await embedding_provider.embed([answer, reference])
        
        # Compute cosine similarity
        import numpy as np
        a = np.array(embeddings[0])
        b = np.array(embeddings[1])
        
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(similarity)
    except Exception:
        return 0.0


def compute_rouge_l(answer: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score (longest common subsequence based).
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
    
    Returns:
        ROUGE-L F1 score between 0 and 1
    """
    def lcs_length(s1: List[str], s2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    answer_tokens = answer.lower().split()
    reference_tokens = reference.lower().split()
    
    if not answer_tokens or not reference_tokens:
        return 0.0
    
    lcs_len = lcs_length(answer_tokens, reference_tokens)
    
    precision = lcs_len / len(answer_tokens)
    recall = lcs_len / len(reference_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_bleu_1(answer: str, reference: str) -> float:
    """
    Compute BLEU-1 score (unigram precision with brevity penalty).
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
    
    Returns:
        BLEU-1 score between 0 and 1
    """
    answer_tokens = answer.lower().split()
    reference_tokens = reference.lower().split()
    
    if not answer_tokens:
        return 0.0
    
    # Count matching unigrams
    reference_counts: Dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    
    matches = 0
    answer_counts: Dict[str, int] = {}
    for token in answer_tokens:
        answer_counts[token] = answer_counts.get(token, 0) + 1
        if answer_counts[token] <= reference_counts.get(token, 0):
            matches += 1
    
    precision = matches / len(answer_tokens)
    
    # Brevity penalty
    if len(answer_tokens) >= len(reference_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(reference_tokens) / len(answer_tokens))
    
    return bp * precision


# ============================================================================
# Aggregated Metrics
# ============================================================================

@dataclass
class IRMetrics:
    """Container for IR metrics results."""
    recall_1: float
    recall_5: float
    recall_10: float
    precision_1: float
    precision_5: float
    precision_10: float
    f1_5: float
    f1_10: float
    ndcg_5: float
    ndcg_10: float
    mrr: float
    map: float  # Mean Average Precision
    hit_rate_1: float
    hit_rate_5: float
    hit_rate_10: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "recall@1": self.recall_1,
            "recall@5": self.recall_5,
            "recall@10": self.recall_10,
            "precision@1": self.precision_1,
            "precision@5": self.precision_5,
            "precision@10": self.precision_10,
            "f1@5": self.f1_5,
            "f1@10": self.f1_10,
            "ndcg@5": self.ndcg_5,
            "ndcg@10": self.ndcg_10,
            "mrr": self.mrr,
            "map": self.map,
            "hit_rate@1": self.hit_rate_1,
            "hit_rate@5": self.hit_rate_5,
            "hit_rate@10": self.hit_rate_10,
        }


@dataclass
class AnswerMetrics:
    """Container for answer quality metrics."""
    keyword_coverage: float
    length_ratio: float
    semantic_similarity: float
    rouge_l: float
    bleu_1: float
    exact_match: float
    f1_token: float
    answer_coverage: float
    llm_accuracy: float
    llm_completeness: float
    llm_relevance: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "keyword_coverage": self.keyword_coverage,
            "length_ratio": self.length_ratio,
            "semantic_similarity": self.semantic_similarity,
            "rouge_l": self.rouge_l,
            "bleu_1": self.bleu_1,
            "exact_match": self.exact_match,
            "f1_token": self.f1_token,
            "answer_coverage": self.answer_coverage,
            "llm_accuracy": self.llm_accuracy,
            "llm_completeness": self.llm_completeness,
            "llm_relevance": self.llm_relevance,
        }


def compute_ir_metrics(retrieved: List[str], relevant: Set[str]) -> IRMetrics:
    """
    Compute all IR metrics for a single query.
    
    Args:
        retrieved: Ordered list of retrieved chunk IDs
        relevant: Set of relevant (gold) chunk IDs
    
    Returns:
        IRMetrics dataclass with all computed metrics
    """
    return IRMetrics(
        recall_1=recall_at_k(retrieved, relevant, 1),
        recall_5=recall_at_k(retrieved, relevant, 5),
        recall_10=recall_at_k(retrieved, relevant, 10),
        precision_1=precision_at_k(retrieved, relevant, 1),
        precision_5=precision_at_k(retrieved, relevant, 5),
        precision_10=precision_at_k(retrieved, relevant, 10),
        f1_5=f1_at_k(retrieved, relevant, 5),
        f1_10=f1_at_k(retrieved, relevant, 10),
        ndcg_5=ndcg_at_k(retrieved, relevant, 5),
        ndcg_10=ndcg_at_k(retrieved, relevant, 10),
        mrr=mrr(retrieved, relevant),
        map=average_precision(retrieved, relevant),
        hit_rate_1=hit_rate_at_k(retrieved, relevant, 1),
        hit_rate_5=hit_rate_at_k(retrieved, relevant, 5),
        hit_rate_10=hit_rate_at_k(retrieved, relevant, 10),
    )


async def compute_answer_metrics(
    answer: str, 
    reference: str, 
    keywords: List[str],
    llm_scores: Optional[Dict[str, float]] = None,
    embedding_provider=None,
    context_chunks: Optional[List[str]] = None
) -> AnswerMetrics:
    """
    Compute all answer quality metrics.
    
    Args:
        answer: Generated answer text
        reference: Reference answer text
        keywords: Expected keywords in answer
        llm_scores: Pre-computed LLM judge scores (optional)
        embedding_provider: Embedding provider for semantic similarity
        context_chunks: Retrieved context chunks (for context relevance)
    
    Returns:
        AnswerMetrics dataclass with all computed metrics
    """
    semantic_sim = await compute_semantic_similarity(answer, reference, embedding_provider)
    
    llm_acc = llm_scores.get("accuracy", 0.0) if llm_scores else 0.0
    llm_comp = llm_scores.get("completeness", 0.0) if llm_scores else 0.0
    llm_rel = llm_scores.get("relevance", 0.0) if llm_scores else 0.0
    
    return AnswerMetrics(
        keyword_coverage=compute_keyword_coverage(answer, keywords),
        length_ratio=compute_answer_length_ratio(answer, reference),
        semantic_similarity=semantic_sim,
        rouge_l=compute_rouge_l(answer, reference),
        bleu_1=compute_bleu_1(answer, reference),
        exact_match=compute_exact_match(answer, reference),
        f1_token=compute_f1_token(answer, reference),
        answer_coverage=compute_answer_coverage(answer, reference),
        llm_accuracy=llm_acc,
        llm_completeness=llm_comp,
        llm_relevance=llm_rel,
    )


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries by computing mean.
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Dictionary with mean values for each metric
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[key] = sum(values) / len(values)
    
    return aggregated
