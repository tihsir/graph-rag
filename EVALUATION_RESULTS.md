# GraphRAG vs Vanilla RAG Evaluation Results

## Overview

This document summarizes the evaluation results comparing **GraphRAG** (graph-augmented retrieval) against **Vanilla RAG** (pure vector search) on medical QA tasks using PubMed articles.

## Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Dataset | PubMed Medical QA (asthma & diabetes articles) |
| Documents | 2 |
| Chunks | 39 |
| Graph Entities | 258 |
| Graph Relations | 77 |
| Queries Evaluated | 3 (quick mode) |
| Retrieval k | 10 |
| Reranking | Enabled (BGE-reranker-base) |
| LLM | GPT-3.5-turbo |
| Embeddings | all-mpnet-base-v2 (local) |

## Results Summary

### 📊 Information Retrieval Metrics

| Metric | GraphRAG | Vanilla RAG | Delta |
|--------|----------|-------------|-------|
| Recall@5 | 0.000 | 0.833 | -0.833 |
| Recall@10 | 0.000 | 1.000 | -1.000 |
| Precision@5 | 0.000 | 0.200 | -0.200 |
| nDCG@10 | 0.000 | 0.939 | -0.939 |
| MRR | 0.000 | 1.000 | -1.000 |
| Hit Rate@5 | 0.000 | 1.000 | -1.000 |

### 📝 Answer Quality Metrics

| Metric | GraphRAG | Vanilla RAG | Delta |
|--------|----------|-------------|-------|
| Keyword Coverage | 0.319 | 0.556 | -0.236 |
| Semantic Similarity | 0.830 | 0.925 | -0.095 |
| ROUGE-L | 0.145 | 0.369 | -0.224 |
| F1 Token | 0.238 | 0.556 | -0.318 |
| Answer Coverage | 0.370 | 0.590 | -0.220 |

### ⏱️ Latency (seconds)

| Stage | GraphRAG | Vanilla RAG |
|-------|----------|-------------|
| Retrieval | 68.362 | 12.170 |
| Generation | 1.705 | 1.366 |
| **Total** | **71.046** | **13.536** |

## 🏆 Winner: Vanilla RAG

In this evaluation, Vanilla RAG outperformed GraphRAG across all metrics.

## Analysis

### Why Vanilla RAG Won

1. **Small Dataset**: With only 39 chunks, the graph structure doesn't provide significant advantages over pure vector search.

2. **Graph Traversal Issues**: GraphRAG returned 0 results from graph traversal (`graph_count: 0`), falling back to vector search but with additional overhead.

3. **Latency Overhead**: Graph entity extraction and traversal added ~55 seconds of overhead per query.

4. **Dense Retrieval Effectiveness**: For factual medical QA, semantic similarity search is highly effective.

### When GraphRAG Would Excel

GraphRAG is expected to outperform Vanilla RAG when:
- **Multi-hop reasoning** is required (e.g., "What drugs treat conditions that affect the same organ as asthma?")
- **Large corpora** where entity relationships help filter noise
- **Complex queries** requiring relationship traversal
- **Entity-centric questions** (e.g., "What are all treatments mentioned for Dr. Smith's patients?")

## Metrics Explained

### IR Metrics
- **Recall@k**: Fraction of relevant chunks retrieved in top-k
- **Precision@k**: Fraction of top-k that are relevant
- **nDCG@k**: Normalized Discounted Cumulative Gain (rewards relevant items at top)
- **MRR**: Mean Reciprocal Rank (1/rank of first relevant item)
- **Hit Rate@k**: 1 if any relevant item in top-k

### Answer Quality Metrics
- **Keyword Coverage**: Fraction of expected keywords in answer
- **Semantic Similarity**: Cosine similarity of answer/reference embeddings
- **ROUGE-L**: Longest common subsequence F1
- **F1 Token**: Token-level F1 score
- **Answer Coverage**: Fraction of reference tokens covered

## Reproducing Results

```bash
# Quick evaluation (3 queries, no LLM judge)
python run_evaluation.py --quick --no-llm-judge

# Full evaluation (all 13 queries)
python run_evaluation.py

# Custom evaluation
python run_evaluation.py --num-queries 5 --k 10 --no-rerank
```

## Files

- `eval_results/eval_report_*.json` - Full evaluation reports
- `data/traces/` - Query execution traces
- `test_eval.py` - Unit tests for metrics

---

*Generated: December 3, 2025*

