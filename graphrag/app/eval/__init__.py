"""Evaluation module for GraphRAG vs Vanilla RAG comparison."""

from .datasets import (
    QAPair,
    EvalDataset,
    create_sample_dataset,
    create_pubmed_evaluation_dataset,
    update_gold_passages_from_chunks,
    get_chunk_texts_from_store,
)

from .mcq_dataset import (
    MCQQuestion,
    MCQDataset,
    create_biomedical_mcq_dataset,
    create_quick_mcq_dataset,
)

from .mcq_judge import (
    MCQGrade,
    MCQEvalSummary,
    GradeLevel,
    grade_mcq_response,
    summarize_grades,
)

from .mcq_eval_runner import (
    MCQMethodResult,
    MCQComparisonResult,
    MCQEvalReport,
    run_mcq_evaluation,
    run_quick_mcq_eval,
)

from .metrics import (
    # IR Metrics
    recall_at_k,
    precision_at_k,
    f1_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
    hit_rate_at_k,
    compute_ir_metrics,
    IRMetrics,
    # Answer Metrics
    normalize_text,
    compute_exact_match,
    compute_f1_token,
    compute_context_relevance,
    compute_answer_coverage,
    compute_keyword_coverage,
    compute_answer_length_ratio,
    compute_semantic_similarity,
    compute_rouge_l,
    compute_bleu_1,
    compute_answer_metrics,
    AnswerMetrics,
    # Aggregation
    aggregate_metrics,
)

from .judge import llm_judge

from .vanilla_rag import (
    VanillaRAG,
    VanillaRAGResult,
    vanilla_rag,
    vanilla_rag_query,
    vanilla_rag_retrieve,
)

from .run_eval import (
    SingleQueryResult,
    EvaluationReport,
    run_single_query_eval,
    run_evaluation,
    run_quick_eval,
)

__all__ = [
    # Datasets
    "QAPair",
    "EvalDataset", 
    "create_sample_dataset",
    "create_pubmed_evaluation_dataset",
    "update_gold_passages_from_chunks",
    "get_chunk_texts_from_store",
    # MCQ Datasets
    "MCQQuestion",
    "MCQDataset",
    "create_biomedical_mcq_dataset",
    "create_quick_mcq_dataset",
    # IR Metrics
    "recall_at_k",
    "precision_at_k",
    "f1_at_k",
    "ndcg_at_k",
    "mrr",
    "average_precision",
    "hit_rate_at_k",
    "compute_ir_metrics",
    "IRMetrics",
    # Answer Metrics
    "normalize_text",
    "compute_exact_match",
    "compute_f1_token",
    "compute_context_relevance",
    "compute_answer_coverage",
    "compute_keyword_coverage",
    "compute_answer_length_ratio",
    "compute_semantic_similarity",
    "compute_rouge_l",
    "compute_bleu_1",
    "compute_answer_metrics",
    "AnswerMetrics",
    # Aggregation
    "aggregate_metrics",
    # LLM Judge
    "llm_judge",
    # MCQ Judge
    "MCQGrade",
    "MCQEvalSummary",
    "GradeLevel",
    "grade_mcq_response",
    "summarize_grades",
    # Vanilla RAG
    "VanillaRAG",
    "VanillaRAGResult",
    "vanilla_rag",
    "vanilla_rag_query",
    "vanilla_rag_retrieve",
    # Evaluation
    "SingleQueryResult",
    "EvaluationReport",
    "run_single_query_eval",
    "run_evaluation",
    "run_quick_eval",
    # MCQ Evaluation
    "MCQMethodResult",
    "MCQComparisonResult",
    "MCQEvalReport",
    "run_mcq_evaluation",
    "run_quick_mcq_eval",
]
