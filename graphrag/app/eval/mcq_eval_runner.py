"""
MCQ Evaluation Runner - Compare GraphRAG, Vanilla RAG, and other methods.

Runs MCQ questions through different RAG systems and grades with LLM judge.
"""

import asyncio
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .mcq_dataset import MCQQuestion, MCQDataset, create_biomedical_mcq_dataset, create_quick_mcq_dataset
from .mcq_judge import (
    MCQGrade, MCQEvalSummary, GradeLevel,
    grade_mcq_response, summarize_grades
)
from .vanilla_rag import vanilla_rag
from ..api.routers import query as graphrag_query
from ..api.schemas import QueryRequest
from ..core.providers import get_llm_provider
from ..core.logging import get_logger

logger = get_logger(__name__)


# System prompt for MCQ answering (based on SARG_HYBRID_MCQ)
MCQ_SYSTEM_PROMPT = """You are an expert biomedical researcher and test taker. You will be provided a multiple-choice question that you must answer to the best of your ability.

IMPORTANT RULES:
1. You MUST select exactly ONE answer from the given choices
2. Do NOT respond with "I am not sure" or "None of the above"
3. Use the provided context to inform your answer
4. Think through your reasoning before answering
5. Similar diseases tend to have similar gene associations - use this to make educated guesses

RESPONSE FORMAT:
1. First, analyze the relevant information from the context
2. Consider each answer choice
3. Provide your final answer in JSON format:

```json
{
    "reasoning": "Your step-by-step reasoning",
    "answer": "The exact answer choice you select"
}
```

Always pick exactly one answer choice, even if you're not 100% certain."""


@dataclass
class MCQMethodResult:
    """Results from running MCQ through a single method."""
    method_name: str
    question_id: str
    response: str
    retrieved_context: str
    grade: Optional[MCQGrade] = None
    latency: float = 0.0
    error: Optional[str] = None


@dataclass
class MCQComparisonResult:
    """Comparison of multiple methods on a single question."""
    question: MCQQuestion
    results_by_method: Dict[str, MCQMethodResult] = field(default_factory=dict)


@dataclass
class MCQEvalReport:
    """Full evaluation report comparing methods."""
    dataset_name: str
    timestamp: str
    num_questions: int
    methods: List[str]
    
    # Summary by method
    summaries_by_method: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Per-question results
    question_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Winner analysis
    method_wins: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def print_summary(self) -> None:
        print("\n" + "=" * 80)
        print(f"MCQ EVALUATION REPORT: {self.dataset_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Questions: {self.num_questions}")
        print("=" * 80)
        
        print("\n📊 ACCURACY BY METHOD")
        print("-" * 60)
        print(f"{'Method':<20} {'Accuracy':>10} {'Avg Score':>12} {'Correct':>10}")
        print("-" * 60)
        
        for method, summary in self.summaries_by_method.items():
            acc = summary.get("accuracy", 0) * 100
            avg_score = summary.get("avg_score", 0)
            correct = summary.get("correct", 0)
            print(f"{method:<20} {acc:>9.1f}% {avg_score:>12.3f} {correct:>10}")
        
        print("\n📈 DETAILED METRICS")
        print("-" * 60)
        print(f"{'Method':<20} {'Reasoning':>12} {'Retrieval':>12} {'Partial':>10}")
        print("-" * 60)
        
        for method, summary in self.summaries_by_method.items():
            reasoning = summary.get("avg_reasoning_quality", 0)
            retrieval = summary.get("avg_retrieval_relevance", 0)
            partial = summary.get("partial", 0)
            print(f"{method:<20} {reasoning:>12.3f} {retrieval:>12.3f} {partial:>10}")
        
        print("\n🏆 METHOD WINS (per question)")
        print("-" * 60)
        for method, wins in sorted(self.method_wins.items(), key=lambda x: -x[1]):
            pct = wins / self.num_questions * 100 if self.num_questions > 0 else 0
            print(f"  {method}: {wins} wins ({pct:.1f}%)")
        
        # Determine overall winner
        if self.method_wins:
            winner = max(self.method_wins.items(), key=lambda x: x[1])[0]
            print(f"\n🥇 Overall Winner: {winner}")
        
        print("=" * 80 + "\n")


async def run_graphrag_mcq(question: MCQQuestion, k: int = 10) -> MCQMethodResult:
    """Run a question through GraphRAG."""
    start_time = time.time()
    
    try:
        # Format question with choices
        formatted_q = f"{question.question}\n\nChoices:\n"
        for i, choice in enumerate(question.choices):
            formatted_q += f"  {chr(65+i)}) {choice}\n"
        
        request = QueryRequest(
            query=formatted_q,
            mode="auto",
            k=k,
            rerank=True
        )
        
        response = await graphrag_query(request)
        
        # Extract context from citations
        context = "\n".join([c.text for c in response.citations]) if response.citations else ""
        
        return MCQMethodResult(
            method_name="graphrag",
            question_id=question.id,
            response=response.answer,
            retrieved_context=context,
            latency=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"GraphRAG failed for {question.id}: {e}")
        return MCQMethodResult(
            method_name="graphrag",
            question_id=question.id,
            response="",
            retrieved_context="",
            latency=time.time() - start_time,
            error=str(e)
        )


async def run_vanilla_rag_mcq(question: MCQQuestion, k: int = 10) -> MCQMethodResult:
    """Run a question through Vanilla RAG."""
    start_time = time.time()
    
    try:
        # Format question with choices
        formatted_q = f"{question.question}\n\nChoices:\n"
        for i, choice in enumerate(question.choices):
            formatted_q += f"  {chr(65+i)}) {choice}\n"
        
        result = await vanilla_rag.query(
            query=formatted_q,
            k=k,
            rerank=True
        )
        
        # Extract context
        context = "\n".join([chunk[1] for chunk in result.retrieved_chunks]) if result.retrieved_chunks else ""
        
        return MCQMethodResult(
            method_name="vanilla_rag",
            question_id=question.id,
            response=result.answer,
            retrieved_context=context,
            latency=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Vanilla RAG failed for {question.id}: {e}")
        return MCQMethodResult(
            method_name="vanilla_rag",
            question_id=question.id,
            response="",
            retrieved_context="",
            latency=time.time() - start_time,
            error=str(e)
        )


async def run_direct_llm_mcq(question: MCQQuestion) -> MCQMethodResult:
    """Run a question through LLM directly (no retrieval - baseline)."""
    start_time = time.time()
    
    try:
        llm = get_llm_provider()
        
        # Format question with choices
        formatted_q = f"{question.question}\n\nChoices:\n"
        for i, choice in enumerate(question.choices):
            formatted_q += f"  {chr(65+i)}) {choice}\n"
        
        prompt = f"""Answer this multiple choice question. Pick exactly one answer.

{formatted_q}

Think through your reasoning, then provide your answer in JSON format:
```json
{{"reasoning": "your reasoning", "answer": "exact answer choice"}}
```"""
        
        response = await llm.generate(
            prompt,
            system_prompt=MCQ_SYSTEM_PROMPT,
            max_tokens=1000,
            temperature=0.0
        )
        
        return MCQMethodResult(
            method_name="direct_llm",
            question_id=question.id,
            response=response,
            retrieved_context="",  # No retrieval
            latency=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Direct LLM failed for {question.id}: {e}")
        return MCQMethodResult(
            method_name="direct_llm",
            question_id=question.id,
            response="",
            retrieved_context="",
            latency=time.time() - start_time,
            error=str(e)
        )


async def evaluate_single_question(
    question: MCQQuestion,
    methods: List[str] = ["graphrag", "vanilla_rag", "direct_llm"],
    k: int = 10,
    use_llm_judge: bool = True
) -> MCQComparisonResult:
    """
    Evaluate a single question across multiple methods.
    """
    comparison = MCQComparisonResult(question=question)
    
    for method in methods:
        # Run the appropriate method
        if method == "graphrag":
            result = await run_graphrag_mcq(question, k=k)
        elif method == "vanilla_rag":
            result = await run_vanilla_rag_mcq(question, k=k)
        elif method == "direct_llm":
            result = await run_direct_llm_mcq(question)
        else:
            logger.warning(f"Unknown method: {method}")
            continue
        
        # Grade the result
        if result.response and not result.error:
            result.grade = await grade_mcq_response(
                question_id=question.id,
                question=question.question,
                choices=question.choices,
                correct_answer=question.correct_answer,
                model_response=result.response,
                retrieved_context=result.retrieved_context,
                use_llm_judge=use_llm_judge
            )
        
        comparison.results_by_method[method] = result
    
    return comparison


async def run_mcq_evaluation(
    dataset: Optional[MCQDataset] = None,
    methods: List[str] = ["graphrag", "vanilla_rag", "direct_llm"],
    k: int = 10,
    use_llm_judge: bool = True,
    output_path: Optional[str] = None,
    num_questions: Optional[int] = None
) -> MCQEvalReport:
    """
    Run full MCQ evaluation comparing multiple methods.
    
    Args:
        dataset: MCQ dataset to use (default: biomedical MCQ)
        methods: List of methods to compare
        k: Number of chunks to retrieve
        use_llm_judge: Whether to use LLM for grading
        output_path: Path to save report JSON
        num_questions: Limit number of questions (for testing)
    
    Returns:
        MCQEvalReport with all results
    """
    # Load dataset
    if dataset is None:
        dataset = create_biomedical_mcq_dataset()
    
    questions = dataset.questions
    if num_questions is not None:
        questions = questions[:num_questions]
    
    logger.info(f"Running MCQ evaluation: {len(questions)} questions, methods: {methods}")
    
    # Run evaluations
    all_comparisons: List[MCQComparisonResult] = []
    grades_by_method: Dict[str, List[MCQGrade]] = {m: [] for m in methods}
    
    for i, question in enumerate(questions):
        logger.info(f"Evaluating question {i+1}/{len(questions)}: {question.id}")
        
        comparison = await evaluate_single_question(
            question=question,
            methods=methods,
            k=k,
            use_llm_judge=use_llm_judge
        )
        all_comparisons.append(comparison)
        
        # Collect grades
        for method, result in comparison.results_by_method.items():
            if result.grade:
                grades_by_method[method].append(result.grade)
        
        # Rate limiting
        await asyncio.sleep(0.5)
    
    # Compute summaries
    summaries_by_method = {}
    for method, grades in grades_by_method.items():
        summary = summarize_grades(
            grades,
            [{"id": q.id, "category": q.category, "difficulty": q.difficulty} 
             for q in questions]
        )
        summaries_by_method[method] = summary.to_dict()
    
    # Determine per-question winners
    method_wins = {m: 0 for m in methods}
    question_results = []
    
    for comparison in all_comparisons:
        q_result = {
            "question_id": comparison.question.id,
            "question": comparison.question.question,
            "correct_answer": comparison.question.correct_answer,
            "category": comparison.question.category,
            "difficulty": comparison.question.difficulty,
            "results": {}
        }
        
        best_score = -1
        best_method = None
        
        for method, result in comparison.results_by_method.items():
            q_result["results"][method] = {
                "predicted": result.grade.predicted_answer if result.grade else "",
                "score": result.grade.score if result.grade else 0.0,
                "grade": result.grade.grade.value if result.grade else "error",
                "latency": result.latency,
                "error": result.error
            }
            
            if result.grade and result.grade.score > best_score:
                best_score = result.grade.score
                best_method = method
        
        if best_method:
            method_wins[best_method] += 1
            q_result["winner"] = best_method
        
        question_results.append(q_result)
    
    # Build report
    report = MCQEvalReport(
        dataset_name=dataset.name,
        timestamp=datetime.now().isoformat(),
        num_questions=len(questions),
        methods=methods,
        summaries_by_method=summaries_by_method,
        question_results=question_results,
        method_wins=method_wins
    )
    
    # Save if path provided
    if output_path:
        report.save(output_path)
        logger.info(f"Saved MCQ evaluation report to {output_path}")
    
    return report


async def run_quick_mcq_eval(
    num_questions: int = 5,
    methods: List[str] = ["vanilla_rag", "direct_llm"],
    use_llm_judge: bool = False
) -> MCQEvalReport:
    """
    Run a quick MCQ evaluation for testing.
    """
    dataset = create_quick_mcq_dataset(num_questions)
    
    return await run_mcq_evaluation(
        dataset=dataset,
        methods=methods,
        use_llm_judge=use_llm_judge
    )

