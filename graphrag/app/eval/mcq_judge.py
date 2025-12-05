"""
LLM-as-a-Judge for MCQ Evaluation with detailed grading rubric.

Based on SARG_HYBRID_MCQ evaluation methodology.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.providers import get_llm_provider
from ..core.logging import get_logger

logger = get_logger(__name__)


class GradeLevel(Enum):
    """Grade levels for MCQ evaluation."""
    CORRECT = "correct"           # Exact match
    PARTIAL = "partial"           # Related/close answer
    INCORRECT = "incorrect"       # Wrong answer
    NO_ANSWER = "no_answer"       # Failed to provide answer


@dataclass
class MCQGrade:
    """Grading result for a single MCQ."""
    question_id: str
    predicted_answer: str
    correct_answer: str
    grade: GradeLevel
    score: float  # 0.0 - 1.0
    reasoning_quality: float  # 0.0 - 1.0 (how well did it reason)
    retrieval_relevance: float  # 0.0 - 1.0 (did it use relevant context)
    confidence: float  # Model's confidence in its answer
    explanation: str  # Why this grade was given
    
    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "predicted_answer": self.predicted_answer,
            "correct_answer": self.correct_answer,
            "grade": self.grade.value,
            "score": self.score,
            "reasoning_quality": self.reasoning_quality,
            "retrieval_relevance": self.retrieval_relevance,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


# Grading rubric for LLM judge
MCQ_GRADING_RUBRIC = """
You are an expert biomedical evaluator grading an MCQ response from a RAG system.

GRADING CRITERIA:
1. **Answer Correctness** (Primary Score):
   - 1.0: Exact match with correct answer
   - 0.5: Semantically equivalent or very close (e.g., same drug class)
   - 0.0: Incorrect answer

2. **Reasoning Quality** (0.0-1.0):
   - Did the model explain its reasoning?
   - Is the reasoning logically sound?
   - Does it reference relevant knowledge?

3. **Retrieval Relevance** (0.0-1.0):
   - Did the response use relevant retrieved context?
   - Was the context appropriately cited/referenced?
   - Did retrieval help arrive at the answer?

4. **Confidence Assessment**:
   - How confident does the model appear in its answer?
   - Does it hedge or express uncertainty appropriately?

OUTPUT FORMAT (JSON only):
```json
{
    "answer_correct": true/false,
    "score": 0.0-1.0,
    "reasoning_quality": 0.0-1.0,
    "retrieval_relevance": 0.0-1.0,
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of the grade"
}
```
"""


async def extract_answer_from_response(response: str, choices: List[str]) -> Tuple[str, float]:
    """
    Extract the selected answer from a model response.
    
    Returns:
        Tuple of (extracted_answer, confidence)
    """
    response_lower = response.lower()
    
    # Try to find JSON answer format first
    json_match = re.search(r'\{[^}]*"answer"[^}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "answer" in data:
                return str(data["answer"]), 0.9
        except json.JSONDecodeError:
            pass
    
    # Look for explicit answer patterns
    patterns = [
        r"(?:the answer is|answer:|correct answer is|i choose|my answer is)[:\s]*[\"']?([^\"'\n]+)[\"']?",
        r"(?:therefore|thus|so)[,\s]+([^.]+?)(?:\.|is the|appears)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower, re.IGNORECASE)
        if match:
            found = match.group(1).strip()
            # Match against choices
            for choice in choices:
                if choice.lower() in found or found in choice.lower():
                    return choice, 0.8
    
    # Check if any choice is mentioned prominently
    choice_mentions = {}
    for choice in choices:
        choice_lower = choice.lower()
        # Count mentions and position
        count = response_lower.count(choice_lower)
        if count > 0:
            # Weight by position (later = more likely the final answer)
            last_pos = response_lower.rfind(choice_lower)
            choice_mentions[choice] = (count, last_pos)
    
    if choice_mentions:
        # Get the choice with highest count, or latest position if tied
        best_choice = max(choice_mentions.items(), key=lambda x: (x[1][0], x[1][1]))[0]
        return best_choice, 0.6
    
    return "", 0.0


async def grade_mcq_response(
    question_id: str,
    question: str,
    choices: List[str],
    correct_answer: str,
    model_response: str,
    retrieved_context: Optional[str] = None,
    use_llm_judge: bool = True
) -> MCQGrade:
    """
    Grade a single MCQ response using LLM-as-a-judge.
    
    Args:
        question_id: Unique identifier for the question
        question: The question text
        choices: List of answer choices
        correct_answer: The correct answer
        model_response: The model's full response
        retrieved_context: Optional retrieved context used
        use_llm_judge: Whether to use LLM for detailed grading
    
    Returns:
        MCQGrade with detailed scoring
    """
    # Extract the answer from response
    predicted_answer, extraction_confidence = await extract_answer_from_response(
        model_response, choices
    )
    
    # Quick scoring without LLM
    if not predicted_answer:
        return MCQGrade(
            question_id=question_id,
            predicted_answer="",
            correct_answer=correct_answer,
            grade=GradeLevel.NO_ANSWER,
            score=0.0,
            reasoning_quality=0.0,
            retrieval_relevance=0.0,
            confidence=0.0,
            explanation="No answer could be extracted from the response"
        )
    
    # Check exact match
    is_correct = predicted_answer.lower().strip() == correct_answer.lower().strip()
    
    if not use_llm_judge:
        # Simple grading without LLM
        return MCQGrade(
            question_id=question_id,
            predicted_answer=predicted_answer,
            correct_answer=correct_answer,
            grade=GradeLevel.CORRECT if is_correct else GradeLevel.INCORRECT,
            score=1.0 if is_correct else 0.0,
            reasoning_quality=0.5,  # Default
            retrieval_relevance=0.5,  # Default
            confidence=extraction_confidence,
            explanation="Exact match comparison" if is_correct else "Answer does not match"
        )
    
    # Use LLM for detailed grading
    try:
        llm = get_llm_provider()
        
        context_section = ""
        if retrieved_context:
            context_section = f"\n\nRETRIEVED CONTEXT:\n{retrieved_context[:2000]}"
        
        judge_prompt = f"""{MCQ_GRADING_RUBRIC}

QUESTION: {question}

ANSWER CHOICES:
{chr(10).join(f'- {c}' for c in choices)}

CORRECT ANSWER: {correct_answer}

MODEL RESPONSE:
{model_response[:3000]}
{context_section}

EXTRACTED ANSWER: {predicted_answer}

Grade this response according to the rubric. Output ONLY valid JSON.
"""
        
        judge_response = await llm.generate(
            judge_prompt,
            max_tokens=500,
            temperature=0.0
        )
        
        # Parse judge response
        json_str = judge_response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        grade_data = json.loads(json_str)
        
        score = float(grade_data.get("score", 1.0 if is_correct else 0.0))
        
        # Determine grade level
        if score >= 0.9:
            grade = GradeLevel.CORRECT
        elif score >= 0.4:
            grade = GradeLevel.PARTIAL
        else:
            grade = GradeLevel.INCORRECT
        
        return MCQGrade(
            question_id=question_id,
            predicted_answer=predicted_answer,
            correct_answer=correct_answer,
            grade=grade,
            score=score,
            reasoning_quality=float(grade_data.get("reasoning_quality", 0.5)),
            retrieval_relevance=float(grade_data.get("retrieval_relevance", 0.5)),
            confidence=float(grade_data.get("confidence", extraction_confidence)),
            explanation=grade_data.get("explanation", "LLM judge evaluation")
        )
        
    except Exception as e:
        logger.warning(f"LLM judge failed, falling back to simple grading: {e}")
        return MCQGrade(
            question_id=question_id,
            predicted_answer=predicted_answer,
            correct_answer=correct_answer,
            grade=GradeLevel.CORRECT if is_correct else GradeLevel.INCORRECT,
            score=1.0 if is_correct else 0.0,
            reasoning_quality=0.5,
            retrieval_relevance=0.5,
            confidence=extraction_confidence,
            explanation=f"Fallback grading (LLM judge error: {str(e)[:100]})"
        )


@dataclass
class MCQEvalSummary:
    """Summary statistics for MCQ evaluation."""
    total_questions: int
    correct: int
    partial: int
    incorrect: int
    no_answer: int
    accuracy: float
    avg_score: float
    avg_reasoning_quality: float
    avg_retrieval_relevance: float
    grades_by_category: Dict[str, Dict[str, float]]
    grades_by_difficulty: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        return {
            "total_questions": self.total_questions,
            "correct": self.correct,
            "partial": self.partial,
            "incorrect": self.incorrect,
            "no_answer": self.no_answer,
            "accuracy": self.accuracy,
            "avg_score": self.avg_score,
            "avg_reasoning_quality": self.avg_reasoning_quality,
            "avg_retrieval_relevance": self.avg_retrieval_relevance,
            "grades_by_category": self.grades_by_category,
            "grades_by_difficulty": self.grades_by_difficulty,
        }


def summarize_grades(
    grades: List[MCQGrade],
    questions: List[Dict] = None
) -> MCQEvalSummary:
    """
    Summarize grading results.
    
    Args:
        grades: List of MCQGrade objects
        questions: Optional list of question dicts with category/difficulty
    
    Returns:
        MCQEvalSummary with aggregate statistics
    """
    if not grades:
        return MCQEvalSummary(
            total_questions=0, correct=0, partial=0, incorrect=0, no_answer=0,
            accuracy=0.0, avg_score=0.0, avg_reasoning_quality=0.0,
            avg_retrieval_relevance=0.0, grades_by_category={}, grades_by_difficulty={}
        )
    
    # Count by grade level
    correct = sum(1 for g in grades if g.grade == GradeLevel.CORRECT)
    partial = sum(1 for g in grades if g.grade == GradeLevel.PARTIAL)
    incorrect = sum(1 for g in grades if g.grade == GradeLevel.INCORRECT)
    no_answer = sum(1 for g in grades if g.grade == GradeLevel.NO_ANSWER)
    
    # Compute averages
    avg_score = sum(g.score for g in grades) / len(grades)
    avg_reasoning = sum(g.reasoning_quality for g in grades) / len(grades)
    avg_retrieval = sum(g.retrieval_relevance for g in grades) / len(grades)
    
    # Group by category and difficulty if questions provided
    grades_by_category = {}
    grades_by_difficulty = {}
    
    if questions:
        q_map = {q.get("id") or q.get("question_id"): q for q in questions}
        
        for grade in grades:
            q = q_map.get(grade.question_id, {})
            
            cat = q.get("category", "unknown")
            diff = q.get("difficulty", "unknown")
            
            if cat not in grades_by_category:
                grades_by_category[cat] = {"count": 0, "correct": 0, "avg_score": 0.0}
            grades_by_category[cat]["count"] += 1
            if grade.grade == GradeLevel.CORRECT:
                grades_by_category[cat]["correct"] += 1
            grades_by_category[cat]["avg_score"] += grade.score
            
            if diff not in grades_by_difficulty:
                grades_by_difficulty[diff] = {"count": 0, "correct": 0, "avg_score": 0.0}
            grades_by_difficulty[diff]["count"] += 1
            if grade.grade == GradeLevel.CORRECT:
                grades_by_difficulty[diff]["correct"] += 1
            grades_by_difficulty[diff]["avg_score"] += grade.score
        
        # Compute averages
        for cat_data in grades_by_category.values():
            if cat_data["count"] > 0:
                cat_data["accuracy"] = cat_data["correct"] / cat_data["count"]
                cat_data["avg_score"] /= cat_data["count"]
        
        for diff_data in grades_by_difficulty.values():
            if diff_data["count"] > 0:
                diff_data["accuracy"] = diff_data["correct"] / diff_data["count"]
                diff_data["avg_score"] /= diff_data["count"]
    
    return MCQEvalSummary(
        total_questions=len(grades),
        correct=correct,
        partial=partial,
        incorrect=incorrect,
        no_answer=no_answer,
        accuracy=correct / len(grades),
        avg_score=avg_score,
        avg_reasoning_quality=avg_reasoning,
        avg_retrieval_relevance=avg_retrieval,
        grades_by_category=grades_by_category,
        grades_by_difficulty=grades_by_difficulty,
    )

