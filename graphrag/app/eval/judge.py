"""LLM-based judge for answer quality evaluation."""

from typing import Dict
from ..core.providers import get_llm_provider
from ..core.logging import get_logger

logger = get_logger(__name__)


async def llm_judge(answer: str, reference: str, rubric: str = None) -> Dict[str, float]:
    """
    Judge answer quality using LLM.
    
    Returns dict with scores for: accuracy, completeness, relevance (0-1 scale).
    """
    default_rubric = """
    Evaluate the answer on:
    1. Accuracy: Is the information factually correct?
    2. Completeness: Does it fully address the question?
    3. Relevance: Is all content relevant to the question?
    
    Score each dimension 0-1.
    """
    
    prompt = f"""Reference Answer: {reference}

Generated Answer: {answer}

{rubric or default_rubric}

Output JSON: {{"accuracy": 0.0-1.0, "completeness": 0.0-1.0, "relevance": 0.0-1.0}}
"""
    
    llm = get_llm_provider()
    try:
        response = await llm.generate(prompt, max_tokens=200, temperature=0.0)
        import json
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        
        scores = json.loads(json_str)
        return {
            "accuracy": float(scores.get("accuracy", 0.0)),
            "completeness": float(scores.get("completeness", 0.0)),
            "relevance": float(scores.get("relevance", 0.0)),
        }
    except Exception as e:
        logger.error("LLM judge failed", error=str(e))
        return {"accuracy": 0.0, "completeness": 0.0, "relevance": 0.0}

