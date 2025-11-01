"""Evaluation datasets."""

from typing import List, Dict
from pydantic import BaseModel


class QAPair(BaseModel):
    """QA pair for evaluation."""

    question: str
    answer: str
    gold_passages: List[str]  # List of chunk IDs that contain the answer


def create_sample_dataset() -> List[QAPair]:
    """Create a small synthetic QA dataset for evaluation based on PubMed topics."""
    # Medical questions that can be answered from PubMed data
    return [
        QAPair(
            question="What are the treatments for asthma?",
            answer="Asthma treatments include inhaled corticosteroids for long-term control, short-acting beta-agonists for quick relief, allergen immunotherapy, and proper inhaler technique.",
            gold_passages=["chunk_1", "chunk_5"],  # Will be updated after ingestion
        ),
        QAPair(
            question="How is diabetes managed?",
            answer="Diabetes management involves blood glucose monitoring, medication (insulin or oral medications), diet control, and regular exercise.",
            gold_passages=["chunk_2", "chunk_7"],  # Will be updated after ingestion
        ),
        QAPair(
            question="What causes hypertension?",
            answer="Hypertension can be caused by genetics, age, obesity, lack of physical activity, high salt intake, and underlying medical conditions.",
            gold_passages=["chunk_3", "chunk_8"],  # Will be updated after ingestion
        ),
    ]

