"""Evaluation datasets with ground truth for GraphRAG vs Vanilla RAG comparison."""

from typing import List, Dict, Optional, Set
from pydantic import BaseModel
from dataclasses import dataclass


class QAPair(BaseModel):
    """QA pair for evaluation with comprehensive ground truth."""

    question: str
    answer: str  # Reference/gold answer
    gold_passages: List[str]  # List of chunk IDs that contain the answer
    keywords: List[str] = []  # Key terms that should appear in correct chunks
    difficulty: str = "medium"  # easy, medium, hard
    reasoning_type: str = "factual"  # factual, multi-hop, comparative, temporal


class EvalDataset(BaseModel):
    """Full evaluation dataset."""

    name: str
    description: str
    qa_pairs: List[QAPair]
    domain: str = "medical"


def create_pubmed_evaluation_dataset() -> EvalDataset:
    """
    Create comprehensive evaluation dataset based on PubMed medical articles.
    These questions are designed to test different retrieval capabilities.
    """
    return EvalDataset(
        name="pubmed_medical_eval",
        description="Medical QA evaluation dataset from PubMed articles on asthma and diabetes",
        domain="medical",
        qa_pairs=[
            # ===== Asthma-related questions =====
            QAPair(
                question="What are the main treatments for asthma?",
                answer="Asthma treatments include inhaled corticosteroids (ICS) for long-term control, long-acting beta-agonists (LABA) for symptom relief, allergen immunotherapy (AIT) as a disease-modifying treatment, leukotriene receptor antagonists (LTRA) like montelukast, and long-acting muscarinic antagonists (LAMA) like tiotropium as add-on therapy.",
                gold_passages=[],  # Will be populated after ingestion
                keywords=["inhaled corticosteroids", "ICS", "LABA", "allergen immunotherapy", "AIT", "montelukast", "tiotropium", "LTRA"],
                difficulty="easy",
                reasoning_type="factual"
            ),
            QAPair(
                question="How does allergen immunotherapy work for allergic patients?",
                answer="Allergen immunotherapy (AIT) is the only disease-modifying treatment for respiratory allergies that can prevent disease progression and the onset of asthma. It works by modulating the immune response to allergens through either sublingual or subcutaneous administration.",
                gold_passages=[],
                keywords=["allergen immunotherapy", "AIT", "disease-modifying", "sublingual", "subcutaneous", "asthma prevention"],
                difficulty="medium",
                reasoning_type="factual"
            ),
            QAPair(
                question="What is the relationship between BMI and lung function in asthmatic children?",
                answer="There is a nonlinear relationship between body mass index (BMI) z-scores and lung function parameters in asthmatic children. This relationship has implications for individualized treatment and management strategies in pediatric asthma patients.",
                gold_passages=[],
                keywords=["BMI", "body mass index", "lung function", "children", "asthmatic", "nonlinear"],
                difficulty="medium",
                reasoning_type="factual"
            ),
            QAPair(
                question="What are the neuropsychiatric concerns with leukotriene receptor antagonists in children?",
                answer="Leukotriene receptor antagonists (LTRAs) like montelukast, widely prescribed for pediatric asthma, have been associated with potential neuropsychiatric adverse events (NPEs). Studies comparing LTRAs with inhaled corticosteroids have shown varying findings, and direct comparisons of NPE risk remain limited in the pediatric population.",
                gold_passages=[],
                keywords=["leukotriene receptor antagonists", "LTRA", "neuropsychiatric", "montelukast", "children", "adverse events"],
                difficulty="hard",
                reasoning_type="comparative"
            ),
            QAPair(
                question="How do mast cells contribute to exercise-induced bronchoconstriction?",
                answer="Mast cells are essential in the development of exercise-induced bronchoconstriction (EIB). Cold air and air pollution are known triggers, and mast cells are hypothesized to be key players in the pathogenesis of EIB.",
                gold_passages=[],
                keywords=["mast cells", "exercise-induced bronchoconstriction", "EIB", "cold air", "pathogenesis"],
                difficulty="hard",
                reasoning_type="factual"
            ),
            QAPair(
                question="What role do epithelial cells play in severe asthma?",
                answer="Airway epithelial cells function as the first physical barrier against pathogens and are key regulators of immune responses. They produce a wide array of cytokines involved in both innate and adaptive immunity, playing a crucial role in the pathogenesis of severe asthma.",
                gold_passages=[],
                keywords=["epithelial cells", "cytokines", "innate immunity", "adaptive immunity", "severe asthma", "pathogens"],
                difficulty="medium",
                reasoning_type="factual"
            ),
            QAPair(
                question="How do aerobic training and behavioral interventions compare for asthma control?",
                answer="Both aerobic training (AT) and behavioral intervention (BI) aimed at increasing physical activity provide benefits to patients with asthma. However, the comparison between these two interventions in the clinical control of asthma is not well understood.",
                gold_passages=[],
                keywords=["aerobic training", "behavioral intervention", "physical activity", "asthma control", "clinical"],
                difficulty="medium",
                reasoning_type="comparative"
            ),
            
            # ===== Diabetes-related questions =====
            QAPair(
                question="How is diabetic kidney disease managed in patients with persistent hypotension?",
                answer="In diabetic kidney disease (DKD) patients with persistent hypotension where ACE inhibitors/ARBs and SGLT2 inhibitors are unsafe, management emphasizes non-pharmacologic measures including tight glycemic control, weight reduction, hydration, and avoidance of nephrotoxins.",
                gold_passages=[],
                keywords=["diabetic kidney disease", "hypotension", "ACE inhibitors", "SGLT2", "glycemic control", "nephrotoxins"],
                difficulty="hard",
                reasoning_type="factual"
            ),
            QAPair(
                question="What is the relationship between PTSD and type 2 diabetes?",
                answer="There is a bidirectional relationship between post-traumatic stress disorder (PTSD) and type 2 diabetes (T2D). PTSD increases T2D risk and is associated with worse glycemic control, while T2D populations exhibit higher PTSD prevalence (30-50% in high-trauma groups). The relationship is driven by HPA axis dysregulation, chronic inflammation, and behavioral factors.",
                gold_passages=[],
                keywords=["PTSD", "type 2 diabetes", "bidirectional", "HPA axis", "inflammation", "glycemic control"],
                difficulty="hard",
                reasoning_type="multi-hop"
            ),
        QAPair(
                question="What are the benefits of nutraceuticals in chronic disease management?",
                answer="Nutraceuticals - bioactive compounds from foods, herbs, and marine organisms - offer anti-inflammatory, antioxidant, immunomodulatory, neuroprotective, and cardioprotective properties. Key classes include polyphenols, flavonoids, omega-3 fatty acids, probiotics, and plant alkaloids. They work through oxidative stress mitigation, immune modulation, gene regulation, and signaling pathway interactions.",
                gold_passages=[],
                keywords=["nutraceuticals", "polyphenols", "flavonoids", "omega-3", "antioxidant", "anti-inflammatory", "chronic diseases"],
                difficulty="medium",
                reasoning_type="factual"
        ),
        QAPair(
                question="How does syringic acid help with metabolic syndrome?",
                answer="Syringic acid is a natural phenolic acid compound that shows protective effects against the main components of metabolic syndrome including hypertension, insulin resistance, hyperlipidemia, and obesity. It helps reduce the risk of cardiovascular diseases, stroke, and type 2 diabetes.",
                gold_passages=[],
                keywords=["syringic acid", "metabolic syndrome", "hypertension", "insulin resistance", "hyperlipidemia", "obesity"],
                difficulty="medium",
                reasoning_type="factual"
            ),
            
            # ===== Cross-domain questions =====
            QAPair(
                question="How can AI be used in stroke care?",
                answer="AI has applications in stroke care across diagnostic, predictive, and operational domains. In diagnostics, AI platforms detect large vessel occlusions, hemorrhage, and perfusion deficits. Predictive tools support outcome forecasting and hemorrhagic risk stratification. Workflow applications improve communication, accelerate decision-making, and reduce treatment delays.",
                gold_passages=[],
                keywords=["AI", "stroke", "diagnostic", "predictive", "large vessel occlusion", "hemorrhage", "workflow"],
                difficulty="medium",
                reasoning_type="factual"
        ),
        QAPair(
                question="What challenges exist in implementing allergen immunotherapy across Europe?",
                answer="Key challenges for AIT in Europe include: lacking class effect requiring product-specific assessment, adherence issues, appropriate prescription, early initiation, increasing awareness among payers and healthcare providers, defining optimal patient populations, improving training of general practitioners, and providing detailed scientific guidance for product selection.",
                gold_passages=[],
                keywords=["allergen immunotherapy", "Europe", "adherence", "prescription", "awareness", "training", "guidelines"],
                difficulty="hard",
                reasoning_type="multi-hop"
            ),
        ],
    )


def create_sample_dataset() -> List[QAPair]:
    """Create sample QA dataset for quick testing (backward compatible)."""
    dataset = create_pubmed_evaluation_dataset()
    return dataset.qa_pairs[:3]  # Return first 3 for quick tests


def update_gold_passages_from_chunks(
    dataset: EvalDataset, 
    chunk_texts: Dict[str, str],
    match_threshold: float = 0.3
) -> EvalDataset:
    """
    Update gold_passages in dataset by matching keywords to actual chunks.
    
    Args:
        dataset: Evaluation dataset with QA pairs
        chunk_texts: Dict of chunk_id -> chunk_text
        match_threshold: Minimum fraction of keywords that must match
    
    Returns:
        Updated dataset with gold_passages populated
    """
    for qa in dataset.qa_pairs:
        matching_chunks = []
        keywords_lower = [kw.lower() for kw in qa.keywords]
        
        for chunk_id, text in chunk_texts.items():
            text_lower = text.lower()
            
            # Count keyword matches
            matches = sum(1 for kw in keywords_lower if kw in text_lower)
            match_ratio = matches / len(keywords_lower) if keywords_lower else 0
            
            if match_ratio >= match_threshold:
                matching_chunks.append((chunk_id, match_ratio))
        
        # Sort by match ratio and take top matches
        matching_chunks.sort(key=lambda x: x[1], reverse=True)
        qa.gold_passages = [chunk_id for chunk_id, _ in matching_chunks[:5]]
    
    return dataset


def get_chunk_texts_from_store() -> Dict[str, str]:
    """Load all chunks from the metadata store."""
    try:
        from ..data.metadata_store import ChunkModel, metadata_store
        session = metadata_store.get_session()
        chunks = session.query(ChunkModel).all()
        session.close()
        return {chunk.chunk_id: chunk.text for chunk in chunks}
    except Exception as e:
        print(f"Error loading chunks: {e}")
        return {}
