"""
MCQ Biomedical Dataset for evaluating RAG systems.

Based on SARG_HYBRID_MCQ format with gene/variant/disease associations.
Questions test the ability to retrieve and reason over biomedical knowledge.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel
from dataclasses import dataclass


class MCQQuestion(BaseModel):
    """A multiple choice question for evaluation."""
    
    id: str
    question: str
    choices: List[str]
    correct_answer: str
    category: str  # gene_disease, variant_disease, drug_target, etc.
    difficulty: str  # easy, medium, hard
    reasoning_hint: Optional[str] = None  # Expected reasoning path
    related_entities: List[str] = []  # Entities that should be retrieved


class MCQDataset(BaseModel):
    """Collection of MCQ questions."""
    
    name: str
    description: str
    questions: List[MCQQuestion]


def create_biomedical_mcq_dataset() -> MCQDataset:
    """
    Create a curated MCQ dataset for biomedical QA evaluation.
    
    Categories:
    - gene_disease: Which gene is associated with disease X?
    - variant_disease: Which variant is linked to condition Y?
    - drug_target: What drug targets protein Z?
    - mechanism: What is the mechanism of action?
    - treatment: What is the recommended treatment?
    """
    
    questions = [
        # ===== Asthma-related (from PubMed data) =====
        MCQQuestion(
            id="asthma_001",
            question="Which of the following is the primary first-line controller medication for persistent asthma?",
            choices=[
                "Inhaled corticosteroids (ICS)",
                "Long-acting beta-agonists (LABA)",
                "Leukotriene receptor antagonists (LTRA)",
                "Short-acting beta-agonists (SABA)",
                "Oral corticosteroids"
            ],
            correct_answer="Inhaled corticosteroids (ICS)",
            category="treatment",
            difficulty="easy",
            reasoning_hint="ICS are the cornerstone of asthma management for persistent disease",
            related_entities=["asthma", "inhaled corticosteroids", "ICS", "controller medication"]
        ),
        MCQQuestion(
            id="asthma_002",
            question="Which treatment option is considered the only disease-modifying therapy for allergic respiratory diseases?",
            choices=[
                "Inhaled corticosteroids",
                "Allergen immunotherapy (AIT)",
                "Montelukast",
                "Omalizumab",
                "Tiotropium"
            ],
            correct_answer="Allergen immunotherapy (AIT)",
            category="treatment",
            difficulty="medium",
            reasoning_hint="AIT can prevent disease progression and onset of asthma",
            related_entities=["allergen immunotherapy", "AIT", "disease-modifying", "respiratory allergies"]
        ),
        MCQQuestion(
            id="asthma_003",
            question="Which cell type is hypothesized to be a key player in the pathogenesis of exercise-induced bronchoconstriction (EIB)?",
            choices=[
                "T lymphocytes",
                "Neutrophils",
                "Mast cells",
                "Eosinophils",
                "Macrophages"
            ],
            correct_answer="Mast cells",
            category="mechanism",
            difficulty="hard",
            reasoning_hint="Cold air and pollution trigger mast cell activation in EIB",
            related_entities=["mast cells", "exercise-induced bronchoconstriction", "EIB", "pathogenesis"]
        ),
        MCQQuestion(
            id="asthma_004",
            question="What potential adverse effect has raised concerns about leukotriene receptor antagonists (LTRAs) in pediatric patients?",
            choices=[
                "Hepatotoxicity",
                "Nephrotoxicity",
                "Neuropsychiatric events",
                "Cardiac arrhythmias",
                "Growth suppression"
            ],
            correct_answer="Neuropsychiatric events",
            category="drug_safety",
            difficulty="medium",
            reasoning_hint="Montelukast has FDA warnings about neuropsychiatric effects",
            related_entities=["leukotriene receptor antagonists", "LTRA", "montelukast", "neuropsychiatric", "pediatric"]
        ),
        
        # ===== Diabetes-related =====
        MCQQuestion(
            id="diabetes_001",
            question="In diabetic kidney disease patients with persistent hypotension, which class of medications may be unsafe to use?",
            choices=[
                "Metformin",
                "ACE inhibitors/ARBs and SGLT2 inhibitors",
                "DPP-4 inhibitors",
                "Sulfonylureas",
                "Insulin"
            ],
            correct_answer="ACE inhibitors/ARBs and SGLT2 inhibitors",
            category="treatment",
            difficulty="hard",
            reasoning_hint="Hypotension contraindicates drugs that lower blood pressure further",
            related_entities=["diabetic kidney disease", "hypotension", "ACE inhibitors", "SGLT2 inhibitors"]
        ),
        MCQQuestion(
            id="diabetes_002",
            question="What is the relationship between PTSD and type 2 diabetes?",
            choices=[
                "PTSD is protective against diabetes",
                "They are unrelated conditions",
                "Bidirectional relationship with shared mechanisms",
                "Diabetes causes PTSD",
                "PTSD only affects type 1 diabetes"
            ],
            correct_answer="Bidirectional relationship with shared mechanisms",
            category="mechanism",
            difficulty="medium",
            reasoning_hint="HPA axis dysregulation and chronic inflammation link both conditions",
            related_entities=["PTSD", "type 2 diabetes", "HPA axis", "inflammation", "bidirectional"]
        ),
        
        # ===== Hypertension-related =====
        MCQQuestion(
            id="hypertension_001",
            question="Which class of antihypertensive medications is typically first-line for patients with diabetes and hypertension?",
            choices=[
                "Beta-blockers",
                "Calcium channel blockers",
                "ACE inhibitors or ARBs",
                "Thiazide diuretics",
                "Alpha-blockers"
            ],
            correct_answer="ACE inhibitors or ARBs",
            category="treatment",
            difficulty="easy",
            reasoning_hint="ACE-I/ARBs provide renal protection in diabetic patients",
            related_entities=["hypertension", "diabetes", "ACE inhibitors", "ARBs", "renal protection"]
        ),
        
        # ===== Cardiac-related =====
        MCQQuestion(
            id="cardiac_001",
            question="Which intervention is critical in managing acute coronary syndrome with calcified lesions?",
            choices=[
                "Immediate bypass surgery",
                "Conservative medical management only",
                "Rotational atherectomy or specialized crossing techniques",
                "Thrombolysis alone",
                "Watchful waiting"
            ],
            correct_answer="Rotational atherectomy or specialized crossing techniques",
            category="treatment",
            difficulty="hard",
            reasoning_hint="Calcified lesions may be balloon-uncrossable requiring special techniques",
            related_entities=["coronary artery", "calcification", "PCI", "atherectomy"]
        ),
        
        # ===== Nutraceuticals =====
        MCQQuestion(
            id="nutra_001",
            question="Which natural compound has shown protective effects against metabolic syndrome components including hypertension, insulin resistance, and obesity?",
            choices=[
                "Vitamin C",
                "Syringic acid",
                "Caffeine",
                "Sodium chloride",
                "Ethanol"
            ],
            correct_answer="Syringic acid",
            category="treatment",
            difficulty="medium",
            reasoning_hint="Syringic acid is a phenolic compound with multiple metabolic benefits",
            related_entities=["syringic acid", "metabolic syndrome", "phenolic acid", "hypertension", "insulin resistance"]
        ),
        MCQQuestion(
            id="nutra_002",
            question="Which class of nutraceuticals includes polyphenols, flavonoids, and omega-3 fatty acids?",
            choices=[
                "Synthetic vitamins",
                "Bioactive compounds from foods and herbs",
                "Pharmaceutical antibiotics",
                "Inorganic minerals only",
                "Processed sugars"
            ],
            correct_answer="Bioactive compounds from foods and herbs",
            category="mechanism",
            difficulty="easy",
            reasoning_hint="Nutraceuticals are bioactive compounds with health benefits",
            related_entities=["nutraceuticals", "polyphenols", "flavonoids", "omega-3", "bioactive"]
        ),
        
        # ===== AI in Medicine =====
        MCQQuestion(
            id="ai_001",
            question="In stroke care, what is the primary application of AI platforms like RapidAI?",
            choices=[
                "Scheduling appointments",
                "Detecting large vessel occlusions and perfusion deficits",
                "Managing hospital billing",
                "Patient entertainment",
                "Cafeteria menu planning"
            ],
            correct_answer="Detecting large vessel occlusions and perfusion deficits",
            category="technology",
            difficulty="medium",
            reasoning_hint="AI expedites triage in time-critical stroke scenarios",
            related_entities=["AI", "stroke", "large vessel occlusion", "perfusion", "RapidAI"]
        ),
        
        # ===== Lung Cancer =====
        MCQQuestion(
            id="lung_001",
            question="Which type of therapy has revolutionized the treatment of non-small cell lung cancer by targeting PD-1/PD-L1 pathway?",
            choices=[
                "Chemotherapy",
                "Radiation therapy",
                "Immunotherapy (checkpoint inhibitors)",
                "Hormone therapy",
                "Antibiotic therapy"
            ],
            correct_answer="Immunotherapy (checkpoint inhibitors)",
            category="treatment",
            difficulty="medium",
            reasoning_hint="PD-1/PD-L1 inhibitors like pembrolizumab are immune checkpoint inhibitors",
            related_entities=["lung cancer", "immunotherapy", "PD-1", "PD-L1", "checkpoint inhibitors"]
        ),
        
        # ===== Alzheimer's =====
        MCQQuestion(
            id="alz_001",
            question="Which protein aggregation is a hallmark pathological feature of Alzheimer's disease?",
            choices=[
                "Alpha-synuclein",
                "Amyloid-beta plaques and tau tangles",
                "Huntingtin",
                "Prion proteins",
                "Hemoglobin"
            ],
            correct_answer="Amyloid-beta plaques and tau tangles",
            category="mechanism",
            difficulty="easy",
            reasoning_hint="Amyloid and tau are the key pathological proteins in AD",
            related_entities=["Alzheimer's disease", "amyloid-beta", "tau", "neurodegeneration"]
        ),
        
        # ===== Rheumatoid Arthritis =====
        MCQQuestion(
            id="ra_001",
            question="Which class of drugs specifically targets TNF-alpha in rheumatoid arthritis treatment?",
            choices=[
                "NSAIDs",
                "Corticosteroids",
                "Biologic DMARDs (anti-TNF agents)",
                "Acetaminophen",
                "Muscle relaxants"
            ],
            correct_answer="Biologic DMARDs (anti-TNF agents)",
            category="treatment",
            difficulty="medium",
            reasoning_hint="Anti-TNF biologics like adalimumab target specific inflammatory pathways",
            related_entities=["rheumatoid arthritis", "TNF-alpha", "biologics", "DMARDs", "adalimumab"]
        ),
        
        # ===== Chronic Kidney Disease =====
        MCQQuestion(
            id="ckd_001",
            question="Which class of medications has shown both cardiovascular and renal protective benefits in patients with CKD and diabetes?",
            choices=[
                "Loop diuretics",
                "SGLT2 inhibitors",
                "Calcium supplements",
                "Proton pump inhibitors",
                "Antihistamines"
            ],
            correct_answer="SGLT2 inhibitors",
            category="treatment",
            difficulty="medium",
            reasoning_hint="SGLT2i like empagliflozin reduce both CV events and CKD progression",
            related_entities=["chronic kidney disease", "CKD", "SGLT2 inhibitors", "renal protection", "diabetes"]
        ),
    ]
    
    return MCQDataset(
        name="biomedical_mcq_eval",
        description="Curated MCQ dataset for evaluating biomedical RAG systems on gene/disease associations and treatment knowledge",
        questions=questions
    )


def create_quick_mcq_dataset(n: int = 5) -> MCQDataset:
    """Create a smaller dataset for quick testing."""
    full_dataset = create_biomedical_mcq_dataset()
    return MCQDataset(
        name="biomedical_mcq_quick",
        description=f"Quick test subset with {n} questions",
        questions=full_dataset.questions[:n]
    )


# Category mappings for analysis
QUESTION_CATEGORIES = {
    "treatment": "Treatment/Therapy questions",
    "mechanism": "Mechanism of action/pathophysiology",
    "drug_safety": "Drug safety and adverse effects",
    "technology": "Technology applications in medicine",
    "gene_disease": "Gene-disease associations",
    "variant_disease": "Variant-disease associations",
}

