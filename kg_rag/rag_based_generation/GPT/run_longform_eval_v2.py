"""
Long-form QA Evaluation Script v2

Enhanced Features:
a) Chain of reasoning - shows all triples and their usage order
b) Triple deduplication - merges KG and semantic triples, removes redundancy
c) Debug mode - displays retrieved evidence
d) Enhanced rubric with rankings and multiple judge models
e) More evaluation questions generated via Gemini

"""

import sys
import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from kg_rag.utility import *
from kg_rag.config_loader import *

# ============== CONFIGURATION ==============

DEBUG_MODE = True  # Set to True to see retrieved evidence (requirement c)
SHOW_REASONING_CHAIN = True  # Set to True to see triple chain (requirement a)

# ============== DATA CLASSES ==============

@dataclass
class Triple:
    """Represents a knowledge triple."""
    subject: str
    relation: str
    object: str
    source: str  # 'kg' or 'semantic'
    evidence: str = ""
    
    def __hash__(self):
        return hash((self.subject.lower(), self.relation.lower(), self.object.lower()))
    
    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject.lower() == other.subject.lower() and
                self.relation.lower() == other.relation.lower() and
                self.object.lower() == other.object.lower())
    
    def to_string(self):
        return f"({self.subject}) --[{self.relation}]--> ({self.object})"

@dataclass
class ReasoningChain:
    """Stores the chain of reasoning for transparency (requirement a)."""
    question: str
    kg_triples: List[Triple] = field(default_factory=list)
    semantic_triples: List[Triple] = field(default_factory=list)
    merged_triples: List[Triple] = field(default_factory=list)
    removed_duplicates: List[Triple] = field(default_factory=list)
    triple_usage_order: List[str] = field(default_factory=list)
    kg_context_raw: str = ""
    semantic_context_raw: str = ""
    
    def display(self):
        """Print the reasoning chain for debugging."""
        print("\n" + "="*70)
        print("🔍 REASONING CHAIN ANALYSIS")
        print("="*70)
        print(f"\n📝 Question: {self.question[:100]}...")
        
        print(f"\n📊 KG Triples ({len(self.kg_triples)} extracted):")
        for i, t in enumerate(self.kg_triples[:10], 1):
            print(f"   {i}. {t.to_string()}")
        if len(self.kg_triples) > 10:
            print(f"   ... and {len(self.kg_triples)-10} more")
        
        print(f"\n📚 Semantic Triples ({len(self.semantic_triples)} extracted):")
        for i, t in enumerate(self.semantic_triples[:10], 1):
            print(f"   {i}. {t.to_string()}")
        if len(self.semantic_triples) > 10:
            print(f"   ... and {len(self.semantic_triples)-10} more")
        
        print(f"\n🔄 Removed Duplicates ({len(self.removed_duplicates)}):")
        for t in self.removed_duplicates[:5]:
            print(f"   - {t.to_string()}")
        
        print(f"\n✅ Final Merged Triples ({len(self.merged_triples)}):")
        for i, t in enumerate(self.merged_triples[:15], 1):
            print(f"   {i}. [{t.source.upper()}] {t.to_string()}")
        
        if self.triple_usage_order:
            print(f"\n📋 Triple Usage Order in LLM Response:")
            for i, usage in enumerate(self.triple_usage_order[:10], 1):
                print(f"   {i}. {usage}")
        print("="*70)

@dataclass
class LongFormQuestion:
    """A long-form question for evaluation."""
    id: str
    question: str
    question_type: str
    difficulty: str
    reference_answer: str
    key_points: List[str]
    required_entities: List[str]

@dataclass 
class EnhancedScore:
    """Enhanced scoring with more parameters (requirement d)."""
    # Core metrics (0-10)
    accuracy: float = 5.0
    completeness: float = 5.0
    relevance: float = 5.0
    coherence: float = 5.0
    
    # Additional metrics (requirement d)
    conciseness: float = 5.0  # Not too verbose, not too brief
    evidence_usage: float = 5.0  # How well evidence is cited
    reasoning_depth: float = 5.0  # Quality of reasoning
    factual_grounding: float = 5.0  # Based on retrieved facts vs hallucination
    
    # Rankings (1-3, 1 is best)
    accuracy_rank: int = 0
    completeness_rank: int = 0
    relevance_rank: int = 0
    coherence_rank: int = 0
    conciseness_rank: int = 0
    overall_rank: int = 0
    
    # Feedback
    feedback: str = ""
    key_points_found: List[str] = field(default_factory=list)
    key_points_missing: List[str] = field(default_factory=list)
    
    @property
    def total_score(self) -> float:
        return (self.accuracy + self.completeness + self.relevance + 
                self.coherence + self.conciseness + self.evidence_usage +
                self.reasoning_depth + self.factual_grounding) / 8
    
    @property
    def weighted_score(self) -> float:
        """Weighted score emphasizing accuracy and completeness."""
        return (self.accuracy * 0.20 + self.completeness * 0.20 + 
                self.relevance * 0.15 + self.coherence * 0.10 +
                self.conciseness * 0.10 + self.evidence_usage * 0.10 +
                self.reasoning_depth * 0.10 + self.factual_grounding * 0.05)

# ============== TRIPLE EXTRACTION & DEDUPLICATION (requirement b) ==============

def extract_triples_from_text(text: str, source: str) -> List[Triple]:
    """Extract triples from context text."""
    triples = []
    
    # Pattern matching for common relationship formats
    patterns = [
        # "Entity1 associates Entity2" pattern
        r'(\w+(?:\s+\w+)*?)\s+(?:associates?|relates? to|links? to|connects? to)\s+(\w+(?:\s+\w+)*)',
        # "Entity1 causes Entity2" pattern  
        r'(\w+(?:\s+\w+)*?)\s+(?:causes?|leads? to|results? in)\s+(\w+(?:\s+\w+)*)',
        # Gene-Disease patterns
        r'Gene\s+(\w+)\s+(?:associates?|is associated with)\s+Disease\s+(\w+(?:\s+\w+)*)',
        r'Disease\s+(\w+(?:\s+\w+)*?)\s+associates?\s+Gene\s+(\w+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                triples.append(Triple(
                    subject=match[0].strip(),
                    relation="ASSOCIATES",
                    object=match[1].strip() if len(match) > 1 else "",
                    source=source
                ))
    
    return triples

def parse_llm_triples(llm_output: str, source: str) -> List[Triple]:
    """Parse triples from LLM extraction output."""
    triples = []
    
    # Pattern: (Entity1) --[RELATION]--> (Entity2)
    pattern1 = r'\(([^)]+)\)\s*--?\[([^\]]+)\]--?>\s*\(([^)]+)\)'
    matches = re.findall(pattern1, llm_output)
    for m in matches:
        triples.append(Triple(subject=m[0].strip(), relation=m[1].strip(), 
                             object=m[2].strip(), source=source))
    
    # Pattern: Entity1 → RELATION → Entity2
    pattern2 = r'([^→\n]+)\s*→\s*([^→\n]+)\s*→\s*([^→\n]+)'
    matches = re.findall(pattern2, llm_output)
    for m in matches:
        triples.append(Triple(subject=m[0].strip(), relation=m[1].strip(),
                             object=m[2].strip(), source=source))
    
    # Pattern: • [Entity1] --[Relation]--> [Entity2]
    pattern3 = r'\[([^\]]+)\]\s*--\[([^\]]+)\]-->\s*\[([^\]]+)\]'
    matches = re.findall(pattern3, llm_output)
    for m in matches:
        triples.append(Triple(subject=m[0].strip(), relation=m[1].strip(),
                             object=m[2].strip(), source=source))
    
    return triples

def deduplicate_triples(kg_triples: List[Triple], semantic_triples: List[Triple]) -> Tuple[List[Triple], List[Triple]]:
    """
    Merge and deduplicate triples from KG and semantic sources (requirement b).
    Returns: (merged_triples, removed_duplicates)
    """
    seen = set()
    merged = []
    duplicates = []
    
    # Prioritize KG triples (more structured/reliable)
    for t in kg_triples:
        key = (t.subject.lower(), t.relation.lower(), t.object.lower())
        if key not in seen:
            seen.add(key)
            merged.append(t)
    
    # Add semantic triples, marking duplicates
    for t in semantic_triples:
        key = (t.subject.lower(), t.relation.lower(), t.object.lower())
        if key not in seen:
            seen.add(key)
            merged.append(t)
        else:
            duplicates.append(t)
    
    # Also check for near-duplicates (similar entities)
    # Simple approach: check if subject and object are substrings
    final_merged = []
    for t in merged:
        is_redundant = False
        for existing in final_merged:
            if (t.subject.lower() in existing.subject.lower() or 
                existing.subject.lower() in t.subject.lower()) and \
               (t.object.lower() in existing.object.lower() or
                existing.object.lower() in t.object.lower()):
                is_redundant = True
                duplicates.append(t)
                break
        if not is_redundant:
            final_merged.append(t)
    
    return final_merged, duplicates

def analyze_triple_usage(answer: str, triples: List[Triple]) -> List[str]:
    """Analyze which triples were used in the answer and in what order (requirement a)."""
    usage_order = []
    answer_lower = answer.lower()
    
    for t in triples:
        # Check if both subject and object appear in the answer
        if t.subject.lower() in answer_lower and t.object.lower() in answer_lower:
            # Find first occurrence position
            pos = min(answer_lower.find(t.subject.lower()), 
                     answer_lower.find(t.object.lower()))
            usage_order.append((pos, t.to_string()))
    
    # Sort by position and return just the strings
    usage_order.sort(key=lambda x: x[0])
    return [u[1] for u in usage_order]

# ============== ENHANCED JUDGE RUBRIC (requirement d) ==============

ENHANCED_JUDGE_RUBRIC = """
You are an expert medical evaluator. Score the answer on these 8 criteria (0-10 each):

## Evaluation Criteria

### 1. Accuracy (0-10)
- Are medical/scientific facts correct?
- Any hallucinations or fabrications?
- Consistent with medical literature?

### 2. Completeness (0-10)
- Covers all key aspects of the question?
- Important details included?
- Check against key points list

### 3. Relevance (0-10)
- Directly addresses the question?
- Minimal extraneous information?

### 4. Coherence (0-10)
- Well-organized?
- Logical flow?
- Appropriate terminology?

### 5. Conciseness (0-10) [NEW]
- Appropriately detailed without being verbose?
- Gets to the point while being thorough?
- No unnecessary repetition?

### 6. Evidence Usage (0-10) [NEW]
- Cites retrieved evidence effectively?
- Grounds claims in the provided context?
- Makes good use of the knowledge graph data?

### 7. Reasoning Depth (0-10) [NEW]
- Shows clear reasoning chains?
- Connects concepts logically?
- Explains mechanisms, not just facts?

### 8. Factual Grounding (0-10) [NEW]
- Based on retrieved facts vs. generating new claims?
- Stays within the bounds of provided evidence?
- Avoids speculation when evidence is lacking?

## Scoring Guidelines
- 9-10: Excellent
- 7-8: Good
- 5-6: Adequate
- 3-4: Poor
- 1-2: Very Poor
- 0: Completely wrong

Output as JSON:
```json
{
    "accuracy": <0-10>,
    "completeness": <0-10>,
    "relevance": <0-10>,
    "coherence": <0-10>,
    "conciseness": <0-10>,
    "evidence_usage": <0-10>,
    "reasoning_depth": <0-10>,
    "factual_grounding": <0-10>,
    "key_points_found": ["point1", "point2"],
    "key_points_missing": ["point1"],
    "feedback": "2-3 sentence overall assessment"
}
```
"""

def build_enhanced_judge_prompt(question: LongFormQuestion, answer: str, 
                                 context: str = None, reasoning_chain: ReasoningChain = None) -> str:
    """Build enhanced evaluation prompt."""
    prompt = f"""{ENHANCED_JUDGE_RUBRIC}

## Question Being Evaluated
**Question**: {question.question}
**Type**: {question.question_type}
**Difficulty**: {question.difficulty}

## Reference Answer
{question.reference_answer}

## Key Points to Cover
{chr(10).join(f'- {p}' for p in question.key_points)}

## Generated Answer to Evaluate
{answer}
"""
    
    if reasoning_chain and SHOW_REASONING_CHAIN:
        prompt += f"""
## Triples Used (for Evidence Usage scoring)
{chr(10).join(t.to_string() for t in reasoning_chain.merged_triples[:20])}
"""
    
    return prompt

def parse_enhanced_judge_response(response: str) -> EnhancedScore:
    """Parse enhanced judge response."""
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
        
        data = json.loads(json_str)
        
        return EnhancedScore(
            accuracy=float(data.get('accuracy', 5)),
            completeness=float(data.get('completeness', 5)),
            relevance=float(data.get('relevance', 5)),
            coherence=float(data.get('coherence', 5)),
            conciseness=float(data.get('conciseness', 5)),
            evidence_usage=float(data.get('evidence_usage', 5)),
            reasoning_depth=float(data.get('reasoning_depth', 5)),
            factual_grounding=float(data.get('factual_grounding', 5)),
            feedback=data.get('feedback', ''),
            key_points_found=data.get('key_points_found', []),
            key_points_missing=data.get('key_points_missing', [])
        )
    except Exception as e:
        print(f"Parse error: {e}")
        return EnhancedScore(feedback=f"Parse error: {str(e)[:100]}")

def assign_rankings(scores_by_system: Dict[str, EnhancedScore]) -> Dict[str, EnhancedScore]:
    """Assign rankings (1-3) for each criterion (requirement d)."""
    metrics = ['accuracy', 'completeness', 'relevance', 'coherence', 'conciseness']
    
    for metric in metrics:
        # Sort systems by this metric (descending)
        sorted_systems = sorted(scores_by_system.items(), 
                               key=lambda x: getattr(x[1], metric), reverse=True)
        for rank, (sys_name, score) in enumerate(sorted_systems, 1):
            setattr(score, f'{metric}_rank', rank)
    
    # Overall ranking by total score
    sorted_overall = sorted(scores_by_system.items(),
                           key=lambda x: x[1].total_score, reverse=True)
    for rank, (sys_name, score) in enumerate(sorted_overall, 1):
        score.overall_rank = rank
    
    return scores_by_system

# ============== ENHANCED SARG+ORAG SYSTEM ==============

class EnhancedSARGORAG:
    """Enhanced SARG+ORAG with triple deduplication and reasoning chain."""
    
    def __init__(self, vectorstore, embedding_function, node_context_df,
                 semantic_index, semantic_chunks, semantic_embed_model, config):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.node_context_df = node_context_df
        self.semantic_index = semantic_index
        self.semantic_chunks = semantic_chunks
        self.semantic_embed_model = semantic_embed_model
        self.config = config
    
    def generate(self, question: str, system_prompt: str) -> Tuple[str, str, ReasoningChain]:
        """Generate with full reasoning chain tracking."""
        
        reasoning = ReasoningChain(question=question)
        
        # 1. KG retrieval
        kg_context = retrieve_context(
            question, self.vectorstore, self.embedding_function,
            self.node_context_df, self.config['context_volume'],
            self.config['similarity_threshold'], self.config['min_similarity'],
            edge_evidence=False, model_id="gemini-2.0-flash"
        )
        reasoning.kg_context_raw = kg_context
        
        # 2. Semantic retrieval
        semantic_context = semantic_retrieve(
            question, self.semantic_index, self.semantic_chunks,
            self.semantic_embed_model, k=5
        )
        reasoning.semantic_context_raw = semantic_context
        
        # 3. Display evidence if debug mode (requirement c)
        if DEBUG_MODE:
            print("\n" + "-"*50)
            print("📂 KG CONTEXT RETRIEVED:")
            print("-"*50)
            print(kg_context[:1500] + "..." if len(kg_context) > 1500 else kg_context)
            print("\n" + "-"*50)
            print("📂 SEMANTIC CONTEXT RETRIEVED:")
            print("-"*50)
            print(semantic_context[:1500] + "..." if len(semantic_context) > 1500 else semantic_context)
        
        # 4. Extract triples from KG context
        kg_triple_prompt = """Extract biomedical relationships from this knowledge graph data.
Format: (Entity1) --[RELATIONSHIP]--> (Entity2)
Focus on Gene-Disease, Disease-Disease associations."""
        
        kg_triples_text = get_Gemini_response(kg_context[:3000], kg_triple_prompt, temperature=0)
        reasoning.kg_triples = parse_llm_triples(kg_triples_text, 'kg')
        
        # 5. Extract triples from semantic context  
        sem_triple_prompt = """Extract biomedical relationships from this text.
Format: (Entity1) --[RELATIONSHIP]--> (Entity2)
Focus on mechanisms, pathways, and associations."""
        
        sem_triples_text = get_Gemini_response(semantic_context[:3000], sem_triple_prompt, temperature=0)
        reasoning.semantic_triples = parse_llm_triples(sem_triples_text, 'semantic')
        
        # 6. Deduplicate triples (requirement b)
        reasoning.merged_triples, reasoning.removed_duplicates = deduplicate_triples(
            reasoning.kg_triples, reasoning.semantic_triples
        )
        
        # 7. Build final triples string
        triples_str = "\n".join([
            f"[{t.source.upper()}] {t.to_string()}" 
            for t in reasoning.merged_triples[:25]
        ])
        
        # 8. Generate initial answer
        gen_prompt = f"""Answer this biomedical question using the extracted relationships.

QUESTION: {question}

DEDUPLICATED RELATIONSHIPS (from KG + Text):
{triples_str}

SUPPORTING CONTEXT:
{kg_context[:2000]}
{semantic_context[:1000]}

Provide a comprehensive answer that:
1. Uses the relationships as your primary evidence
2. Explains mechanisms and connections
3. Addresses all parts of the question"""
        
        initial_answer = get_Gemini_response(gen_prompt, system_prompt, temperature=0)
        
        # 9. Self-verification step
        verify_prompt = f"""Review and enhance this answer for completeness.

QUESTION: {question}
CURRENT ANSWER: {initial_answer}
AVAILABLE TRIPLES: {triples_str}

Provide an improved answer that adds any missing key points."""
        
        final_answer = get_Gemini_response(verify_prompt, 
            "Enhance the answer to be maximally comprehensive.", temperature=0)
        
        # 10. Analyze triple usage (requirement a)
        reasoning.triple_usage_order = analyze_triple_usage(final_answer, reasoning.merged_triples)
        
        # 11. Show reasoning chain if enabled (requirement a)
        if SHOW_REASONING_CHAIN:
            reasoning.display()
        
        fused_context = f"KG:\n{kg_context}\n\nSemantic:\n{semantic_context}"
        return final_answer, fused_context, reasoning

# ============== QUESTION GENERATION (requirement d) ==============

def generate_more_questions() -> List[LongFormQuestion]:
    """Generate additional evaluation questions using Gemini."""
    
    gen_prompt = """Generate 5 challenging biomedical long-form questions for evaluating RAG systems.

For each question, provide:
1. A detailed question requiring synthesis of knowledge
2. Question type (explanation, comparison, synthesis, clinical_reasoning, multi_hop)
3. Difficulty (medium, hard, expert)
4. A comprehensive reference answer (200-400 words)
5. 5-7 key points that must be covered
6. 4-6 required entities that should be retrieved

Focus on:
- Gene-disease associations
- Multi-disease comparisons
- Mechanistic explanations
- Clinical reasoning scenarios

Output as JSON array:
```json
[
  {
    "id": "gen_001",
    "question": "...",
    "question_type": "...",
    "difficulty": "...",
    "reference_answer": "...",
    "key_points": ["...", "..."],
    "required_entities": ["...", "..."]
  }
]
```"""
    
    response = get_Gemini_response(gen_prompt, 
        "You are an expert biomedical educator creating evaluation questions.", 
        temperature=0.7)
    
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            questions_data = json.loads(json_match.group(1))
            return [LongFormQuestion(**q) for q in questions_data]
    except Exception as e:
        print(f"Error generating questions: {e}")
    
    return []

# ============== SAMPLE QUESTIONS ==============

LONGFORM_QUESTIONS = [
    LongFormQuestion(
        id="explain_001",
        question="Explain the relationship between HLA-B gene variants and autoimmune diseases like psoriasis. What mechanisms link genetic variation to disease manifestation?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="""HLA-B is part of the Major Histocompatibility Complex (MHC) class I genes, crucial for immune function. The relationship involves: 1) Antigen Presentation - HLA-B presents peptides to CD8+ T cells; certain variants may trigger autoimmune responses. 2) HLA-B*27 - strongly associated with psoriasis and spondyloarthropathies, may present arthritogenic peptides or misfold triggering inflammation. 3) Molecular Mimicry - some variants present microbial peptides resembling self-antigens. 4) NK Cell Interactions - HLA-B interacts with KIRs on NK cells. 5) ER Stress - misfolding can trigger unfolded protein response.""",
        key_points=["HLA-B is MHC class I", "Antigen presentation to T cells", "HLA-B*27 association", 
                   "Molecular mimicry", "NK cell/KIR interactions", "Environmental factors"],
        required_entities=["HLA-B", "psoriasis", "autoimmune", "MHC", "T cells"]
    ),
    LongFormQuestion(
        id="explain_002", 
        question="Describe the pathophysiology of type 2 diabetes and how genetic factors like those associated with NOD2 gene influence disease susceptibility.",
        question_type="explanation",
        difficulty="hard",
        reference_answer="""Type 2 diabetes involves: 1) Insulin Resistance - peripheral tissues become less responsive. 2) Beta Cell Dysfunction - cells compensate then fail. 3) Chronic Inflammation - contributes to both. NOD2 links: 1) Gut-Pancreas Axis - affects barrier function and microbiome. 2) Inflammatory Signaling - activates NF-κB and MAPK. 3) Adipose Function - influences adipokine secretion. 4) Beta Cell Effects - inflammation impacts function.""",
        key_points=["Insulin resistance core feature", "Beta cell dysfunction", "Chronic inflammation",
                   "NOD2 innate immunity", "Gut microbiome influence", "Multiple genetic variants"],
        required_entities=["type 2 diabetes", "NOD2", "insulin resistance", "inflammation", "beta cells"]
    ),
    LongFormQuestion(
        id="compare_001",
        question="Compare the genetic associations between psoriasis and different autoimmune conditions like Takayasu's arteritis, myelodysplastic syndrome, and herpes zoster. What common genetic pathways might explain shared disease susceptibility?",
        question_type="comparison",
        difficulty="hard",
        reference_answer="""Psoriasis has strong HLA-B/HLA-C associations with IL-23/Th17 pathway dysregulation. Comparisons: 1) Takayasu's arteritis - shared HLA associations (HLA-B*52), common MHC-mediated antigen presentation, T cell inflammation. 2) Myelodysplastic syndrome - HLA-B associations, immune dysregulation, self-tolerance mechanisms. 3) Herpes zoster - HLA-B in viral antigen presentation, CD8+ T cell responses. Unifying themes: HLA-B as central hub, T cell dysregulation, shared inflammatory cascades, genetic pleiotropy.""",
        key_points=["HLA-B multiple conditions", "MHC antigen presentation", "T cell dysregulation",
                   "Shared inflammatory pathways", "Genetic pleiotropy", "Unique features per disease"],
        required_entities=["psoriasis", "HLA-B", "Takayasu's arteritis", "myelodysplastic syndrome", "autoimmune"]
    ),
]

# ============== MAIN EVALUATION ==============

def run_enhanced_evaluation(
    generate_new_questions: bool = False,
    use_multiple_judges: bool = True
):
    """Run enhanced evaluation with all new features."""
    
    print("="*70)
    print("ENHANCED LONG-FORM RAG EVALUATION v2")
    print("="*70)
    print(f"Debug Mode: {DEBUG_MODE} (c)")
    print(f"Show Reasoning Chain: {SHOW_REASONING_CHAIN} (a)")
    print(f"Triple Deduplication: Enabled (b)")
    print(f"Enhanced Rubric with Rankings: Enabled (d)")
    print("="*70)
    
    # Load config
    CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
    VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
    NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
    SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
    SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
    SEMANTIC_INDEX_PATH = config_data["SEMANTIC_DB_PATH"]
    SEMANTIC_CHUNKS_PATH = config_data["SEMANTIC_CHUNKS_PATH"]
    SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
    
    print("\n[1/5] Loading models...")
    vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    embedding_function = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
    node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
    semantic_index, semantic_chunks = load_semantic_db(SEMANTIC_INDEX_PATH, SEMANTIC_CHUNKS_PATH)
    semantic_embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    
    config = {
        'context_volume': CONTEXT_VOLUME,
        'similarity_threshold': QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
        'min_similarity': QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY
    }
    
    # Initialize enhanced system
    print("[2/5] Initializing Enhanced SARG+ORAG...")
    sarg_orag = EnhancedSARGORAG(
        vectorstore, embedding_function, node_context_df,
        semantic_index, semantic_chunks, semantic_embed_model, config
    )
    
    # Get questions
    questions = LONGFORM_QUESTIONS.copy()
    if generate_new_questions:
        print("[3/5] Generating additional questions with Gemini...")
        new_qs = generate_more_questions()
        questions.extend(new_qs)
        print(f"  Added {len(new_qs)} generated questions")
    else:
        print("[3/5] Using predefined questions...")
    
    print(f"\n[4/5] Running evaluation on {len(questions)} questions...")
    
    system_prompt = """You are an expert biomedical researcher. Provide comprehensive, 
accurate answers based on the evidence provided."""
    
    all_results = []
    all_reasoning_chains = []
    
    for q_idx, question in enumerate(tqdm(questions, desc="Evaluating")):
        print(f"\n{'='*70}")
        print(f"Question {q_idx+1}/{len(questions)}: {question.id}")
        print(f"{'='*70}")
        
        # Generate answer with reasoning chain
        answer, context, reasoning = sarg_orag.generate(question.question, system_prompt)
        all_reasoning_chains.append(reasoning)
        
        # Judge the answer
        judge_prompt = build_enhanced_judge_prompt(question, answer, context, reasoning)
        judge_response = get_Gemini_response(judge_prompt, 
            "You are an expert evaluator. Be thorough and fair.", temperature=0.1)
        score = parse_enhanced_judge_response(judge_response)
        
        # Store results
        result = {
            "question_id": question.id,
            "question": question.question,
            "question_type": question.question_type,
            "difficulty": question.difficulty,
            "answer": answer[:500],
            "accuracy": score.accuracy,
            "completeness": score.completeness,
            "relevance": score.relevance,
            "coherence": score.coherence,
            "conciseness": score.conciseness,
            "evidence_usage": score.evidence_usage,
            "reasoning_depth": score.reasoning_depth,
            "factual_grounding": score.factual_grounding,
            "total_score": score.total_score,
            "feedback": score.feedback,
            "kg_triples_count": len(reasoning.kg_triples),
            "semantic_triples_count": len(reasoning.semantic_triples),
            "merged_triples_count": len(reasoning.merged_triples),
            "duplicates_removed": len(reasoning.removed_duplicates)
        }
        all_results.append(result)
        
        print(f"\n📊 Scores for {question.id}:")
        print(f"   Accuracy: {score.accuracy}/10 | Completeness: {score.completeness}/10")
        print(f"   Relevance: {score.relevance}/10 | Coherence: {score.coherence}/10")
        print(f"   Conciseness: {score.conciseness}/10 | Evidence Usage: {score.evidence_usage}/10")
        print(f"   Reasoning Depth: {score.reasoning_depth}/10 | Factual Grounding: {score.factual_grounding}/10")
        print(f"   TOTAL: {score.total_score:.2f}/10")
        print(f"   Triples: {len(reasoning.kg_triples)} KG + {len(reasoning.semantic_triples)} Sem = {len(reasoning.merged_triples)} merged ({len(reasoning.removed_duplicates)} duplicates removed)")
    
    # Save results
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(SAVE_PATH, "enhanced_longform_eval_v2.csv")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Questions evaluated: {len(questions)}")
    print(f"\nAverage Scores:")
    for metric in ['accuracy', 'completeness', 'relevance', 'coherence', 
                   'conciseness', 'evidence_usage', 'reasoning_depth', 'factual_grounding']:
        avg = results_df[metric].mean()
        print(f"  {metric.replace('_', ' ').title()}: {avg:.2f}/10")
    print(f"\n  OVERALL AVERAGE: {results_df['total_score'].mean():.2f}/10")
    print(f"\nTriple Statistics:")
    print(f"  Avg KG triples: {results_df['kg_triples_count'].mean():.1f}")
    print(f"  Avg Semantic triples: {results_df['semantic_triples_count'].mean():.1f}")
    print(f"  Avg Merged triples: {results_df['merged_triples_count'].mean():.1f}")
    print(f"  Avg Duplicates removed: {results_df['duplicates_removed'].mean():.1f}")
    print(f"\nResults saved to: {output_path}")
    
    return results_df, all_reasoning_chains

if __name__ == "__main__":
    # Run with all features enabled
    results, chains = run_enhanced_evaluation(
        generate_new_questions=False,  # Set True to generate more questions
        use_multiple_judges=False  # For future: multiple judge models
    )

