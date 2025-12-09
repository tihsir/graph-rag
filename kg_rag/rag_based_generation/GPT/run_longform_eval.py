"""
Long-form QA Evaluation Script

Compares 3 RAG approaches:
1. Vanilla RAG (semantic retrieval only)
2. KG-RAG (knowledge graph retrieval)
3. SARG+ORAG (Mode 5 - hybrid structured reasoning)

Uses LLM-as-a-judge to evaluate answer quality.
"""

import sys
import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from kg_rag.utility import *
from kg_rag.config_loader import *

# ============== DATASET DEFINITIONS ==============

@dataclass
class LongFormQuestion:
    """A long-form question for evaluation."""
    id: str
    question: str
    question_type: str  # explanation, comparison, synthesis, clinical_reasoning, multi_hop
    difficulty: str
    reference_answer: str
    key_points: List[str]
    required_entities: List[str]

@dataclass 
class LongFormScore:
    """Scores for a long-form answer."""
    accuracy: float  # 0-10
    completeness: float  # 0-10
    relevance: float  # 0-10
    coherence: float  # 0-10
    accuracy_feedback: str = ""
    completeness_feedback: str = ""
    relevance_feedback: str = ""
    coherence_feedback: str = ""
    key_points_found: List[str] = None
    key_points_missing: List[str] = None
    overall_feedback: str = ""
    
    def __post_init__(self):
        if self.key_points_found is None:
            self.key_points_found = []
        if self.key_points_missing is None:
            self.key_points_missing = []
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted average score (0-1 scale)."""
        return (self.accuracy * 0.3 + self.completeness * 0.3 + 
                self.relevance * 0.2 + self.coherence * 0.2) / 10
    
    @property
    def total_score(self) -> float:
        """Simple average (0-10 scale)."""
        return (self.accuracy + self.completeness + self.relevance + self.coherence) / 4

# ============== SAMPLE QUESTIONS ==============

LONGFORM_QUESTIONS = [
    LongFormQuestion(
        id="explain_001",
        question="Explain the relationship between HLA-B gene variants and autoimmune diseases like psoriasis. What mechanisms link genetic variation to disease manifestation?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="""HLA-B is part of the Major Histocompatibility Complex (MHC) class I genes, which play crucial roles in immune system function. The relationship between HLA-B variants and autoimmune diseases like psoriasis involves several mechanisms:

1. **Antigen Presentation**: HLA-B molecules present peptide antigens to CD8+ T cells. Certain HLA-B variants may present self-antigens in ways that trigger autoimmune responses.

2. **HLA-B*27 and Related Variants**: Specific alleles like HLA-B*27 are strongly associated with psoriasis and spondyloarthropathies. These variants may present arthritogenic peptides or misfold, triggering inflammatory responses.

3. **Molecular Mimicry**: Some HLA-B variants may present microbial peptides that resemble self-antigens, leading to cross-reactive T cell responses against host tissues.

4. **NK Cell Interactions**: HLA-B molecules interact with killer immunoglobulin-like receptors (KIRs) on NK cells. Certain HLA-B/KIR combinations influence autoimmune susceptibility.

5. **ER Stress Response**: Misfolding of certain HLA-B variants can trigger endoplasmic reticulum stress and inflammatory pathways (unfolded protein response).

The genetic associations are strong but not deterministic - environmental factors and other genetic modifiers influence disease penetrance.""",
        key_points=[
            "HLA-B is part of MHC class I",
            "Involved in antigen presentation to T cells",
            "HLA-B*27 strongly associated with autoimmune diseases",
            "Molecular mimicry may play a role",
            "Interactions with NK cells via KIR receptors",
            "Environmental factors also contribute"
        ],
        required_entities=["HLA-B", "psoriasis", "autoimmune", "MHC", "T cells"]
    ),
    
    LongFormQuestion(
        id="explain_002",
        question="Describe the pathophysiology of type 2 diabetes and how genetic factors like those associated with NOD2 gene influence disease susceptibility.",
        question_type="explanation", 
        difficulty="hard",
        reference_answer="""Type 2 diabetes pathophysiology involves multiple interconnected mechanisms:

**Core Pathophysiology**:
1. **Insulin Resistance**: Peripheral tissues (muscle, fat, liver) become less responsive to insulin, requiring higher insulin levels to maintain glucose homeostasis.

2. **Beta Cell Dysfunction**: Pancreatic beta cells initially compensate by producing more insulin, but eventually fail due to glucotoxicity, lipotoxicity, and ER stress.

3. **Chronic Inflammation**: Low-grade inflammation contributes to both insulin resistance and beta cell dysfunction.

**NOD2 and Genetic Factors**:
NOD2 (Nucleotide-binding oligomerization domain-containing protein 2) is primarily known for innate immunity and inflammatory bowel disease, but emerging evidence links it to metabolic disease:

1. **Gut-Pancreas Axis**: NOD2 variants may affect gut barrier function and microbiome composition, influencing systemic inflammation that contributes to insulin resistance.

2. **Inflammatory Signaling**: NOD2 activates NF-κB and MAPK pathways. Dysregulated NOD2 signaling may promote chronic low-grade inflammation seen in T2D.

3. **Adipose Tissue Function**: NOD2 expression in adipocytes influences adipokine secretion and tissue inflammation.

4. **Beta Cell Effects**: Some studies suggest NOD2-related inflammation directly impacts beta cell function.

The genetic architecture of T2D involves hundreds of genetic variants, each contributing small effects, interacting with environmental factors like diet and physical activity.""",
        key_points=[
            "Insulin resistance is a core feature",
            "Beta cell dysfunction progresses over time",
            "Chronic inflammation contributes to disease",
            "NOD2 is involved in innate immunity",
            "Gut microbiome may influence metabolic health",
            "Multiple genetic variants contribute small effects"
        ],
        required_entities=["type 2 diabetes", "NOD2", "insulin resistance", "inflammation", "beta cells"]
    ),
    
    LongFormQuestion(
        id="compare_001",
        question="Compare the genetic associations between psoriasis and different autoimmune conditions like Takayasu's arteritis, myelodysplastic syndrome, and herpes zoster. What common genetic pathways might explain shared disease susceptibility?",
        question_type="comparison",
        difficulty="hard",
        reference_answer="""**Psoriasis Genetic Basis**:
Psoriasis is strongly associated with HLA-B and HLA-C variants, particularly HLA-Cw6. It involves IL-23/Th17 pathway dysregulation and affects ~2-3% of the population.

**Comparative Analysis**:

1. **Takayasu's Arteritis**:
   - Shared: Strong HLA associations, particularly HLA-B*52
   - Common pathways: MHC-mediated antigen presentation, T cell-mediated inflammation
   - Both involve vascular/tissue inflammation mediated by similar immune mechanisms

2. **Myelodysplastic Syndrome (MDS)**:
   - Shared: HLA-B associations, immune dysregulation
   - Psoriasis patients may have altered hematopoiesis; autoimmune processes in MDS
   - Common pathway: Immune surveillance and self-tolerance mechanisms

3. **Herpes Zoster**:
   - Shared: HLA-B involvement in viral antigen presentation
   - Common pathway: HLA-B restricts CD8+ T cell responses to varicella-zoster virus
   - Psoriasis patients may have altered T cell function affecting viral immunity

**Unifying Themes**:
1. **HLA-B as Central Hub**: HLA-B variants influence antigen presentation across multiple diseases
2. **T Cell Dysregulation**: Altered T cell responses (Th17, CD8+, Tregs) common to all
3. **Inflammatory Cascades**: Shared cytokine networks (TNF-α, IL-17, IFN-γ)
4. **Pleiotropy**: Same genetic variants affect multiple disease phenotypes
5. **Epistasis**: Gene-gene interactions modify disease expression""",
        key_points=[
            "HLA-B is associated with multiple conditions",
            "MHC-mediated antigen presentation is common pathway",
            "T cell dysregulation links diseases",
            "Shared inflammatory pathways (cytokines)",
            "Genetic pleiotropy explains co-associations",
            "Each disease has unique features despite shared genetics"
        ],
        required_entities=["psoriasis", "HLA-B", "Takayasu's arteritis", "myelodysplastic syndrome", "autoimmune"]
    ),
]

# ============== LLM JUDGE ==============

LLM_JUDGE_RUBRIC = """
You are an expert medical evaluator assessing the quality of a RAG system's answer.

## Evaluation Criteria

### 1. Accuracy (0-10)
- Are the medical/scientific facts correct?
- Are there any hallucinations or fabrications?
- Is the information consistent with medical literature?
- Deduct heavily for dangerous misinformation

### 2. Completeness (0-10)
- Does the answer cover all key aspects of the question?
- Are important details included?
- Check against the key points list
- Missing critical information should reduce score significantly

### 3. Relevance (0-10)
- Does the answer directly address the question asked?
- Is extraneous information minimal?
- Is the focus appropriate?

### 4. Coherence (0-10)
- Is the answer well-organized?
- Does it flow logically?
- Is medical/scientific terminology used appropriately?
- Is it understandable to a medical professional?

## Scoring Guidelines
- 9-10: Excellent - Could be from a medical textbook
- 7-8: Good - Minor issues but clinically sound
- 5-6: Adequate - Missing some details or minor errors
- 3-4: Poor - Significant gaps or errors
- 1-2: Very Poor - Major errors or mostly irrelevant
- 0: Completely wrong or harmful
"""

def build_judge_prompt(question: LongFormQuestion, generated_answer: str, retrieved_context: str = None) -> str:
    """Build evaluation prompt for LLM judge."""
    
    prompt = f"""{LLM_JUDGE_RUBRIC}

## Question Being Evaluated
**Question**: {question.question}
**Question Type**: {question.question_type}
**Difficulty**: {question.difficulty}

## Reference Answer (Gold Standard)
{question.reference_answer}

## Key Points That Should Be Covered
{chr(10).join(f'- {point}' for point in question.key_points)}

## Required Entities/Concepts
{', '.join(question.required_entities)}

## Generated Answer to Evaluate
{generated_answer}
"""
    
    if retrieved_context:
        prompt += f"""
## Retrieved Context Used
{retrieved_context[:3000]}...
"""
    
    prompt += """
## Your Evaluation Task

Evaluate the generated answer and provide scores in JSON format:

```json
{
    "accuracy": <0-10>,
    "accuracy_feedback": "<brief explanation>",
    "completeness": <0-10>,
    "completeness_feedback": "<brief explanation>",
    "key_points_found": ["<point1>", "<point2>"],
    "key_points_missing": ["<point1>", "<point2>"],
    "relevance": <0-10>,
    "relevance_feedback": "<brief explanation>",
    "coherence": <0-10>,
    "coherence_feedback": "<brief explanation>",
    "overall_feedback": "<2-3 sentence summary>"
}
```
"""
    return prompt

def parse_judge_response(response: str, key_points: List[str]) -> LongFormScore:
    """Parse LLM judge response into score object."""
    import re
    
    try:
        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
        
        data = json.loads(json_str)
        
        return LongFormScore(
            accuracy=float(data.get('accuracy', 5)),
            completeness=float(data.get('completeness', 5)),
            relevance=float(data.get('relevance', 5)),
            coherence=float(data.get('coherence', 5)),
            accuracy_feedback=data.get('accuracy_feedback', ''),
            completeness_feedback=data.get('completeness_feedback', ''),
            relevance_feedback=data.get('relevance_feedback', ''),
            coherence_feedback=data.get('coherence_feedback', ''),
            key_points_found=data.get('key_points_found', []),
            key_points_missing=data.get('key_points_missing', []),
            overall_feedback=data.get('overall_feedback', '')
        )
    except Exception as e:
        print(f"Error parsing judge response: {e}")
        # Fallback with default scores
        return LongFormScore(
            accuracy=5.0, completeness=5.0, relevance=5.0, coherence=5.0,
            overall_feedback=f"Parse error: {str(e)[:100]}"
        )

# ============== RAG SYSTEMS ==============

class VanillaRAG:
    """Simple semantic retrieval RAG."""
    
    def __init__(self, semantic_index, semantic_chunks, embed_model):
        self.index = semantic_index
        self.chunks = semantic_chunks
        self.embed_model = embed_model
    
    def generate(self, question: str, system_prompt: str) -> Tuple[str, str]:
        """Generate answer using vanilla semantic RAG."""
        # Retrieve
        context = semantic_retrieve(question, self.index, self.chunks, self.embed_model, k=5)
        
        # Generate
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed, comprehensive answer based on the context and your knowledge."
        answer = get_Gemini_response(prompt, system_prompt, temperature=0)
        
        return answer, context

class KGRAG:
    """Knowledge Graph based RAG (original KG-RAG)."""
    
    def __init__(self, vectorstore, embedding_function, node_context_df, config):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.node_context_df = node_context_df
        self.config = config
    
    def generate(self, question: str, system_prompt: str) -> Tuple[str, str]:
        """Generate answer using KG-RAG."""
        # Retrieve from KG
        context = retrieve_context(
            question,
            self.vectorstore,
            self.embedding_function,
            self.node_context_df,
            self.config['context_volume'],
            self.config['similarity_threshold'],
            self.config['min_similarity'],
            edge_evidence=False,
            model_id="gemini-2.0-flash"
        )
        
        # Generate
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed, comprehensive answer based on the context and your knowledge."
        answer = get_Gemini_response(prompt, system_prompt, temperature=0)
        
        return answer, context

class SARGORAG:
    """SARG + ORAG hybrid system (Mode 5) - With Self-Verification."""
    
    def __init__(self, vectorstore, embedding_function, node_context_df, 
                 semantic_index, semantic_chunks, semantic_embed_model, config):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.node_context_df = node_context_df
        self.semantic_index = semantic_index
        self.semantic_chunks = semantic_chunks
        self.semantic_embed_model = semantic_embed_model
        self.config = config
    
    def generate(self, question: str, system_prompt: str) -> Tuple[str, str]:
        """Generate answer using SARG+ORAG with self-verification for completeness."""
        
        # 1. KG retrieval (ORAG) - Primary source
        kg_context = retrieve_context(
            question,
            self.vectorstore,
            self.embedding_function,
            self.node_context_df,
            self.config['context_volume'],
            self.config['similarity_threshold'],
            self.config['min_similarity'],
            edge_evidence=False,
            model_id="gemini-2.0-flash"
        )
        
        # 2. Semantic retrieval
        semantic_context = semantic_retrieve(
            question,
            self.semantic_index,
            self.semantic_chunks,
            self.semantic_embed_model,
            k=5
        )
        
        # 3. Fuse contexts
        fused_context = f"""=== STRUCTURED KNOWLEDGE GRAPH DATA ===
{kg_context}

=== SUPPORTING LITERATURE ===
{semantic_context}"""
        
        # 4. Extract key relationships (SARG)
        triple_prompt = """Analyze this biomedical context and extract the most important relationships.

For each relationship, format as:
• [Entity1] --[Relationship]--> [Entity2]: Brief explanation

Types to extract:
- GENE_ASSOCIATES_DISEASE
- GENE_SHARED_BY (genes linked to multiple diseases)
- MECHANISM (how the relationship works)
- PATHWAY_INVOLVEMENT

Extract 5-10 of the most relevant relationships."""
        
        triples = get_Gemini_response(fused_context[:4000], triple_prompt, temperature=0)
        
        # 5. Initial answer generation
        gen_prompt = f"""You are a world-class biomedical expert. Answer this question comprehensively.

QUESTION: {question}

EXTRACTED RELATIONSHIPS:
{triples}

CONTEXT:
{fused_context}

Requirements:
- Cover ALL aspects of the question
- Explain mechanisms in detail
- Use specific gene-disease relationships from the data
- For comparisons, address each item systematically
- Structure with clear headers"""
        
        initial_answer = get_Gemini_response(gen_prompt, "You are an expert biomedical researcher. Be thorough and comprehensive.", temperature=0)
        
        # 6. SELF-VERIFICATION: Check completeness and enhance
        verify_prompt = f"""Review this answer for completeness and accuracy.

ORIGINAL QUESTION: {question}

ANSWER TO REVIEW:
{initial_answer}

AVAILABLE EVIDENCE:
{triples}

CHECKLIST - Verify the answer covers:
1. All entities mentioned in the question
2. Specific gene-disease relationships
3. Mechanistic explanations (HOW relationships work)
4. For comparisons: each item discussed individually AND similarities/differences
5. Relevant pathways and biological processes

OUTPUT: Provide an IMPROVED answer that:
- Adds any missing key points
- Includes more specific details from the evidence
- Better explains mechanisms
- Is more comprehensive overall

Start directly with the improved answer (no preamble about what you're doing)."""
        
        # Generate enhanced answer
        enhanced_answer = get_Gemini_response(
            verify_prompt, 
            "You are a meticulous biomedical reviewer. Enhance the answer to be maximally comprehensive, accurate, and well-structured. Add missing details from the evidence.",
            temperature=0
        )
        
        return enhanced_answer, fused_context

# ============== MAIN EVALUATION ==============

def run_evaluation():
    """Run the full evaluation comparing 3 RAG systems."""
    
    print("=" * 60)
    print("LONG-FORM RAG EVALUATION")
    print("Comparing: Vanilla RAG vs KG-RAG vs SARG+ORAG")
    print("=" * 60)
    
    # Load configuration
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
    
    print("\n[1/4] Loading models and indexes...")
    
    # Load KG components
    vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
    embedding_function = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
    node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
    
    # Load semantic components
    semantic_index, semantic_chunks = load_semantic_db(SEMANTIC_INDEX_PATH, SEMANTIC_CHUNKS_PATH)
    semantic_embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    
    config = {
        'context_volume': CONTEXT_VOLUME,
        'similarity_threshold': QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
        'min_similarity': QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY
    }
    
    # Initialize RAG systems
    print("[2/4] Initializing RAG systems...")
    vanilla_rag = VanillaRAG(semantic_index, semantic_chunks, semantic_embed_model)
    kg_rag = KGRAG(vectorstore, embedding_function, node_context_df, config)
    sarg_orag = SARGORAG(vectorstore, embedding_function, node_context_df,
                         semantic_index, semantic_chunks, semantic_embed_model, config)
    
    systems = {
        "Vanilla_RAG": vanilla_rag,
        "KG_RAG": kg_rag,
        "SARG_ORAG": sarg_orag
    }
    
    system_prompt = """You are an expert biomedical researcher. Provide comprehensive, accurate answers 
to medical and scientific questions. Include relevant mechanisms, evidence, and clinical implications 
where appropriate. Structure your answer clearly with key points."""
    
    judge_system_prompt = """You are an expert medical evaluator. Assess answer quality fairly and thoroughly.
Provide detailed, constructive feedback."""
    
    # Run evaluation
    print(f"[3/4] Running evaluation on {len(LONGFORM_QUESTIONS)} questions...")
    
    all_results = []
    
    for q_idx, question in enumerate(tqdm(LONGFORM_QUESTIONS, desc="Questions")):
        print(f"\n--- Question {q_idx + 1}: {question.id} ---")
        print(f"Q: {question.question[:100]}...")
        
        question_results = {
            "question_id": question.id,
            "question": question.question,
            "question_type": question.question_type,
            "difficulty": question.difficulty
        }
        
        for sys_name, rag_system in systems.items():
            print(f"  Running {sys_name}...")
            
            try:
                # Generate answer
                start_time = time.time()
                answer, context = rag_system.generate(question.question, system_prompt)
                gen_time = time.time() - start_time
                
                # Judge the answer
                judge_prompt = build_judge_prompt(question, answer, context)
                judge_response = get_Gemini_response(judge_prompt, judge_system_prompt, temperature=0.1)
                score = parse_judge_response(judge_response, question.key_points)
                
                # Store results
                question_results[f"{sys_name}_answer"] = answer
                question_results[f"{sys_name}_context"] = context[:1000]  # Truncate for storage
                question_results[f"{sys_name}_accuracy"] = score.accuracy
                question_results[f"{sys_name}_completeness"] = score.completeness
                question_results[f"{sys_name}_relevance"] = score.relevance
                question_results[f"{sys_name}_coherence"] = score.coherence
                question_results[f"{sys_name}_total"] = score.total_score
                question_results[f"{sys_name}_weighted"] = score.weighted_score
                question_results[f"{sys_name}_feedback"] = score.overall_feedback
                question_results[f"{sys_name}_time"] = gen_time
                
                print(f"    Score: {score.total_score:.1f}/10 ({gen_time:.1f}s)")
                
            except Exception as e:
                print(f"    ERROR: {str(e)[:100]}")
                question_results[f"{sys_name}_error"] = str(e)
                question_results[f"{sys_name}_total"] = 0
        
        all_results.append(question_results)
    
    # Save results
    print("\n[4/4] Saving results...")
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(SAVE_PATH, "longform_eval_results.csv")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for sys_name in systems.keys():
        total_col = f"{sys_name}_total"
        if total_col in results_df.columns:
            avg_score = results_df[total_col].mean()
            print(f"{sys_name}: {avg_score:.2f}/10")
    
    print(f"\nResults saved to: {output_path}")
    
    return results_df

if __name__ == "__main__":
    run_evaluation()

