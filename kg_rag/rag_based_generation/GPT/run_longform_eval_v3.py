"""
Long-form QA Evaluation Script v3

Improvements:
1. LLM-based triple deduplication (not string matching)
2. Self-verification for ALL systems (KG, Vanilla, SARG+ORAG)
3. Two triple extraction modes:
   - Fused context extraction
   - Separate extraction then merge
4. LLM-as-judge for identifying duplicate/similar triples

"""

import sys
import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from kg_rag.utility import *
from kg_rag.config_loader import *

# ============== CONFIGURATION ==============
DEBUG_MODE = True
SHOW_REASONING_CHAIN = True
EXTRACTION_MODE = "separate"  # "fused" or "separate"

# ============== DATA CLASSES ==============

@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source: str  # 'kg', 'semantic', or 'fused'
    confidence: float = 1.0
    
    def to_string(self):
        return f"({self.subject}) --[{self.relation}]--> ({self.object})"
    
    def to_dict(self):
        return {"subject": self.subject, "relation": self.relation, 
                "object": self.object, "source": self.source}

@dataclass
class LongFormQuestion:
    id: str
    question: str
    question_type: str
    difficulty: str
    reference_answer: str
    key_points: List[str]
    required_entities: List[str]

@dataclass
class EvalScore:
    accuracy: float = 5.0
    completeness: float = 5.0
    relevance: float = 5.0
    coherence: float = 5.0
    feedback: str = ""
    
    @property
    def total(self) -> float:
        return (self.accuracy + self.completeness + self.relevance + self.coherence) / 4

# ============== LLM-BASED TRIPLE OPERATIONS ==============

def extract_triples_llm(context: str, source: str) -> List[Triple]:
    """Extract triples using LLM."""
    
    prompt = f"""Extract biomedical relationship triples from this context.

CONTEXT:
{context[:4000]}

OUTPUT FORMAT - Return as JSON array:
```json
[
  {{"subject": "Entity1", "relation": "RELATION_TYPE", "object": "Entity2"}},
  {{"subject": "Entity2", "relation": "RELATION_TYPE", "object": "Entity3"}}
]
```

RELATION TYPES to use:
- ASSOCIATES_WITH (gene-disease associations)
- TREATS (drug-disease)
- CAUSES (causal relationships)
- ISA (hierarchical/type relationships)
- PRESENTS (disease-symptom)
- REGULATES (gene regulation)

Extract ALL relevant triples. Be comprehensive."""

    response = get_Gemini_response(prompt, "Extract biomedical relationships as JSON.", temperature=0)
    
    triples = []
    try:
        # Find JSON in response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            # Try parsing as raw JSON array
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
            else:
                data = []
        
        for item in data:
            triples.append(Triple(
                subject=item.get('subject', ''),
                relation=item.get('relation', 'ASSOCIATES_WITH'),
                object=item.get('object', ''),
                source=source
            ))
    except Exception as e:
        print(f"  Triple extraction error: {e}")
    
    return triples

def deduplicate_triples_llm(kg_triples: List[Triple], semantic_triples: List[Triple]) -> Tuple[List[Triple], List[str], str]:
    """
    Use LLM to identify duplicate/similar triples and merge intelligently.
    Returns: (merged_triples, duplicates_found, llm_reasoning)
    """
    
    # Limit to avoid token limits
    kg_str = "\n".join([f"KG-{i+1}: {t.to_string()}" for i, t in enumerate(kg_triples[:20])])
    sem_str = "\n".join([f"SEM-{i+1}: {t.to_string()}" for i, t in enumerate(semantic_triples[:20])])
    
    prompt = f"""Analyze these biomedical triples and identify DUPLICATES (same relationship, different wording).

KG TRIPLES:
{kg_str}

SEMANTIC TRIPLES:
{sem_str}

Find pairs where KG and SEM express the SAME relationship (same entities, same meaning).
Consider: "Disease X associates Gene Y" = "Gene Y associates Disease X" as duplicates.

Output JSON:
```json
{{
  "duplicates": [
    {{"kg": "KG-1", "sem": "SEM-3", "reason": "why they are the same"}}
  ],
  "reasoning": "summary of analysis"
}}
```

Only list actual duplicates. If none found, return empty duplicates array."""

    response = get_Gemini_response(prompt, "Identify duplicate triples.", temperature=0)
    
    duplicates_found = []
    reasoning = ""
    
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            reasoning = data.get('reasoning', '')
            
            # Track duplicates
            for dup in data.get('duplicates', []):
                kg_id = dup.get('kg', dup.get('kg_id', ''))
                sem_id = dup.get('sem', dup.get('sem_id', ''))
                reason = dup.get('reason', '')
                duplicates_found.append(f"{kg_id} = {sem_id}: {reason}")
                
    except Exception as e:
        print(f"  Deduplication parse error: {e}")
        reasoning = f"Parse error: {e}"
    
    # Build merged list: all KG + semantic triples that aren't duplicates
    dup_sem_ids = set()
    for d in duplicates_found:
        # Extract SEM-X from the duplicate string
        match = re.search(r'SEM-(\d+)', d)
        if match:
            dup_sem_ids.add(int(match.group(1)) - 1)  # 0-indexed
    
    merged = list(kg_triples)  # All KG triples
    for i, t in enumerate(semantic_triples):
        if i not in dup_sem_ids:
            merged.append(t)
    
    return merged, duplicates_found, reasoning

def extract_triples_fused(kg_context: str, semantic_context: str) -> List[Triple]:
    """Extract triples from fused context (single pass)."""
    fused = f"""=== KNOWLEDGE GRAPH DATA ===
{kg_context[:2500]}

=== SEMANTIC/LITERATURE DATA ===
{semantic_context[:2500]}"""
    
    return extract_triples_llm(fused, "fused")

def extract_triples_separate(kg_context: str, semantic_context: str) -> Tuple[List[Triple], List[Triple]]:
    """Extract triples separately from each source."""
    kg_triples = extract_triples_llm(kg_context, "kg")
    semantic_triples = extract_triples_llm(semantic_context, "semantic")
    return kg_triples, semantic_triples

# ============== SELF-VERIFICATION (for all systems) ==============

def self_verify_answer(question: str, initial_answer: str, context: str, triples_str: str = "") -> str:
    """Self-verification step to enhance answer completeness."""
    
    verify_prompt = f"""Review this answer for completeness and accuracy.

QUESTION: {question}

INITIAL ANSWER:
{initial_answer}

AVAILABLE CONTEXT:
{context[:3000]}

{"EXTRACTED RELATIONSHIPS:" + chr(10) + triples_str if triples_str else ""}

TASK: Provide an IMPROVED answer that:
1. Adds any missing key information from the context
2. Corrects any inaccuracies
3. Improves structure and clarity
4. Ensures ALL parts of the question are addressed

Output the improved answer directly (no preamble)."""

    enhanced = get_Gemini_response(verify_prompt, 
        "You are a meticulous biomedical reviewer. Enhance answers to be comprehensive and accurate.",
        temperature=0)
    
    return enhanced

# ============== RAG SYSTEMS (all with self-verification option) ==============

class VanillaRAG:
    """Semantic retrieval RAG with optional self-verification."""
    
    def __init__(self, semantic_index, semantic_chunks, embed_model):
        self.index = semantic_index
        self.chunks = semantic_chunks
        self.embed_model = embed_model
    
    def generate(self, question: str, system_prompt: str, use_self_verify: bool = True) -> Tuple[str, str, dict]:
        """Generate with optional self-verification."""
        
        # Retrieve
        context = semantic_retrieve(question, self.index, self.chunks, self.embed_model, k=5)
        
        if DEBUG_MODE:
            print("\n📂 VANILLA RAG - Semantic Context:")
            print("-" * 50)
            print(context[:1000] + "..." if len(context) > 1000 else context)
        
        # Initial generation
        prompt = f"""Context:
{context}

Question: {question}

Provide a detailed, comprehensive answer based on the context."""
        
        initial_answer = get_Gemini_response(prompt, system_prompt, temperature=0)
        
        # Self-verification (NEW for Vanilla)
        if use_self_verify:
            final_answer = self_verify_answer(question, initial_answer, context)
        else:
            final_answer = initial_answer
        
        metadata = {
            "self_verified": use_self_verify,
            "context_length": len(context)
        }
        
        return final_answer, context, metadata

class KGRAG:
    """Knowledge Graph RAG with optional self-verification."""
    
    def __init__(self, vectorstore, embedding_function, node_context_df, config):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.node_context_df = node_context_df
        self.config = config
    
    def generate(self, question: str, system_prompt: str, use_self_verify: bool = True) -> Tuple[str, str, dict]:
        """Generate with optional self-verification."""
        
        # KG Retrieval
        context = retrieve_context(
            question, self.vectorstore, self.embedding_function,
            self.node_context_df, self.config['context_volume'],
            self.config['similarity_threshold'], self.config['min_similarity'],
            edge_evidence=False, model_id="gemini-2.0-flash"
        )
        
        if DEBUG_MODE:
            print("\n📂 KG-RAG - Knowledge Graph Context:")
            print("-" * 50)
            print(context[:1000] + "..." if len(context) > 1000 else context)
        
        # Initial generation
        prompt = f"""Context:
{context}

Question: {question}

Provide a detailed, comprehensive answer based on the context."""
        
        initial_answer = get_Gemini_response(prompt, system_prompt, temperature=0)
        
        # Self-verification (NEW for KG-RAG)
        if use_self_verify:
            final_answer = self_verify_answer(question, initial_answer, context)
        else:
            final_answer = initial_answer
        
        metadata = {
            "self_verified": use_self_verify,
            "context_length": len(context)
        }
        
        return final_answer, context, metadata

class SARGORAG:
    """SARG+ORAG with LLM-based deduplication and self-verification."""
    
    def __init__(self, vectorstore, embedding_function, node_context_df,
                 semantic_index, semantic_chunks, semantic_embed_model, config):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.node_context_df = node_context_df
        self.semantic_index = semantic_index
        self.semantic_chunks = semantic_chunks
        self.semantic_embed_model = semantic_embed_model
        self.config = config
    
    def generate(self, question: str, system_prompt: str, 
                 extraction_mode: str = "separate") -> Tuple[str, str, dict]:
        """
        Generate with triple extraction and LLM-based deduplication.
        
        extraction_mode: "fused" or "separate"
        """
        
        # 1. KG Retrieval
        kg_context = retrieve_context(
            question, self.vectorstore, self.embedding_function,
            self.node_context_df, self.config['context_volume'],
            self.config['similarity_threshold'], self.config['min_similarity'],
            edge_evidence=False, model_id="gemini-2.0-flash"
        )
        
        # 2. Semantic Retrieval
        semantic_context = semantic_retrieve(
            question, self.semantic_index, self.semantic_chunks,
            self.semantic_embed_model, k=5
        )
        
        if DEBUG_MODE:
            print("\n📂 SARG+ORAG - KG Context:")
            print("-" * 50)
            print(kg_context[:800] + "..." if len(kg_context) > 800 else kg_context)
            print("\n📂 SARG+ORAG - Semantic Context:")
            print("-" * 50)
            print(semantic_context[:800] + "..." if len(semantic_context) > 800 else semantic_context)
        
        # 3. Triple Extraction (two modes)
        if extraction_mode == "fused":
            # Mode 1: Extract from fused context
            print("  📊 Triple extraction mode: FUSED")
            all_triples = extract_triples_fused(kg_context, semantic_context)
            kg_triples = [t for t in all_triples if 'kg' in t.source.lower()]
            semantic_triples = [t for t in all_triples if 'semantic' in t.source.lower() or 'sem' in t.source.lower()]
            merged_triples = all_triples
            duplicates = []
            dedup_reasoning = "Fused extraction - no separate deduplication needed"
        else:
            # Mode 2: Extract separately then merge with LLM deduplication
            print("  📊 Triple extraction mode: SEPARATE + LLM DEDUP")
            kg_triples, semantic_triples = extract_triples_separate(kg_context, semantic_context)
            
            # LLM-based deduplication
            merged_triples, duplicates, dedup_reasoning = deduplicate_triples_llm(
                kg_triples, semantic_triples
            )
        
        if SHOW_REASONING_CHAIN:
            print("\n" + "="*60)
            print("🔍 TRIPLE ANALYSIS")
            print("="*60)
            print(f"📊 KG Triples: {len(kg_triples)}")
            for t in kg_triples[:5]:
                print(f"   • {t.to_string()}")
            if len(kg_triples) > 5:
                print(f"   ... and {len(kg_triples)-5} more")
            
            print(f"\n📚 Semantic Triples: {len(semantic_triples)}")
            for t in semantic_triples[:5]:
                print(f"   • {t.to_string()}")
            if len(semantic_triples) > 5:
                print(f"   ... and {len(semantic_triples)-5} more")
            
            print(f"\n🔄 Duplicates Identified: {len(duplicates)}")
            for d in duplicates[:3]:
                print(f"   • {d}")
            
            print(f"\n✅ Final Merged: {len(merged_triples)} triples")
            print(f"\n💭 LLM Dedup Reasoning: {dedup_reasoning[:200]}...")
            print("="*60)
        
        # 4. Build triples string for prompt
        triples_str = "\n".join([f"• {t.to_string()}" for t in merged_triples[:25]])
        
        # 5. Fused context
        fused_context = f"""=== KNOWLEDGE GRAPH ===
{kg_context}

=== SEMANTIC ===
{semantic_context}"""
        
        # 6. Enhanced multi-step reasoning
        # Step 6a: Analyze the question to identify what needs to be addressed
        analyze_prompt = f"""Analyze this biomedical question and identify:
1. Main entities/topics to discuss
2. Type of answer needed (explanation, comparison, mechanism, etc.)
3. Key aspects that MUST be covered

Question: {question}

Output a brief analysis."""
        
        question_analysis = get_Gemini_response(analyze_prompt, "Analyze the question structure.", temperature=0)
        
        # Step 6b: Generate comprehensive answer using triples + analysis
        gen_prompt = f"""You are a world-renowned biomedical expert. Answer this question with exceptional depth and accuracy.

QUESTION: {question}

QUESTION ANALYSIS:
{question_analysis}

EXTRACTED KNOWLEDGE RELATIONSHIPS:
{triples_str}

RAW EVIDENCE:
{fused_context[:5000]}

CRITICAL INSTRUCTIONS:
1. Address EVERY aspect identified in the question analysis
2. For COMPARISON questions: systematically compare EACH entity mentioned
3. Use specific relationships from the extracted knowledge as evidence
4. Explain underlying MECHANISMS, not just associations
5. Connect concepts through multi-hop reasoning (A→B→C)
6. Structure your answer with clear headers for each major point
7. Be COMPREHENSIVE - missing key points will result in a lower score

Provide a detailed, well-structured answer:"""
        
        initial_answer = get_Gemini_response(gen_prompt, 
            "You are the world's leading expert in biomedical science. Your answers are exceptionally thorough, accurate, and well-organized.",
            temperature=0)
        
        # 7. Enhanced self-verification with checklist
        verify_prompt = f"""You are a senior reviewer. Check this answer against the question requirements.

ORIGINAL QUESTION: {question}

QUESTION ANALYSIS: {question_analysis}

CURRENT ANSWER:
{initial_answer}

AVAILABLE EVIDENCE:
{triples_str}

VERIFICATION CHECKLIST:
□ Does it address ALL entities mentioned in the question?
□ For comparisons: is EACH item discussed separately AND compared?
□ Are mechanisms explained (not just associations stated)?
□ Is evidence from the relationships cited?
□ Is the answer well-structured with clear sections?
□ Are there any gaps or missing key information?

If ANY checkbox fails, REWRITE the answer to fix it.
If all checkboxes pass, enhance the answer with any additional relevant details.

Output the FINAL improved answer directly:"""

        final_answer = get_Gemini_response(verify_prompt,
            "You are a meticulous scientific reviewer. Ensure completeness and accuracy.",
            temperature=0)
        
        metadata = {
            "extraction_mode": extraction_mode,
            "kg_triples": len(kg_triples),
            "semantic_triples": len(semantic_triples),
            "merged_triples": len(merged_triples),
            "duplicates_found": len(duplicates),
            "dedup_reasoning": dedup_reasoning,
            "self_verified": True
        }
        
        return final_answer, fused_context, metadata

# ============== JUDGE ==============

def judge_answer(question: LongFormQuestion, answer: str) -> EvalScore:
    """LLM judge for answer evaluation."""
    
    prompt = f"""Evaluate this biomedical answer.

QUESTION: {question.question}
DIFFICULTY: {question.difficulty}

REFERENCE ANSWER:
{question.reference_answer}

KEY POINTS TO COVER:
{chr(10).join(f'• {p}' for p in question.key_points)}

GENERATED ANSWER:
{answer}

Score 0-10 on:
1. Accuracy - Medical facts correct?
2. Completeness - Covers all key points?
3. Relevance - Addresses the question?
4. Coherence - Well-organized?

Output as JSON:
```json
{{
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "relevance": <0-10>,
  "coherence": <0-10>,
  "feedback": "<brief assessment>"
}}
```"""

    response = get_Gemini_response(prompt, "Evaluate fairly and thoroughly.", temperature=0.1)
    
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return EvalScore(
                accuracy=float(data.get('accuracy', 5)),
                completeness=float(data.get('completeness', 5)),
                relevance=float(data.get('relevance', 5)),
                coherence=float(data.get('coherence', 5)),
                feedback=data.get('feedback', '')
            )
    except:
        pass
    return EvalScore(feedback="Parse error")

# ============== QUESTIONS ==============

QUESTIONS = [
    # Original 3 questions
    LongFormQuestion(
        id="q1_hla_psoriasis",
        question="Explain the relationship between HLA-B gene variants and autoimmune diseases like psoriasis. What mechanisms link genetic variation to disease manifestation?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="HLA-B is MHC class I, presents antigens to CD8+ T cells. HLA-B*27 strongly associated with psoriasis. Mechanisms include molecular mimicry, NK cell interactions via KIRs, and ER stress from misfolding.",
        key_points=["HLA-B is MHC class I", "Antigen presentation", "HLA-B*27 association", "Molecular mimicry", "NK cell interactions"],
        required_entities=["HLA-B", "psoriasis", "autoimmune", "T cells"]
    ),
    LongFormQuestion(
        id="q2_diabetes_nod2",
        question="Describe the pathophysiology of type 2 diabetes and how genetic factors like NOD2 influence disease susceptibility.",
        question_type="explanation",
        difficulty="hard",
        reference_answer="T2D involves insulin resistance and beta cell dysfunction. NOD2 affects gut barrier, microbiome, and inflammatory signaling via NF-κB. Chronic inflammation links NOD2 to metabolic dysfunction.",
        key_points=["Insulin resistance", "Beta cell dysfunction", "NOD2 innate immunity", "Gut microbiome", "Inflammation"],
        required_entities=["type 2 diabetes", "NOD2", "insulin resistance", "inflammation"]
    ),
    LongFormQuestion(
        id="q3_multi_disease",
        question="Compare genetic associations between psoriasis and Takayasu's arteritis, myelodysplastic syndrome, and herpes zoster. What common pathways explain shared susceptibility?",
        question_type="comparison",
        difficulty="hard",
        reference_answer="HLA-B is central hub for all conditions. Shared pathways include MHC antigen presentation, T cell dysregulation, inflammatory cascades. Genetic pleiotropy explains co-associations.",
        key_points=["HLA-B multiple conditions", "MHC antigen presentation", "T cell dysregulation", "Shared inflammation", "Genetic pleiotropy"],
        required_entities=["psoriasis", "HLA-B", "Takayasu's arteritis", "autoimmune"]
    ),
    
    # NEW QUESTIONS (4-20)
    LongFormQuestion(
        id="q4_brca_cancer",
        question="Explain how BRCA1 and BRCA2 gene mutations increase cancer risk. What are the molecular mechanisms and which cancers are most affected?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="BRCA1/2 are tumor suppressors involved in DNA double-strand break repair via homologous recombination. Mutations impair DNA repair, leading to genomic instability. Primarily increases breast and ovarian cancer risk, also pancreatic and prostate.",
        key_points=["Tumor suppressor genes", "DNA repair mechanisms", "Homologous recombination", "Breast cancer risk", "Ovarian cancer", "Genomic instability"],
        required_entities=["BRCA1", "BRCA2", "breast cancer", "DNA repair"]
    ),
    LongFormQuestion(
        id="q5_alzheimer_apoe",
        question="Describe the role of APOE gene variants in Alzheimer's disease susceptibility. How does APOE4 differ from APOE2 and APOE3?",
        question_type="comparison",
        difficulty="hard",
        reference_answer="APOE4 is the strongest genetic risk factor for late-onset Alzheimer's. It impairs amyloid-beta clearance, promotes tau pathology, and affects lipid metabolism. APOE2 is protective, APOE3 is neutral. APOE4 carriers have 3-15x increased risk.",
        key_points=["APOE4 risk factor", "Amyloid-beta clearance", "Tau pathology", "APOE2 protective", "Lipid metabolism", "Risk stratification"],
        required_entities=["APOE", "Alzheimer's disease", "amyloid", "dementia"]
    ),
    LongFormQuestion(
        id="q6_cystic_fibrosis",
        question="Explain the molecular basis of cystic fibrosis and how CFTR gene mutations lead to disease manifestations in different organs.",
        question_type="explanation",
        difficulty="medium",
        reference_answer="CFTR encodes a chloride channel. Mutations (most common F508del) cause protein misfolding or dysfunction. This leads to thick mucus in lungs, pancreatic insufficiency, elevated sweat chloride, and male infertility.",
        key_points=["CFTR chloride channel", "F508del mutation", "Protein misfolding", "Lung disease", "Pancreatic insufficiency", "Multi-organ effects"],
        required_entities=["CFTR", "cystic fibrosis", "chloride channel", "lung disease"]
    ),
    LongFormQuestion(
        id="q7_parkinsons_genetics",
        question="What genetic factors contribute to Parkinson's disease? Compare the roles of SNCA, LRRK2, and PARK2 genes.",
        question_type="comparison",
        difficulty="hard",
        reference_answer="SNCA encodes alpha-synuclein, mutations cause protein aggregation. LRRK2 is most common genetic cause, affects kinase activity. PARK2 (Parkin) causes early-onset PD through impaired mitophagy. All converge on dopaminergic neuron dysfunction.",
        key_points=["SNCA alpha-synuclein", "LRRK2 mutations", "PARK2 early-onset", "Protein aggregation", "Mitophagy", "Dopamine neurons"],
        required_entities=["Parkinson's disease", "SNCA", "LRRK2", "dopamine"]
    ),
    LongFormQuestion(
        id="q8_rheumatoid_arthritis",
        question="Describe the genetic and immunological basis of rheumatoid arthritis. How do HLA-DRB1 variants contribute to disease?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="RA is autoimmune disease attacking joints. HLA-DRB1 shared epitope increases risk 5-fold. Involves citrullinated protein antibodies, TNF-alpha, IL-6 inflammation, and synovial destruction.",
        key_points=["HLA-DRB1 shared epitope", "Autoantibodies", "Citrullinated proteins", "TNF-alpha", "IL-6", "Joint destruction"],
        required_entities=["rheumatoid arthritis", "HLA-DRB1", "autoimmune", "TNF"]
    ),
    LongFormQuestion(
        id="q9_sickle_cell",
        question="Explain the molecular mechanism of sickle cell disease. How does a single nucleotide change lead to the disease phenotype?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="Single nucleotide mutation in HBB gene (Glu6Val) causes hemoglobin S. Under low oxygen, HbS polymerizes causing RBC sickling. Leads to vaso-occlusion, hemolysis, organ damage. Heterozygotes have malaria protection.",
        key_points=["HBB gene mutation", "Hemoglobin S", "RBC sickling", "Vaso-occlusion", "Hemolysis", "Malaria protection"],
        required_entities=["sickle cell", "hemoglobin", "HBB", "anemia"]
    ),
    LongFormQuestion(
        id="q10_schizophrenia",
        question="What is known about the genetic architecture of schizophrenia? Discuss both common variants and rare mutations.",
        question_type="synthesis",
        difficulty="hard",
        reference_answer="Schizophrenia is highly polygenic with hundreds of common variants of small effect. Key genes include DRD2, COMT, DISC1. Rare CNVs (22q11 deletion) confer high risk. Involves dopamine, glutamate systems, and synaptic function.",
        key_points=["Polygenic architecture", "Common variants", "Rare CNVs", "DRD2 dopamine", "22q11 deletion", "Synaptic genes"],
        required_entities=["schizophrenia", "DRD2", "COMT", "psychiatric"]
    ),
    LongFormQuestion(
        id="q11_inflammatory_bowel",
        question="Compare the genetic factors involved in Crohn's disease versus ulcerative colitis. What pathways are shared and what differs?",
        question_type="comparison",
        difficulty="hard",
        reference_answer="Both involve NOD2, IL23R, ATG16L1. Crohn's has stronger NOD2 association and autophagy defects. UC more associated with HLA region. Shared: barrier dysfunction, microbiome dysbiosis. Different: Crohn's transmural, UC mucosal.",
        key_points=["NOD2 Crohn's association", "IL23R shared", "Autophagy defects", "HLA in UC", "Microbiome", "Barrier dysfunction"],
        required_entities=["Crohn's disease", "ulcerative colitis", "NOD2", "inflammatory bowel"]
    ),
    LongFormQuestion(
        id="q12_hypertension",
        question="Describe the genetic factors contributing to essential hypertension. How do genes like AGT, ACE, and ADD1 influence blood pressure?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="Essential hypertension is polygenic. AGT affects renin-angiotensin system. ACE I/D polymorphism influences enzyme levels. ADD1 affects sodium handling. GWAS identified 100+ loci affecting vascular tone, sodium balance, and cardiac output.",
        key_points=["Polygenic trait", "Renin-angiotensin system", "AGT variants", "ACE polymorphism", "Sodium handling", "Multiple pathways"],
        required_entities=["hypertension", "AGT", "ACE", "blood pressure"]
    ),
    LongFormQuestion(
        id="q13_lupus",
        question="Explain the genetic basis of systemic lupus erythematosus (SLE). What role do complement genes and interferon pathways play?",
        question_type="explanation",
        difficulty="hard",
        reference_answer="SLE involves HLA class II, complement deficiencies (C1q, C4), and interferon pathway genes (IRF5, STAT4). Complement defects impair immune complex clearance. Type I interferon signature is hallmark. Affects multiple organs.",
        key_points=["HLA class II", "Complement deficiency", "C1q C4 genes", "Interferon signature", "IRF5 STAT4", "Immune complex clearance"],
        required_entities=["lupus", "SLE", "complement", "interferon", "autoimmune"]
    ),
    LongFormQuestion(
        id="q14_hemophilia",
        question="Compare hemophilia A and hemophilia B. What genes are involved and how do they affect the coagulation cascade?",
        question_type="comparison",
        difficulty="medium",
        reference_answer="Hemophilia A: F8 gene mutations, Factor VIII deficiency, more common (80%). Hemophilia B: F9 gene mutations, Factor IX deficiency. Both X-linked recessive, affect intrinsic coagulation pathway, cause similar bleeding phenotype.",
        key_points=["F8 gene hemophilia A", "F9 gene hemophilia B", "X-linked inheritance", "Coagulation cascade", "Factor VIII/IX", "Intrinsic pathway"],
        required_entities=["hemophilia", "F8", "F9", "coagulation", "bleeding"]
    ),
    LongFormQuestion(
        id="q15_asthma_genetics",
        question="What genetic variants are associated with asthma susceptibility? How do they relate to immune function and airway biology?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="Asthma involves IL33, TSLP, IL4, IL13 (Th2 pathway), ORMDL3/GSDMB (17q21), and ADAM33 (airway remodeling). Genetic variants affect IgE levels, eosinophilia, bronchial hyperreactivity. Gene-environment interactions important.",
        key_points=["Th2 pathway genes", "IL33 TSLP", "17q21 locus", "ADAM33 remodeling", "IgE regulation", "Gene-environment"],
        required_entities=["asthma", "IL33", "IL4", "airway", "allergy"]
    ),
    LongFormQuestion(
        id="q16_colorectal_cancer",
        question="Describe the genetic pathways involved in colorectal cancer development. Compare sporadic cases with Lynch syndrome.",
        question_type="comparison",
        difficulty="hard",
        reference_answer="Sporadic CRC follows adenoma-carcinoma sequence: APC→KRAS→TP53. Lynch syndrome: MLH1/MSH2 mismatch repair defects cause microsatellite instability. Lynch has earlier onset, right-sided tumors, better prognosis. Both involve Wnt pathway.",
        key_points=["APC KRAS TP53 pathway", "Lynch syndrome MMR", "Microsatellite instability", "Adenoma-carcinoma sequence", "Wnt pathway", "Different prognosis"],
        required_entities=["colorectal cancer", "APC", "Lynch syndrome", "MLH1"]
    ),
    LongFormQuestion(
        id="q17_muscular_dystrophy",
        question="Explain how mutations in the DMD gene cause Duchenne muscular dystrophy. What is the role of dystrophin in muscle function?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="DMD gene is largest human gene. Dystrophin links cytoskeleton to extracellular matrix in muscle. Mutations cause absent/truncated protein, membrane instability, calcium influx, muscle fiber necrosis, and progressive weakness.",
        key_points=["DMD gene mutations", "Dystrophin function", "Membrane stability", "Cytoskeleton link", "Muscle degeneration", "X-linked inheritance"],
        required_entities=["Duchenne", "dystrophin", "DMD", "muscle", "muscular dystrophy"]
    ),
    LongFormQuestion(
        id="q18_thyroid_disease",
        question="Compare the genetic factors in Graves' disease versus Hashimoto's thyroiditis. What immune mechanisms differ?",
        question_type="comparison",
        difficulty="hard",
        reference_answer="Both involve HLA-DR3, CTLA4, PTPN22. Graves': TSH receptor antibodies cause hyperthyroidism. Hashimoto's: cytotoxic T cells destroy thyroid causing hypothyroidism. Different end phenotypes from shared genetic susceptibility.",
        key_points=["HLA-DR3 shared", "CTLA4 association", "TSH receptor antibodies", "Cytotoxic T cells", "Hyper vs hypothyroid", "Autoimmune thyroid"],
        required_entities=["Graves' disease", "Hashimoto's", "thyroid", "autoimmune", "HLA"]
    ),
    LongFormQuestion(
        id="q19_epilepsy",
        question="What genetic factors contribute to epilepsy? Discuss ion channel genes and their role in seizure disorders.",
        question_type="explanation",
        difficulty="medium",
        reference_answer="Many epilepsies involve ion channel mutations (channelopathies). SCN1A (sodium), KCNQ2 (potassium), GABRA1 (GABA receptor) affect neuronal excitability. Can cause Dravet syndrome, benign familial seizures, or generalized epilepsy.",
        key_points=["Ion channel mutations", "SCN1A sodium channel", "KCNQ2 potassium", "GABA receptors", "Neuronal excitability", "Dravet syndrome"],
        required_entities=["epilepsy", "SCN1A", "ion channel", "seizure"]
    ),
    LongFormQuestion(
        id="q20_obesity_genetics",
        question="Describe the genetic architecture of obesity. How do genes like FTO, MC4R, and LEP contribute to body weight regulation?",
        question_type="explanation",
        difficulty="medium",
        reference_answer="FTO is strongest common variant, affects satiety. MC4R mutations cause monogenic obesity via hypothalamic appetite regulation. LEP/LEPR defects cause severe early-onset obesity. Most obesity is polygenic with 100+ variants.",
        key_points=["FTO common variant", "MC4R appetite regulation", "Leptin pathway", "Hypothalamic control", "Polygenic architecture", "Energy homeostasis"],
        required_entities=["obesity", "FTO", "MC4R", "leptin", "body weight"]
    ),
]

# ============== MAIN ==============

def run_evaluation():
    """Run full evaluation with all improvements."""
    
    print("="*70)
    print("LONG-FORM RAG EVALUATION v3")
    print("="*70)
    print("Improvements:")
    print("  1. LLM-based triple deduplication (not string matching)")
    print("  2. Self-verification for ALL systems")
    print(f"  3. Triple extraction mode: {EXTRACTION_MODE}")
    print("="*70)
    
    # Load config
    config = {
        'context_volume': int(config_data["CONTEXT_VOLUME"]),
        'similarity_threshold': float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"]),
        'min_similarity': float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
    }
    
    print("\n[1/4] Loading models...")
    vectorstore = load_chroma(config_data["VECTOR_DB_PATH"], 
                              config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"])
    embedding_function = load_sentence_transformer(config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"])
    node_context_df = pd.read_csv(config_data["NODE_CONTEXT_PATH"])
    semantic_index, semantic_chunks = load_semantic_db(
        config_data["SEMANTIC_DB_PATH"], config_data["SEMANTIC_CHUNKS_PATH"])
    semantic_embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    
    print("[2/4] Initializing systems...")
    vanilla = VanillaRAG(semantic_index, semantic_chunks, semantic_embed_model)
    kg_rag = KGRAG(vectorstore, embedding_function, node_context_df, config)
    sarg_orag = SARGORAG(vectorstore, embedding_function, node_context_df,
                         semantic_index, semantic_chunks, semantic_embed_model, config)
    
    system_prompt = "You are an expert biomedical researcher. Provide comprehensive, accurate answers."
    
    print(f"\n[3/4] Evaluating {len(QUESTIONS)} questions...")
    
    all_results = []
    
    for q in tqdm(QUESTIONS, desc="Questions"):
        print(f"\n{'='*70}")
        print(f"📝 {q.id}: {q.question[:60]}...")
        print("="*70)
        
        result = {"question_id": q.id, "question": q.question}
        
        # Vanilla RAG (with self-verify)
        print("\n🟢 Running Vanilla RAG (with self-verification)...")
        v_answer, v_ctx, v_meta = vanilla.generate(q.question, system_prompt, use_self_verify=True)
        v_score = judge_answer(q, v_answer)
        result["vanilla_score"] = v_score.total
        result["vanilla_meta"] = str(v_meta)
        print(f"   Score: {v_score.total:.1f}/10")
        
        # KG-RAG (with self-verify)
        print("\n🔵 Running KG-RAG (with self-verification)...")
        k_answer, k_ctx, k_meta = kg_rag.generate(q.question, system_prompt, use_self_verify=True)
        k_score = judge_answer(q, k_answer)
        result["kg_score"] = k_score.total
        result["kg_meta"] = str(k_meta)
        print(f"   Score: {k_score.total:.1f}/10")
        
        # SARG+ORAG (with LLM dedup)
        print("\n🟣 Running SARG+ORAG (LLM dedup + self-verify)...")
        s_answer, s_ctx, s_meta = sarg_orag.generate(q.question, system_prompt, extraction_mode=EXTRACTION_MODE)
        s_score = judge_answer(q, s_answer)
        result["sarg_score"] = s_score.total
        result["sarg_meta"] = str(s_meta)
        print(f"   Score: {s_score.total:.1f}/10")
        print(f"   Triples: {s_meta['kg_triples']} KG + {s_meta['semantic_triples']} Sem → {s_meta['merged_triples']} merged ({s_meta['duplicates_found']} dups)")
        
        all_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("[4/4] RESULTS SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    
    print(f"\n📊 Average Scores (all with self-verification):")
    print(f"   Vanilla RAG: {df['vanilla_score'].mean():.2f}/10")
    print(f"   KG-RAG:      {df['kg_score'].mean():.2f}/10")
    print(f"   SARG+ORAG:   {df['sarg_score'].mean():.2f}/10")
    
    # Save
    output_path = os.path.join(config_data["SAVE_RESULTS_PATH"], "longform_eval_v3.csv")
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    run_evaluation()

