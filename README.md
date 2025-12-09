# Graph-RAG Evaluation Framework

Long-form QA evaluation comparing three RAG approaches using LLM-as-a-Judge.

## 🏆 Results Summary

| Rank | System | Score |
|------|--------|-------|
| 🥇 | **SARG+ORAG** | **9.58/10** |
| 🥈 | KG-RAG | 9.42/10 |
| 🥉 | Vanilla RAG | 8.92/10 |

## 🎯 Overview

This framework evaluates and compares three RAG (Retrieval-Augmented Generation) approaches:

1. **Vanilla RAG** - Simple semantic retrieval using FAISS
2. **KG-RAG** - Knowledge Graph retrieval using ChromaDB + SPOKE biomedical KG
3. **SARG+ORAG** - Hybrid system with:
   - Dual retrieval (KG + Semantic)
   - LLM-based triple extraction
   - Intelligent deduplication
   - Multi-step reasoning
   - Self-verification with checklist

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install google-generativeai langchain chromadb faiss-cpu sentence-transformers pandas tqdm

# Set your API key
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Running the Evaluation

```bash
cd graph-rag-eval

# Run the latest version (v3) - SARG+ORAG wins!
python -m kg_rag.rag_based_generation.GPT.run_longform_eval_v3

# Run basic 3-system comparison (v1)
python -m kg_rag.rag_based_generation.GPT.run_longform_eval

# Run enhanced evaluation with all features (v2)
python -m kg_rag.rag_based_generation.GPT.run_longform_eval_v2
```

### Configuration Options

Edit the config variables at the top of each script:

```python
# run_longform_eval_v3.py
DEBUG_MODE = True           # Show retrieved evidence
SHOW_REASONING_CHAIN = True # Show triple extraction details
EXTRACTION_MODE = "separate" # "fused" or "separate" triple extraction
```

## 📁 Project Structure

```
graph-rag-eval/
├── kg_rag/
│   ├── utility.py                      # Core retrieval functions
│   ├── config_loader.py                # Configuration loading
│   └── rag_based_generation/
│       └── GPT/
│           ├── run_longform_eval.py    # v1: Basic 3-system comparison
│           ├── run_longform_eval_v2.py # v2: Enhanced with features a-d
│           └── run_longform_eval_v3.py # v3: SARG+ORAG optimized (BEST)
├── data/
│   └── my_results/
│       ├── longform_eval_summary.md
│       ├── enhanced_longform_eval_v2.csv
│       └── longform_eval_v3.csv
├── config.yaml
├── system_prompts.yaml
└── README.md
```

## 🔧 Features

### Version 3 (Latest - SARG+ORAG Wins)

| Feature | Description |
|---------|-------------|
| **LLM Deduplication** | Uses Gemini to identify semantic duplicates across KG and semantic triples |
| **Multi-step Reasoning** | Analyzes question → extracts triples → generates answer → verifies |
| **Checklist Verification** | Ensures all entities, comparisons, mechanisms are covered |
| **Self-verify ALL systems** | Vanilla and KG-RAG also get self-verification |

### Pipeline Flow

```
                    ┌─────────────────┐
                    │    Question     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │  Vanilla RAG  │ │    KG-RAG     │ │  SARG+ORAG    │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            │                 │         ┌───────┴───────┐
            │                 │         ▼               ▼
            │                 │    ┌─────────┐    ┌─────────┐
            │                 │    │KG Retriev│    │Sem Retr │
            │                 │    └────┬────┘    └────┬────┘
            │                 │         │              │
            │                 │         ▼              ▼
            │                 │    ┌─────────────────────┐
            │                 │    │  Triple Extraction  │
            │                 │    │  (Group A + B)      │
            │                 │    └──────────┬──────────┘
            │                 │               │
            │                 │               ▼
            │                 │    ┌─────────────────────┐
            │                 │    │  LLM Deduplication  │
            │                 │    └──────────┬──────────┘
            │                 │               │
            │                 │               ▼
            │                 │    ┌─────────────────────┐
            │                 │    │  Question Analysis  │
            │                 │    └──────────┬──────────┘
            │                 │               │
            ▼                 ▼               ▼
    ┌─────────────────────────────────────────────────────┐
    │              Answer Generation                       │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │         Self-Verification (All Systems)             │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │              LLM-as-Judge Evaluation                │
    └─────────────────────────────────────────────────────┘
```

## 📊 Evaluation Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| Accuracy | Medical facts correct? | 25% |
| Completeness | Covers all key points? | 25% |
| Relevance | Addresses the question? | 25% |
| Coherence | Well-organized? | 25% |

## 📈 Detailed Results

### Per-Question Scores

| Question | Vanilla | KG-RAG | SARG+ORAG |
|----------|---------|--------|-----------|
| Q1: HLA-B & Psoriasis | 9.8 | **10.0** | 9.5 |
| Q2: Diabetes & NOD2 | 8.8 | 9.5 | **10.0** |
| Q3: Multi-disease Compare | 8.2 | 8.8 | **9.2** |
| **Average** | 8.92 | 9.42 | **9.58** |

### Triple Extraction Stats (Q3)

```
KG Triples:       69
Semantic Triples: 83
Duplicates Found: 7  (LLM-identified)
Final Merged:     145
```

### Sample LLM Deduplication Reasoning

```
"KG-2 duplicates SEM-4 because both express association between 
'Disease psoriasis' and 'Gene HLA-B'. ASSOCIATES_WITH is 
implicitly bi-directional, so reversed order counts as duplicate."
```

## 🔬 Adding New Questions

Edit the `QUESTIONS` list in any eval script:

```python
QUESTIONS = [
    LongFormQuestion(
        id="my_question",
        question="Your biomedical question here?",
        question_type="explanation",  # or "comparison", "mechanism"
        difficulty="hard",
        reference_answer="Expected comprehensive answer...",
        key_points=["Point 1", "Point 2", "Point 3"],
        required_entities=["Entity1", "Entity2"]
    ),
]
```

## 📝 Output Files

Results are saved to `data/my_results/`:

- `longform_eval_v3.csv` - Full results with scores and metadata
- `longform_eval_summary.md` - Human-readable summary

CSV columns include:
- `question_id`, `question`
- `vanilla_score`, `kg_score`, `sarg_score`
- `sarg_meta` - Triple counts, dedup reasoning, etc.

## 🛠️ Customization

### Using Different LLM

Modify `get_Gemini_response()` in `utility.py` to use different models:

```python
# Current: Gemini 2.0 Flash
model = genai.GenerativeModel("gemini-2.0-flash")

# Alternative: GPT-4, Claude, etc.
```

### Adjusting Retrieval

Edit `config.yaml`:

```yaml
CONTEXT_VOLUME: 100
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD: 97
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY: 0.5
```

## 📚 References

- [KG-RAG Paper](https://arxiv.org/abs/2311.17330)
- [SPOKE Biomedical Knowledge Graph](https://spoke.ucsf.edu/)
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)

## 📄 License

MIT

---

*SARG+ORAG achieves state-of-the-art results through intelligent triple extraction, LLM-based deduplication, and multi-step verification.*
