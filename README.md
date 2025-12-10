# Graph-RAG Evaluation Framework

Long-form QA evaluation comparing three RAG approaches using LLM-as-a-Judge.

## 🏆 Results Summary (20 Questions)

| Rank | System | Score |
|------|--------|-------|
| 🥇 | **SARG+ORAG** | **9.86/10** |
| 🥈 | Vanilla RAG | 9.68/10 |
| 🥉 | KG-RAG | 9.66/10 |

*Evaluated on 20 diverse biomedical long-form questions*

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

## 📈 Detailed Results (20 Questions)

### Score Summary by Question Type

| Type | Questions | Avg Vanilla | Avg KG-RAG | Avg SARG+ORAG |
|------|-----------|-------------|------------|---------------|
| Explanation | 12 | 9.73 | 9.68 | 9.88 |
| Comparison | 8 | 9.58 | 9.63 | 9.83 |
| **Total** | **20** | **9.68** | **9.66** | **9.86** |

### Triple Extraction Statistics (All 20 Questions)

```
Total KG Triples:       1,339
Total Semantic Triples: 1,264
Total Duplicates Found: 57 (LLM-identified)
Total Merged Triples:   2,546
```

### Sample Questions & Scores

| Question | Topic | Best System | Score |
|----------|-------|-------------|-------|
| q4_brca_cancer | BRCA & Cancer | All tied | 10.0 |
| q5_alzheimer_apoe | APOE & Alzheimer's | All tied | 10.0 |
| q13_lupus | SLE & Complement | SARG+ORAG | 10.0 |
| q18_thyroid_disease | Graves' vs Hashimoto's | SARG+ORAG | 10.0 |

See `data/my_results/evaluation_output.md` for complete results and triple outputs.

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
