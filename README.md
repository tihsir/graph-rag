# Graph-RAG Evaluation Framework

Long-form QA evaluation comparing three RAG approaches using LLM-as-a-Judge.

## 🎯 Overview

This framework evaluates and compares:
1. **Vanilla RAG** - Simple semantic retrieval (FAISS)
2. **KG-RAG** - Knowledge Graph retrieval (ChromaDB + SPOKE)
3. **SARG+ORAG** - Hybrid system with triple extraction and self-verification

## 📊 Results Summary

| System | Score | Rank |
|--------|-------|------|
| **SARG+ORAG** | **8.88/10** | 🥇 |
| **KG-RAG** | **8.67/10** | 🥈 |
| **Vanilla RAG** | **8.00/10** | 🥉 |

## 🔧 Features

### (a) Chain of Reasoning
Shows all triples extracted and their usage order in the final answer.

### (b) Triple Deduplication  
Merges KG triples (Group A) + Semantic triples (Group B), removes duplicates.

### (c) Debug Evidence Display
Optionally displays raw KG and semantic context retrieved.

### (d) Enhanced Evaluation Rubric
8 metrics with rankings:
- Accuracy, Completeness, Relevance, Coherence
- Conciseness, Evidence Usage, Reasoning Depth, Factual Grounding

## 📁 Project Structure

```
graph-rag-eval/
├── kg_rag/
│   ├── utility.py                 # Core retrieval functions
│   ├── config_loader.py           # Configuration loading
│   └── rag_based_generation/
│       └── GPT/
│           ├── run_longform_eval.py    # Basic 3-system comparison
│           └── run_longform_eval_v2.py # Enhanced with all features
├── data/
│   └── my_results/
│       ├── longform_eval_summary.md
│       └── enhanced_longform_eval_v2.csv
├── config.yaml                    # System configuration
├── system_prompts.yaml            # LLM prompts
└── README.md
```

## 🚀 Usage

### Basic Evaluation (3 systems)
```bash
export GOOGLE_API_KEY="your-api-key"
python -m kg_rag.rag_based_generation.GPT.run_longform_eval
```

### Enhanced Evaluation (with all features)
```bash
python -m kg_rag.rag_based_generation.GPT.run_longform_eval_v2
```

### Configuration
Edit `run_longform_eval_v2.py`:
```python
DEBUG_MODE = True          # Show retrieved evidence (c)
SHOW_REASONING_CHAIN = True # Show triple chain (a)
```

## 📋 Triple Extraction Pipeline

```
Question
    │
    ├──► KG Retrieval ──► KG Triples (Group A)
    │                            │
    └──► Semantic Retrieval ──► Semantic Triples (Group B)
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │  Merge & Deduplicate │
                    └─────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │   Final Triples     │
                    │   + Raw Context     │
                    └─────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │  LLM Answer Gen     │
                    └─────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │  Self-Verification  │
                    └─────────────────────┘
                                 │
                                 ▼
                          Final Answer
```

## 📈 Evaluation Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| Accuracy | Medical facts correct? | 20% |
| Completeness | Covers all key points? | 20% |
| Relevance | Addresses the question? | 15% |
| Coherence | Well-organized? | 10% |
| Conciseness | Appropriately detailed? | 10% |
| Evidence Usage | Uses retrieved evidence? | 10% |
| Reasoning Depth | Shows clear reasoning? | 10% |
| Factual Grounding | Based on evidence? | 5% |

## 🔗 Dependencies

- `google-generativeai` - Gemini API
- `langchain` - LLM framework
- `chromadb` - Vector store for KG
- `faiss` - Semantic search
- `sentence-transformers` - Embeddings
- `pandas` - Data handling

## 📝 License

MIT

