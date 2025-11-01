# GraphRAG: Production-Ready Knowledge Graph Augmented Retrieval

A production-ready implementation of GraphRAG with promptable query flows, supporting both graph-aware and vanilla RAG retrieval modes.

## Features

- **Multi-format Ingestion**: PDFs, HTML, Markdown, plain text, and web URLs
- **Semantic Chunking**: Heading-aware boundaries with token-based fallback
- **Knowledge Graph**: Entity/relation extraction with NetworkX+Postgres or Neo4j backends
- **Hybrid Retrieval**: Vector search + graph traversal with reranking
- **Promptable Query Flows**: Automatic routing between GraphRAG, vanilla RAG, and direct generation
- **Multiple Interfaces**: CLI, REST API, and Jupyter notebooks
- **Observability**: Structured logging and trace storage
- **Evaluation**: IR metrics and LLM-based judge

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd graphRAG
```

2. Install dependencies:
```bash
poetry install
# or
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

4. Prepare test data (PubMed abstracts):
```bash
# Option 1: Automated setup script
python setup_test_data.py

# Option 2: Use CLI
grag setup-pubmed
grag ingest --path ./test_data/pubmed/*.md --tag pubmed
grag build-graph
```

5. Start services with Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Usage

#### CLI

```bash
# Setup PubMed test data
grag setup-pubmed

# Ingest documents (using PubMed data or your own)
grag ingest --path ./test_data/pubmed/*.md --tag pubmed

# Build knowledge graph
grag build-graph --min-conf 0.4

# Query (using medical questions that match PubMed data)
grag query "What are the treatments for asthma?" --mode auto

# Check status
grag status
```

#### REST API

```bash
# Start API server
uvicorn graphrag.app.api.routers:app --reload

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["./data/doc.pdf"], "tags": ["demo"]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who founded company X?", "mode": "auto", "k": 10}'
```

#### Python API

```python
from graphrag.app.services.pipeline import pipeline
from graphrag.app.api.routers import query
from graphrag.app.api.schemas import QueryRequest
from graphrag.tests.pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries

# Setup PubMed test data
files = create_pubmed_test_documents(get_sample_medical_queries()[:2])

# Ingest
doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "test"])

# Build graph
result = await pipeline.rebuild_graph()

# Query
request = QueryRequest(query="What are treatments for asthma?", mode="auto")
response = await query(request)
print(response.answer)
```

## Configuration

All configuration is done via environment variables. See `.env.example` for options:

- **LLM Provider**: `LLM_PROVIDER=openai|anthropic|vllm`
- **Embedding Provider**: `EMBEDDING_PROVIDER=openai|local`
- **Vector Store**: `VECTOR_STORE=faiss|qdrant`
- **Graph Store**: `GRAPH_STORE=nx_pg|neo4j`
- **Reranker**: `RERANKER_PROVIDER=local|cohere`

## Architecture

```
graphrag/
├── app/
│   ├── api/          # FastAPI REST endpoints
│   ├── core/         # Config, logging, providers
│   ├── data/         # Ingestion, chunking, metadata store
│   ├── graph/        # Entity extraction, canonicalization, graph stores
│   ├── retrieval/    # Vector store, planner, ranker, aggregator
│   ├── prompts/      # TOML prompt files
│   ├── services/     # Pipeline orchestration
│   └── eval/         # Evaluation harness
├── cli/              # Typer CLI
├── examples/         # Jupyter notebooks
└── tests/            # Pytest tests
```

## Query Flow

1. **Routing**: LLM classifies query intent (graph/rag/direct)
2. **Graph Path**: Entity linking → k-hop expansion → chunk collection
3. **Vector Path**: Embedding-based search
4. **Aggregation**: Merge and deduplicate results
5. **Reranking**: Score and reorder by relevance
6. **Generation**: LLM synthesizes answer with citations

## Testing

```bash
# Run unit tests
pytest tests/

# Run end-to-end test with PubMed data
python test_end_to_end.py

# Or use the setup script for interactive setup
python setup_test_data.py
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing instructions.

## Evaluation

Run evaluation comparing GraphRAG vs vanilla RAG:

```python
from graphrag.app.eval.run_eval import run_evaluation
from graphrag.app.eval.datasets import create_sample_dataset

dataset = create_sample_dataset()
results = await run_evaluation(dataset, modes=["graph", "rag"])
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

