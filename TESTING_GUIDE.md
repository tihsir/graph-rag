# GraphRAG Testing Guide

This guide shows you how to test the GraphRAG system using PubMed data.

## Prerequisites

1. **Install dependencies:**
```bash
poetry install
# or
pip install -r requirements.txt  # if you create one
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY (or ANTHROPIC_API_KEY)
# - Database connection strings if needed
```

## Quick Test Options

### Option 1: Automated Setup Script (Recommended)

```bash
python setup_test_data.py
```

This script will:
- Fetch PubMed abstracts automatically
- Ingest them into the system
- Optionally build the knowledge graph
- Show you how to query

### Option 2: CLI Commands

```bash
# 1. Fetch PubMed data
grag setup-pubmed

# 2. Ingest the documents
grag ingest --path ./test_data/pubmed/*.md --tag pubmed

# 3. Build knowledge graph
grag build-graph --min-conf 0.4

# 4. Query
grag query "What are the treatments for asthma?" --mode auto

# 5. Check status
grag status
```

### Option 3: End-to-End Test Script

```bash
python test_end_to_end.py
```

This runs the complete workflow automatically and shows results.

### Option 4: Jupyter Notebook

Open `graphrag/examples/00_quickstart.ipynb` and run all cells. The notebook automatically:
- Fetches PubMed data if needed
- Ingests documents
- Builds the graph
- Runs queries

## What PubMed Data is Used

The system uses these sample medical queries by default:
- "asthma treatment"
- "diabetes management"
- "hypertension medication"
- "cardiac arrhythmia"
- "pneumonia diagnosis"

For each query, it fetches up to 10 PubMed abstracts, creating markdown files in `./test_data/pubmed/`.

## Test Queries to Try

Once the system is set up, try these queries that match the PubMed data:

- "What are the treatments for asthma?"
- "How is diabetes managed?"
- "What medications are used for hypertension?"
- "What is cardiac arrhythmia?"
- "How is pneumonia diagnosed?"

## Troubleshooting

### No PubMed data fetched

If fetching fails, check:
- Internet connection
- PubMed API accessibility
- Firewall settings

The PubMed API has rate limits (3 requests/second). If you hit limits, wait a few seconds and try again.

### Database connection errors

Make sure PostgreSQL is running and configured in `.env`:
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/graphrag
```

Or use SQLite for testing (update `metadata_store.py`).

### API key errors

Ensure your LLM provider API key is set in `.env`:
```
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

## Next Steps

After testing with PubMed data:

1. **Try your own documents**: Use the same ingestion pipeline with your own PDFs, markdown, or HTML files
2. **Custom queries**: Add your own PubMed queries using `grag setup-pubmed --query "your query"`
3. **Evaluation**: Run the evaluation harness to compare GraphRAG vs vanilla RAG
4. **API testing**: Start the FastAPI server and test via REST endpoints

## Files Created During Testing

- `./test_data/pubmed/*.md` - PubMed abstracts in markdown format
- `./data/faiss_index*` - Vector search indices (if using FAISS)
- `./data/traces/*.json` - Query traces for debugging
- Database tables in Postgres/SQLite for documents and chunks

## Support

If you encounter issues:
1. Check logs: The system uses structured logging
2. Review traces: `GET /traces/{trace_id}` shows detailed query execution
3. Check status: `grag status` shows system state

