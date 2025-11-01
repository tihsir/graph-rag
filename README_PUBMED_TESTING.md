# PubMed Data Testing for GraphRAG

This document describes how to use PubMed data for testing the GraphRAG system, based on the approach from [KARE](https://github.com/pat-jj/KARE).

## Overview

The PubMed testing module allows you to:
1. Fetch medical abstracts from PubMed using the NCBI E-utilities API
2. Convert them to markdown format for ingestion
3. Test the GraphRAG pipeline with real medical literature

## Usage

### Quick Start

```python
from graphrag.tests.pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries

# Create test documents from PubMed
queries = get_sample_medical_queries()  # Returns sample medical queries
files = create_pubmed_test_documents(queries[:2], output_dir="./test_data/pubmed")

# Ingest into GraphRAG
from graphrag.app.services.pipeline import pipeline
doc_ids = await pipeline.ingest(paths=files, tags=["pubmed", "test"])

# Build knowledge graph
result = await pipeline.rebuild_graph(min_conf=0.4)
print(f"Extracted {result['entities']} entities and {result['relations']} relations")
```

### Using CLI

```bash
# First, create PubMed test documents
python -m graphrag.tests.pubmed_test_data

# Then ingest them
python -m graphrag.cli.grag ingest --path ./test_data/pubmed/*.md --tag pubmed

# Build graph
python -m graphrag.cli.grag build-graph

# Query
python -m graphrag.cli.grag query "What are the treatments for asthma?"
```

### Custom Queries

```python
from graphrag.tests.pubmed_test_data import create_pubmed_test_documents

# Your own medical queries
custom_queries = [
    "heart failure treatment",
    "COVID-19 vaccination",
    "mental health therapy"
]

files = create_pubmed_test_documents(custom_queries)
```

## How It Works

Based on KARE's implementation:

1. **Search PubMed**: Uses NCBI E-utilities to search for articles matching medical concepts
2. **Fetch Abstracts**: Retrieves full abstracts for top results
3. **Format for Ingestion**: Converts to markdown format compatible with our ingestion pipeline
4. **Graph Building**: Entity and relation extraction works the same as with any ingested documents

## Rate Limiting

The module respects PubMed's rate limits (3 requests/second). For large-scale testing, consider:
- Using a local PubMed database dump (see KARE's `download_pubmed.py`)
- Implementing caching
- Using batch processing with delays

## Notes

- The PubMed API is free but has rate limits
- Abstracts are fetched in batches of 5 per request
- Each query retrieves up to 10 abstracts by default
- Test files are saved as markdown for easy inspection

## Reference

This implementation is inspired by the KARE project's PubMed integration:
- [KARE GitHub](https://github.com/pat-jj/KARE)
- [KARE Paper](https://arxiv.org/abs/2410.04585)

