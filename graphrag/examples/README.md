# GraphRAG Examples

## Running Notebooks

### Setup

Before running the notebooks, make sure:

1. **Install the package in development mode:**
```bash
# From the project root
pip install -e .
```

OR

2. **The notebook will automatically add the project root to Python path** (included in the first cell)

### Quickstart Notebook

The `00_quickstart.ipynb` notebook demonstrates:
- Fetching PubMed data
- Ingesting documents
- Building a knowledge graph
- Querying with GraphRAG

**To run:**
```bash
jupyter notebook 00_quickstart.ipynb
# or
jupyter lab 00_quickstart.ipynb
```

**Note:** If you get `ModuleNotFoundError`, make sure you've either:
- Installed the package with `pip install -e .`
- Or run the first cell which adds the path automatically

### Alternative: Use Python Scripts

If notebooks aren't working, use the Python scripts:
- `setup_test_data.py` - Interactive setup
- `test_end_to_end.py` - Automated test

