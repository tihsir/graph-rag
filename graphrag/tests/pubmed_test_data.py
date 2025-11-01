"""PubMed data integration for testing GraphRAG using KARE's approach."""

import json
import time
import re
from typing import List, Tuple
from urllib.request import urlopen
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

# PubMed API constants (from KARE)
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_SEARCH_URL = PUBMED_BASE_URL + "esearch.fcgi"
PUBMED_FETCH_URL = PUBMED_BASE_URL + "efetch.fcgi"
MAX_ABSTRACTS = 10
ABSTRACTS_PER_REQUEST = 5


def search_pubmed(term: str, retmax: int = MAX_ABSTRACTS) -> List[str]:
    """Search PubMed for articles related to the given term."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "xml"
    }
    url = PUBMED_SEARCH_URL + "?" + urlencode(params)
    with urlopen(url) as response:
        tree = ET.parse(response)
        root = tree.getroot()
        id_list = root.find("IdList")
        if id_list is None:
            return []
        return [id_elem.text for id_elem in id_list.findall("Id")]


def fetch_pubmed_abstracts(pmids: List[str]) -> List[dict]:
    """Fetch abstracts for given PubMed IDs. Returns list of dicts with title, abstract, authors."""
    if not pmids:
        return []
    
    documents = []
    for i in range(0, len(pmids), ABSTRACTS_PER_REQUEST):
        batch = pmids[i:i+ABSTRACTS_PER_REQUEST]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract"
        }
        url = PUBMED_FETCH_URL + "?" + urlencode(params)
        try:
            with urlopen(url) as response:
                tree = ET.parse(response)
                root = tree.getroot()
                for article in root.findall(".//PubmedArticle"):
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    
                    abstract_elem = article.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Get authors
                    authors_list = []
                    for author in article.findall(".//Author"):
                        last_name = author.findtext("LastName")
                        fore_name = author.findtext("ForeName")
                        if last_name and fore_name:
                            authors_list.append(f"{fore_name} {last_name}")
                    authors = ", ".join(authors_list)
                    
                    if pmid and (title or abstract):
                        documents.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        })
            time.sleep(0.5)  # Respect PubMed's rate limit
        except Exception as e:
            print(f"Error fetching batch: {e}")
            continue
    
    return documents


def create_pubmed_test_documents(queries: List[str], output_dir: str = "./test_data/pubmed") -> List[str]:
    """
    Create test documents from PubMed abstracts for given medical queries.
    Returns list of file paths created.
    """
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    for i, query in enumerate(queries):
        print(f"Fetching PubMed data for query {i+1}/{len(queries)}: {query}")
        
        # Search PubMed
        pmids = search_pubmed(query, retmax=MAX_ABSTRACTS)
        if not pmids:
            print(f"No results found for query: {query}")
            continue
        
        # Fetch abstracts
        documents = fetch_pubmed_abstracts(pmids)
        if not documents:
            print(f"No abstracts retrieved for query: {query}")
            continue
        
        # Combine documents into a single markdown file
        content_parts = [f"# PubMed Articles for: {query}\n\n"]
        for doc in documents:
            content_parts.append(f"## {doc['title']}\n\n")
            if doc['authors']:
                content_parts.append(f"**Authors:** {doc['authors']}\n\n")
            content_parts.append(f"**PMID:** {doc['pmid']}\n\n")
            content_parts.append(f"**Abstract:**\n\n{doc['abstract']}\n\n")
            content_parts.append(f"**Link:** {doc['url']}\n\n")
            content_parts.append("---\n\n")
        
        # Save as markdown file
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')[:50]
        file_path = os.path.join(output_dir, f"pubmed_{safe_query}_{i}.md")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("".join(content_parts))
        
        created_files.append(file_path)
        print(f"Created: {file_path} ({len(documents)} articles)")
    
    return created_files


def get_sample_medical_queries() -> List[str]:
    """Get sample medical queries for testing."""
    return [
        "asthma treatment",
        "diabetes management",
        "hypertension medication",
        "cardiac arrhythmia",
        "pneumonia diagnosis"
    ]


if __name__ == "__main__":
    # Example usage
    queries = get_sample_medical_queries()
    files = create_pubmed_test_documents(queries[:2])  # Just test with first 2 queries
    print(f"\nCreated {len(files)} test files:")
    for f in files:
        print(f"  - {f}")

