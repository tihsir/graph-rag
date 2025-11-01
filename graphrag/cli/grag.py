"""CLI interface for GraphRAG using Typer."""

import asyncio
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..app.services.pipeline import pipeline
from ..app.retrieval.planner import planner
from ..app.retrieval.vector_store import get_vector_store
from ..app.services.pipeline import get_graph_store
from ..app.retrieval.ranker import reranker
from ..app.retrieval.aggregator import aggregator
from ..app.data.metadata_store import metadata_store
from ..app.core.providers import get_llm_provider, get_embedding_provider
from ..app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = typer.Typer(name="grag", help="GraphRAG CLI")
console = Console()


@app.command()
def ingest(
    paths: Optional[List[str]] = typer.Option(None, "--path", "-p", help="File paths to ingest"),
    urls: Optional[List[str]] = typer.Option(None, "--url", "-u", help="URLs to ingest"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags for ingested documents"),
):
    """Ingest documents from files or URLs."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Ingesting documents...", total=None)
        doc_ids = asyncio.run(pipeline.ingest(paths=paths or [], urls=urls or [], tags=tags))
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Ingested {len(doc_ids)} documents")
    for doc_id in doc_ids:
        console.print(f"  - {doc_id}")


@app.command()
def build_graph(
    min_conf: float = typer.Option(0.4, "--min-conf", help="Minimum confidence threshold"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for processing"),
):
    """Rebuild knowledge graph from ingested chunks."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Building knowledge graph...", total=None)
        result = asyncio.run(pipeline.rebuild_graph(min_conf=min_conf, batch_size=batch_size))
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Graph built successfully")
    console.print(f"  - Entities: {result['entities']}")
    console.print(f"  - Relations: {result['relations']}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Query mode: auto, graph, or rag"),
    k: int = typer.Option(10, "--k", help="Number of results to retrieve"),
    hops: int = typer.Option(2, "--hops", help="Maximum graph hops"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable reranking"),
    max_tokens: int = typer.Option(1000, "--max-tokens", help="Maximum tokens in response"),
    temperature: float = typer.Option(0.7, "--temperature", help="LLM temperature"),
):
    """Query the GraphRAG system."""
    async def _query():
        from ..app.api.routers import query as query_endpoint
        from ..app.api.schemas import QueryRequest
        
        request = QueryRequest(
            query=query_text,
            mode=mode,
            k=k,
            hops=hops,
            rerank=rerank,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return await query_endpoint(request)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Querying...", total=None)
        response = asyncio.run(_query())
        progress.update(task, completed=True)
    
    console.print("\n[bold cyan]Answer:[/bold cyan]")
    console.print(response.answer)
    
    if response.citations:
        console.print("\n[bold cyan]Citations:[/bold cyan]")
        for citation in response.citations:
            console.print(f"  - {citation.doc_id}#{citation.chunk_id}")
    
    if response.evidence_graph:
        console.print("\n[bold cyan]Evidence Graph:[/bold cyan]")
        console.print(f"  - Seed nodes: {len(response.evidence_graph.seed_nodes)}")
        console.print(f"  - Expanded nodes: {len(response.evidence_graph.expanded_nodes)}")
    
    console.print(f"\n[dim]Route: {response.route} | Trace ID: {response.trace_id}[/dim]")


@app.command()
def setup_pubmed(
    queries: Optional[List[str]] = typer.Option(None, "--query", "-q", help="Medical queries to search"),
    output_dir: str = typer.Option("./test_data/pubmed", "--output", "-o", help="Output directory"),
):
    """Fetch PubMed data for testing."""
    from ..tests.pubmed_test_data import create_pubmed_test_documents, get_sample_medical_queries
    
    if not queries:
        queries = get_sample_medical_queries()
        console.print(f"[yellow]No queries provided, using sample queries: {len(queries)}[/yellow]")
    
    console.print(f"[cyan]Fetching PubMed data for {len(queries)} queries...[/cyan]")
    files = create_pubmed_test_documents(queries, output_dir=output_dir)
    
    console.print(f"[green]✓[/green] Created {len(files)} PubMed document files")
    for f in files:
        console.print(f"  - {f}")


@app.command()
def status():
    """Show system status."""
    from ..app.data.metadata_store import DocumentModel, ChunkModel, metadata_store
    
    session = metadata_store.get_session()
    try:
        doc_count = session.query(DocumentModel).count()
        chunk_count = session.query(ChunkModel).count()
        
        table = Table(title="GraphRAG Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Documents", str(doc_count))
        table.add_row("Chunks", str(chunk_count))
        
        console.print(table)
    finally:
        session.close()


if __name__ == "__main__":
    app()

