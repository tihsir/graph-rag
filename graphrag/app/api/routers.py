"""FastAPI routers for REST API."""

import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    IngestRequest,
    IngestResponse,
    RebuildGraphRequest,
    RebuildGraphResponse,
    QueryRequest,
    QueryResponse,
    TraceResponse,
    HealthResponse,
    Citation,
    EvidenceGraph,
)
from ..services.pipeline import pipeline
from ..retrieval.planner import planner
from ..services.pipeline import get_graph_store
from ..retrieval.ranker import reranker
from ..retrieval.aggregator import aggregator
from ..data.metadata_store import metadata_store
from ..core.providers import get_llm_provider, get_embedding_provider
from ..core.tracing import create_trace, trace_store
from ..core.config import settings
from ..core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="GraphRAG API", version="0.1.0")

# Initialize providers and stores (lazy - only when actually used)
_llm = None
_embedding_provider = None
_vector_store = None
_graph_store = None

def get_llm():
    """Lazy get LLM provider."""
    global _llm
    if _llm is None:
        _llm = get_llm_provider()
    return _llm

def get_embedding():
    """Lazy get embedding provider."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = get_embedding_provider()
    return _embedding_provider

def get_vector_store_instance():
    """Lazy get vector store."""
    global _vector_store
    if _vector_store is None:
        from ..retrieval.vector_store import _get_vector_store
        _vector_store = _get_vector_store()
    return _vector_store

def get_graph_store_instance():
    """Lazy get graph store."""
    global _graph_store
    if _graph_store is None:
        _graph_store = get_graph_store()
    return _graph_store


def load_prompt(prompt_name: str) -> str:
    """Load prompt from TOML file."""
    from pathlib import Path
    try:
        import tomli
    except ImportError:
        # Fallback if tomli not available
        return ""
    
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.toml"
    if prompt_path.exists():
        with open(prompt_path, "rb") as f:
            data = tomli.load(f)
            return data.get("system", {}).get("prompt", "")
    return ""


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents from paths or URLs."""
    try:
        doc_ids = await pipeline.ingest(paths=request.paths, urls=request.urls, tags=request.tags)
        return IngestResponse(doc_ids=doc_ids, message=f"Ingested {len(doc_ids)} documents")
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild_graph", response_model=RebuildGraphResponse)
async def rebuild_graph(request: RebuildGraphRequest):
    """Rebuild knowledge graph from chunks."""
    try:
        result = await pipeline.rebuild_graph(min_conf=request.min_conf, batch_size=request.batch_size)
        return RebuildGraphResponse(
            entities=result["entities"],
            relations=result["relations"],
            message="Graph rebuilt successfully",
        )
    except Exception as e:
        logger.error("Graph rebuild failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute query with GraphRAG or vanilla RAG."""
    start_time = time.time()
    trace = create_trace(request.query)
    
    try:
        # Step 1: Plan retrieval (lazy planner access)
        plan_start = time.time()
        from ..retrieval.planner import _get_planner
        planner_instance = _get_planner()
        plan = await planner_instance.plan(request.query, mode=request.mode)
        trace.route = plan.route
        trace.route_rationale = plan.rationale
        trace.latencies["planning"] = time.time() - plan_start
        
        # Step 2: Retrieve
        retrieve_start = time.time()
        graph_chunks = []
        vector_chunks = []
        
        if plan.route == "graph":
            # Entity linking to find seed nodes
            entity_link_start = time.time()
            from ..graph.extraction import _get_entity_extractor
            entity_ext = _get_entity_extractor()
            entities = await entity_ext.extract(request.query, "query", "query_chunk", 0)
            
            # Search graph for matching nodes
            graph_store = get_graph_store_instance()
            seed_node_ids = []
            for entity in entities:
                if entity.confidence >= 0.5:
                    nodes = await graph_store.search_nodes(entity.name, top_k=3)
                    seed_node_ids.extend([n.node_id for n in nodes])
            
            trace.candidates["seed_nodes"] = seed_node_ids
            
            if seed_node_ids:
                # K-hop expansion
                from ..graph.traversal import k_hop_expansion
                expanded_nodes = await k_hop_expansion(
                    graph_store,
                    seed_node_ids[:10],  # Limit seeds
                    max_hops=request.hops,
                    min_confidence=settings.min_confidence,
                    max_nodes=settings.max_graph_nodes,
                )
                
                # Get connected chunks
                chunk_ids = await graph_store.get_connected_chunks(
                    [n.node_id for n in expanded_nodes],
                    max_hops=request.hops,
                    min_confidence=settings.min_confidence,
                )
                
                # Load chunk texts
                chunks = metadata_store.get_chunks_by_ids(chunk_ids[:request.k * 2])
                graph_chunks = [(c.chunk_id, c.text, {"doc_id": c.doc_id, "section": c.section}) for c in chunks]
                
                trace.candidates["expanded_nodes"] = [n.node_id for n in expanded_nodes]
                trace.latencies["entity_linking"] = time.time() - entity_link_start
            
            # Fallback to vector search if graph is empty
            if not graph_chunks:
                plan.route = "rag"
                trace.route = "rag"
                trace.route_rationale = "Graph traversal empty, falling back to RAG"
        
        # Vector search (always performed, but mixed with graph results)
        vector_start = time.time()
        embedding_provider = get_embedding()
        vector_store = get_vector_store_instance()
        query_embedding = await embedding_provider.embed([request.query])
        vector_results = await vector_store.search(query_embedding[0], top_k=request.k * 2)
        
        # Load chunk texts for vector results
        vector_chunk_ids = [r[0] for r in vector_results]
        vector_chunks_loaded = metadata_store.get_chunks_by_ids(vector_chunk_ids)
        vector_chunks = [
            (c.chunk_id, c.text, {"doc_id": c.doc_id, "section": c.section, "score": next((r[1] for r in vector_results if r[0] == c.chunk_id), 0.0)})
            for c in vector_chunks_loaded
        ]
        trace.latencies["vector_search"] = time.time() - vector_start
        
        # Aggregate results
        all_chunks = aggregator.aggregate(graph_chunks, vector_chunks, top_k=request.k * 2)
        trace.retrieved_chunks = [{"chunk_id": c[0], "text": c[1][:100], "metadata": c[2]} for c in all_chunks]
        
        # Rerank if enabled (lazy reranker access)
        if request.rerank and all_chunks:
            rerank_start = time.time()
            from ..retrieval.ranker import _get_reranker
            reranker_instance = _get_reranker()
            all_chunks = await reranker_instance.rerank(request.query, all_chunks, top_k=request.k)
            trace.latencies["reranking"] = time.time() - rerank_start
        
        trace.latencies["retrieval"] = time.time() - retrieve_start
        
        # Step 3: Generate answer
        generate_start = time.time()
        chunks_for_context = all_chunks[:request.k]
        context_texts = [f"[S:{c[2].get('doc_id', 'unknown')}#{c[0]}]\n{c[1]}" for c in chunks_for_context]
        context = "\n\n".join(context_texts)
        
        if plan.route == "graph":
            system_prompt = load_prompt("graph_answer") or "Use graph neighbors and retrieved passages to answer precisely. Cite each claim with [S:doc#chunk]."
        else:
            system_prompt = load_prompt("rag_answer") or "Synthesize from top-k passages; prioritize precision; include inline citations [S:doc#chunk]."
        
        user_prompt = f"""Context:
{context}

Question: {request.query}

Answer:"""
        
        trace.prompts["system"] = system_prompt
        trace.prompts["user"] = user_prompt
        
        llm = get_llm()
        answer = await llm.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        trace.output = answer
        trace.latencies["generation"] = time.time() - generate_start
        
        # Extract citations
        import re
        citation_pattern = re.compile(r"\[S:([^#]+)#([^\]]+)\]")
        citations_found = citation_pattern.findall(answer)
        citations = []
        citation_map = {c[0]: c for c in chunks_for_context}
        
        for doc_id, chunk_id in citations_found:
            if chunk_id in citation_map:
                chunk_data = citation_map[chunk_id]
                citations.append(
                    Citation(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=chunk_data[1][:200],
                        score=chunk_data[2].get("score"),
                    )
                )
        
        trace.citations = [f"{doc_id}#{chunk_id}" for doc_id, chunk_id in citations_found]
        trace.latencies["total"] = time.time() - start_time
        
        # Build evidence graph if graph route
        evidence_graph = None
        if plan.route == "graph" and "seed_nodes" in trace.candidates:
            evidence_graph = EvidenceGraph(
                seed_nodes=trace.candidates.get("seed_nodes", []),
                expanded_nodes=trace.candidates.get("expanded_nodes", []),
                used_edges=[],
            )
            trace.evidence_graph = evidence_graph.model_dump()
        
        # Save trace
        if trace_store:
            trace_store.save(trace)
        
        return QueryResponse(
            answer=answer,
            route=plan.route,
            citations=citations,
            evidence_graph=evidence_graph,
            trace_id=trace.trace_id,
            latencies=trace.latencies,
        )
    
    except Exception as e:
        logger.error("Query failed", error=str(e), trace_id=trace.trace_id)
        trace.latencies["total"] = time.time() - start_time
        if trace_store:
            trace_store.save(trace)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str):
    """Retrieve trace by ID."""
    if not trace_store:
        raise HTTPException(status_code=404, detail="Tracing not enabled")
    
    trace = trace_store.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return TraceResponse(**trace.model_dump(mode="json"))


@app.get("/healthz", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")

