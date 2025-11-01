"""Pydantic schemas for API requests and responses."""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class IngestRequest(BaseModel):
    """Request schema for document ingestion."""

    paths: Optional[List[str]] = None
    urls: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class IngestResponse(BaseModel):
    """Response schema for ingestion."""

    doc_ids: List[str]
    message: str


class RebuildGraphRequest(BaseModel):
    """Request schema for graph rebuild."""

    min_conf: float = 0.4
    batch_size: int = 64


class RebuildGraphResponse(BaseModel):
    """Response schema for graph rebuild."""

    entities: int
    relations: int
    message: str


class QueryRequest(BaseModel):
    """Request schema for query."""

    query: str
    mode: Literal["auto", "graph", "rag"] = "auto"
    k: int = 10
    hops: int = 2
    rerank: bool = True
    max_tokens: int = 1000
    temperature: float = 0.7


class Citation(BaseModel):
    """Citation information."""

    doc_id: str
    chunk_id: str
    text: str
    score: Optional[float] = None


class EvidenceGraph(BaseModel):
    """Evidence graph summary."""

    seed_nodes: List[str]
    expanded_nodes: List[str]
    used_edges: List[Dict[str, Any]]


class QueryResponse(BaseModel):
    """Response schema for query."""

    answer: str
    route: str
    citations: List[Citation]
    evidence_graph: Optional[EvidenceGraph] = None
    trace_id: str
    latencies: Dict[str, float] = {}


class TraceResponse(BaseModel):
    """Response schema for trace retrieval."""

    trace_id: str
    query: str
    route: str
    route_rationale: Optional[str]
    timestamp: datetime
    latencies: Dict[str, float]
    candidates: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    prompts: Dict[str, str]
    output: Optional[str]
    citations: List[str]
    evidence_graph: Optional[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str

