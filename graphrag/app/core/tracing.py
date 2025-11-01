"""Tracing and observability for query flows."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class Trace(BaseModel):
    """Trace model for a single query execution."""

    trace_id: str
    query: str
    route: str  # graph, rag, direct
    route_rationale: Optional[str] = None
    timestamp: datetime
    latencies: Dict[str, float] = {}
    candidates: Dict[str, Any] = {}  # seed_nodes, expanded_nodes, vector_results, etc.
    retrieved_chunks: List[Dict[str, Any]] = []
    prompts: Dict[str, str] = {}
    output: Optional[str] = None
    citations: List[str] = []
    evidence_graph: Optional[Dict[str, Any]] = None


class TraceStore:
    """Simple file-based trace store."""

    def __init__(self, storage_path: str = "./data/traces"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, trace: Trace) -> None:
        """Save trace to disk."""
        trace_file = self.storage_path / f"{trace.trace_id}.json"
        with open(trace_file, "w") as f:
            json.dump(trace.model_dump(mode="json"), f, indent=2, default=str)
        logger.info("Trace saved", trace_id=trace.trace_id)

    def get(self, trace_id: str) -> Optional[Trace]:
        """Retrieve trace by ID."""
        trace_file = self.storage_path / f"{trace_id}.json"
        if not trace_file.exists():
            return None
        with open(trace_file, "r") as f:
            data = json.load(f)
        return Trace(**data)

    def list_recent(self, limit: int = 100) -> List[str]:
        """List recent trace IDs."""
        trace_files = sorted(self.storage_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [f.stem for f in trace_files[:limit]]


def create_trace(query: str) -> Trace:
    """Create a new trace instance."""
    return Trace(
        trace_id=str(uuid.uuid4()),
        query=query,
        route="",
        timestamp=datetime.utcnow(),
    )


trace_store = TraceStore(settings.trace_storage_path) if settings.enable_tracing else None

