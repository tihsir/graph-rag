"""Base abstract class for graph stores."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
from pydantic import BaseModel

from .extraction import Entity, Relation


class GraphNode(BaseModel):
    """Graph node (canonical entity)."""

    node_id: str
    name: str
    type: str
    aliases: List[str] = []
    metadata: Dict = {}


class GraphEdge(BaseModel):
    """Graph edge (relation)."""

    edge_id: str
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    provenance: List[Dict] = []  # List of {doc_id, chunk_id, char_spans}


class GraphStore(ABC):
    """Abstract graph store interface."""

    @abstractmethod
    async def add_node(self, node: GraphNode) -> None:
        """Add or update a node."""
        pass

    @abstractmethod
    async def add_edge(self, edge: GraphEdge) -> None:
        """Add or update an edge."""
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        pass

    @abstractmethod
    async def get_neighbors(
        self, node_id: str, relation_types: Optional[List[str]] = None, min_confidence: float = 0.0, max_hops: int = 1
    ) -> List[GraphNode]:
        """Get neighboring nodes via k-hop expansion."""
        pass

    @abstractmethod
    async def get_connected_chunks(
        self, node_ids: List[str], max_hops: int = 2, min_confidence: float = 0.0
    ) -> List[str]:
        """Get chunk IDs connected to nodes via graph traversal."""
        pass

    @abstractmethod
    async def search_nodes(self, query: str, top_k: int = 10) -> List[GraphNode]:
        """Search nodes by name/alias (text similarity)."""
        pass

