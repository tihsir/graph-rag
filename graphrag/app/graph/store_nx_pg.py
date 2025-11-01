"""NetworkX + Postgres graph store implementation."""

import networkx as nx
from typing import List, Dict, Optional
import json
from sqlalchemy import create_engine, Column, String, Text, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from .store_base import GraphStore, GraphNode, GraphEdge
from ..core.config import settings
from ..core.logging import get_logger
from ..core.providers import get_embedding_provider

logger = get_logger(__name__)

Base = declarative_base()


class NodeTable(Base):
    """Postgres table for graph nodes."""

    __tablename__ = "graph_nodes"

    node_id = Column(String, primary_key=True)
    name = Column(String)
    type = Column(String)
    aliases = Column(JSON)
    node_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    embedding = Column(JSON)  # For similarity search


class EdgeTable(Base):
    """Postgres table for graph edges."""

    __tablename__ = "graph_edges"

    edge_id = Column(String, primary_key=True)
    source_id = Column(String)
    target_id = Column(String)
    relation_type = Column(String)
    confidence = Column(Float)
    provenance = Column(JSON)


class NetworkXPostgresStore(GraphStore):
    """NetworkX in-memory graph with Postgres persistence."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables - SQLAlchemy will add new columns but won't rename existing ones
        # For existing databases, you may need to manually migrate 'metadata' -> 'node_metadata'
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.warning("Table creation/migration may have issues", error=str(e))
        
        self.graph = nx.DiGraph()
        self._load_from_db()
        
        self.embedding_provider = get_embedding_provider()
        logger.info("NetworkX+Postgres graph store initialized")

    def _load_from_db(self) -> None:
        """Load graph from Postgres into NetworkX."""
        session = self.SessionLocal()
        try:
            nodes = session.query(NodeTable).all()
            for node in nodes:
                self.graph.add_node(
                    node.node_id,
                    name=node.name,
                    type=node.type,
                    aliases=node.aliases or [],
                    metadata=node.node_metadata or {},
                )
            
            edges = session.query(EdgeTable).all()
            for edge in edges:
                self.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation_type=edge.relation_type,
                    confidence=edge.confidence,
                    provenance=edge.provenance or [],
                    edge_id=edge.edge_id,
                )
            
            logger.info("Loaded graph from database", nodes=len(nodes), edges=len(edges))
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    async def add_node(self, node: GraphNode) -> None:
        """Add or update a node."""
        # Add to NetworkX
        self.graph.add_node(
            node.node_id,
            name=node.name,
            type=node.type,
            aliases=node.aliases,
            metadata=node.metadata,
        )
        
        # Persist to Postgres
        session = self.get_session()
        try:
            embedding = await self.embedding_provider.embed([node.name])
            
            # SQLAlchemy JSON columns accept Python objects directly
            db_node = NodeTable(
                node_id=node.node_id,
                name=node.name,
                type=node.type,
                aliases=node.aliases,
                node_metadata=node.metadata,
                embedding=embedding[0],
            )
            session.merge(db_node)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to persist node", node_id=node.node_id, error=str(e))
            raise
        finally:
            session.close()

    async def add_edge(self, edge: GraphEdge) -> None:
        """Add or update an edge."""
        # Add to NetworkX
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type,
            confidence=edge.confidence,
            provenance=edge.provenance,
            edge_id=edge.edge_id,
        )
        
        # Persist to Postgres
        session = self.get_session()
        try:
            # SQLAlchemy JSON columns accept Python objects directly
            db_edge = EdgeTable(
                edge_id=edge.edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation_type=edge.relation_type,
                confidence=edge.confidence,
                provenance=edge.provenance,
            )
            session.merge(db_edge)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to persist edge", edge_id=edge.edge_id, error=str(e))
            raise
        finally:
            session.close()

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        if node_id not in self.graph:
            return None
        
        data = self.graph.nodes[node_id]
        return GraphNode(
            node_id=node_id,
            name=data["name"],
            type=data["type"],
            aliases=data.get("aliases", []),
            metadata=data.get("metadata", {}),
        )

    async def get_neighbors(
        self, node_id: str, relation_types: Optional[List[str]] = None, min_confidence: float = 0.0, max_hops: int = 1
    ) -> List[GraphNode]:
        """Get neighboring nodes via k-hop expansion."""
        if node_id not in self.graph:
            return []
        
        visited = {node_id}
        current_level = {node_id}
        neighbors = []
        
        for hop in range(max_hops):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.successors(node):
                    if neighbor in visited:
                        continue
                    
                    edge_data = self.graph.edges[node, neighbor]
                    if relation_types and edge_data.get("relation_type") not in relation_types:
                        continue
                    if edge_data.get("confidence", 0.0) < min_confidence:
                        continue
                    
                    visited.add(neighbor)
                    next_level.add(neighbor)
                    neighbors.append(await self.get_node(neighbor))
            
            current_level = next_level
            if not current_level:
                break
        
        return neighbors

    async def get_connected_chunks(
        self, node_ids: List[str], max_hops: int = 2, min_confidence: float = 0.0
    ) -> List[str]:
        """Get chunk IDs connected to nodes via graph traversal."""
        chunk_ids = set()
        
        for node_id in node_ids:
            neighbors = await self.get_neighbors(node_id, min_confidence=min_confidence, max_hops=max_hops)
            # Collect chunks from node metadata and edge provenance
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                if "chunk_ids" in node_data.get("metadata", {}):
                    chunk_ids.update(node_data["metadata"]["chunk_ids"])
            
            # Get chunks from edge provenance
            for neighbor in neighbors:
                if node_id in self.graph and neighbor.node_id in self.graph:
                    if self.graph.has_edge(node_id, neighbor.node_id):
                        edge_data = self.graph.edges[node_id, neighbor.node_id]
                        provenance = edge_data.get("provenance", [])
                        for prov in provenance:
                            if isinstance(prov, dict) and "chunk_id" in prov:
                                chunk_ids.add(prov["chunk_id"])
        
        return list(chunk_ids)

    async def search_nodes(self, query: str, top_k: int = 10) -> List[GraphNode]:
        """Search nodes by name/alias using embedding similarity."""
        query_embedding = await self.embedding_provider.embed([query])
        query_vec = query_embedding[0]
        
        session = self.get_session()
        try:
            nodes = session.query(NodeTable).all()
            similarities = []
            
            for node in nodes:
                if node.embedding:
                    # Handle both Python objects and JSON strings
                    node_vec = node.embedding if isinstance(node.embedding, list) else json.loads(node.embedding) if isinstance(node.embedding, str) else []
                    if node_vec:
                        import numpy as np
                        similarity = np.dot(query_vec, node_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(node_vec))
                        similarities.append((similarity, node.node_id))
            
            # Sort by similarity and return top_k
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_node_ids = [node_id for _, node_id in similarities[:top_k]]
            
            results = []
            for node_id in top_node_ids:
                node = await self.get_node(node_id)
                if node:
                    results.append(node)
            
            return results
        finally:
            session.close()


