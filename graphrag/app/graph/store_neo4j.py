"""Neo4j graph store implementation."""

from typing import List, Dict, Optional
from neo4j import AsyncGraphDatabase

from .store_base import GraphStore, GraphNode, GraphEdge
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class Neo4jStore(GraphStore):
    """Neo4j graph store implementation."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info("Neo4j graph store initialized", uri=self.uri)

    async def close(self) -> None:
        """Close the Neo4j driver."""
        await self.driver.close()

    async def add_node(self, node: GraphNode) -> None:
        """Add or update a node."""
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (n:Entity {id: $node_id})
                SET n.name = $name,
                    n.type = $type,
                    n.aliases = $aliases,
                    n.metadata = $metadata
                """,
                node_id=node.node_id,
                name=node.name,
                type=node.type,
                aliases=node.aliases,
                metadata=node.metadata,
            )

    async def add_edge(self, edge: GraphEdge) -> None:
        """Add or update an edge."""
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (a:Entity {id: $source_id})
                MATCH (b:Entity {id: $target_id})
                MERGE (a)-[r:RELATES {id: $edge_id}]->(b)
                SET r.type = $relation_type,
                    r.confidence = $confidence,
                    r.provenance = $provenance
                """,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_id=edge.edge_id,
                relation_type=edge.relation_type,
                confidence=edge.confidence,
                provenance=edge.provenance,
            )

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        async with self.driver.session() as session:
            result = await session.run(
                "MATCH (n:Entity {id: $node_id}) RETURN n",
                node_id=node_id,
            )
            record = await result.single()
            if not record:
                return None
            
            node_data = record["n"]
            return GraphNode(
                node_id=node_data["id"],
                name=node_data.get("name", ""),
                type=node_data.get("type", "other"),
                aliases=node_data.get("aliases", []),
                metadata=node_data.get("metadata", {}),
            )

    async def get_neighbors(
        self, node_id: str, relation_types: Optional[List[str]] = None, min_confidence: float = 0.0, max_hops: int = 1
    ) -> List[GraphNode]:
        """Get neighboring nodes via k-hop expansion."""
        if relation_types:
            rel_filter = f"r.type IN {relation_types} AND"
        else:
            rel_filter = ""
        
        query = f"""
        MATCH path = (start:Entity {{id: $node_id}})-[r:RELATES*1..{max_hops}]->(n:Entity)
        WHERE {rel_filter} ALL(rel IN relationships(path) WHERE rel.confidence >= $min_conf)
        RETURN DISTINCT n
        LIMIT 100
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                node_id=node_id,
                min_conf=min_confidence,
            )
            neighbors = []
            async for record in result:
                node_data = record["n"]
                neighbors.append(
                    GraphNode(
                        node_id=node_data["id"],
                        name=node_data.get("name", ""),
                        type=node_data.get("type", "other"),
                        aliases=node_data.get("aliases", []),
                        metadata=node_data.get("metadata", {}),
                    )
                )
            return neighbors

    async def get_connected_chunks(
        self, node_ids: List[str], max_hops: int = 2, min_confidence: float = 0.0
    ) -> List[str]:
        """Get chunk IDs connected to nodes via graph traversal."""
        chunk_ids = set()
        
        async with self.driver.session() as session:
            for node_id in node_ids:
                result = await session.run(
                    f"""
                    MATCH path = (start:Entity {{id: $node_id}})-[r:RELATES*1..{max_hops}]->(n:Entity)
                    WHERE ALL(rel IN relationships(path) WHERE rel.confidence >= $min_conf)
                    UNWIND relationships(path) AS rel
                    UNWIND rel.provenance AS prov
                    RETURN prov.chunk_id AS chunk_id
                    """,
                    node_id=node_id,
                    min_conf=min_confidence,
                )
                async for record in result:
                    if record.get("chunk_id"):
                        chunk_ids.add(record["chunk_id"])
            
            # Also get chunks from node metadata
            for node_id in node_ids:
                node = await self.get_node(node_id)
                if node and "chunk_ids" in node.metadata:
                    chunk_ids.update(node.metadata["chunk_ids"])
        
        return list(chunk_ids)

    async def search_nodes(self, query: str, top_k: int = 10) -> List[GraphNode]:
        """Search nodes by name/alias (text similarity)."""
        # Simple text search - in production, use full-text search or embeddings
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (n:Entity)
                WHERE n.name CONTAINS $query OR ANY(alias IN n.aliases WHERE alias CONTAINS $query)
                RETURN n
                LIMIT $top_k
                """,
                query=query.lower(),
                top_k=top_k,
            )
            nodes = []
            async for record in result:
                node_data = record["n"]
                nodes.append(
                    GraphNode(
                        node_id=node_data["id"],
                        name=node_data.get("name", ""),
                        type=node_data.get("type", "other"),
                        aliases=node_data.get("aliases", []),
                        metadata=node_data.get("metadata", {}),
                    )
                )
            return nodes

