"""Graph traversal utilities for k-hop expansion with filters."""

from typing import List, Set, Optional
from .store_base import GraphStore, GraphNode

from ..core.logging import get_logger

logger = get_logger(__name__)


async def k_hop_expansion(
    graph_store: GraphStore,
    seed_node_ids: List[str],
    max_hops: int = 2,
    relation_types: Optional[List[str]] = None,
    min_confidence: float = 0.0,
    max_nodes: int = 100,
) -> List[GraphNode]:
    """Perform k-hop expansion from seed nodes with filters."""
    if not seed_node_ids:
        return []
    
    visited: Set[str] = set(seed_node_ids)
    current_nodes: Set[str] = set(seed_node_ids)
    all_nodes: List[GraphNode] = []
    
    # Get initial nodes
    for node_id in seed_node_ids:
        node = await graph_store.get_node(node_id)
        if node:
            all_nodes.append(node)
    
    for hop in range(max_hops):
        if len(all_nodes) >= max_nodes:
            break
        
        next_level: Set[str] = set()
        
        for node_id in current_nodes:
            neighbors = await graph_store.get_neighbors(
                node_id,
                relation_types=relation_types,
                min_confidence=min_confidence,
                max_hops=1,  # Single hop per iteration
            )
            
            for neighbor in neighbors:
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    next_level.add(neighbor.node_id)
                    all_nodes.append(neighbor)
                    
                    if len(all_nodes) >= max_nodes:
                        break
        
        if not next_level:
            break
        
        current_nodes = next_level
    
    logger.info("K-hop expansion completed", seeds=len(seed_node_ids), hops=max_hops, nodes=len(all_nodes))
    return all_nodes

