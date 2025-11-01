"""Tests for graph traversal."""

import pytest
from graphrag.app.graph.traversal import k_hop_expansion
from graphrag.app.graph.store_base import GraphStore, GraphNode


@pytest.mark.asyncio
async def test_k_hop_expansion_empty():
    """Test k-hop expansion with empty graph."""
    # Mock graph store
    class MockGraphStore(GraphStore):
        async def add_node(self, node):
            pass
        async def add_edge(self, edge):
            pass
        async def get_node(self, node_id):
            return None
        async def get_neighbors(self, node_id, relation_types=None, min_confidence=0.0, max_hops=1):
            return []
        async def get_connected_chunks(self, node_ids, max_hops=2, min_confidence=0.0):
            return []
        async def search_nodes(self, query, top_k=10):
            return []
    
    store = MockGraphStore()
    result = await k_hop_expansion(store, [], max_hops=2)
    assert result == []


@pytest.mark.asyncio
async def test_k_hop_expansion_with_nodes():
    """Test k-hop expansion with mock nodes."""
    class MockGraphStore(GraphStore):
        def __init__(self):
            self.nodes_data = {
                "node1": GraphNode(node_id="node1", name="Node 1", type="person"),
                "node2": GraphNode(node_id="node2", name="Node 2", type="org"),
            }
        
        async def add_node(self, node):
            pass
        async def add_edge(self, edge):
            pass
        async def get_node(self, node_id):
            return self.nodes_data.get(node_id)
        async def get_neighbors(self, node_id, relation_types=None, min_confidence=0.0, max_hops=1):
            if node_id == "node1":
                return [self.nodes_data["node2"]]
            return []
        async def get_connected_chunks(self, node_ids, max_hops=2, min_confidence=0.0):
            return []
        async def search_nodes(self, query, top_k=10):
            return []
    
    store = MockGraphStore()
    result = await k_hop_expansion(store, ["node1"], max_hops=1)
    assert len(result) >= 1
    assert any(n.node_id == "node1" for n in result)

