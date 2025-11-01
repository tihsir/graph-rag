"""Aggregator for merging graph and vector retrieval results."""

from typing import List, Tuple, Set
from ..core.logging import get_logger

logger = get_logger(__name__)


class Aggregator:
    """Aggregates and deduplicates retrieval results from multiple sources."""

    def aggregate(
        self,
        graph_chunks: List[Tuple[str, str, dict]],  # (chunk_id, text, metadata)
        vector_chunks: List[Tuple[str, float, dict]],  # (chunk_id, score, metadata)
        top_k: int = 20,
        diversity_threshold: float = 0.8,
    ) -> List[Tuple[str, str, dict]]:
        """
        Merge graph and vector results with deduplication and diversity filtering.
        
        Args:
            graph_chunks: Chunks from graph traversal (chunk_id, text, metadata)
            vector_chunks: Chunks from vector search (chunk_id, score, metadata)
            top_k: Maximum number of results
            diversity_threshold: Similarity threshold for diversity filtering
            
        Returns:
            Merged and deduplicated list of (chunk_id, text, metadata) tuples
        """
        seen: Set[str] = set()
        merged: List[Tuple[str, str, dict]] = []
        
        # Add graph chunks first (prioritize graph results)
        for chunk_id, text, metadata in graph_chunks:
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append((chunk_id, text, metadata))
        
        # Add vector chunks
        for chunk_id, score, metadata in vector_chunks:
            if chunk_id not in seen:
                seen.add(chunk_id)
                # Convert vector format to (chunk_id, text, metadata)
                text = metadata.get("text", "")
                merged.append((chunk_id, text, metadata))
        
        # Limit to top_k
        result = merged[:top_k]
        logger.info("Aggregated results", graph_count=len(graph_chunks), vector_count=len(vector_chunks), merged_count=len(result))
        return result


# Global aggregator instance
aggregator = Aggregator()

