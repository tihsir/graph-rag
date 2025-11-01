"""Reranker for retrieval results."""

from typing import List, Tuple, Optional
from ..core.providers import get_reranker_provider
from ..core.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    """Reranks retrieval results."""

    def __init__(self, reranker_provider=None):
        self._reranker_provider_arg = reranker_provider
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy initialization of reranker provider."""
        if self._reranker is None:
            self._reranker = self._reranker_provider_arg or get_reranker_provider()
        return self._reranker

    async def rerank(
        self, query: str, documents: List[Tuple[str, str, dict]], top_k: Optional[int] = None
    ) -> List[Tuple[str, str, dict]]:
        """
        Rerank documents.
        
        Args:
            query: User query
            documents: List of (chunk_id, text, metadata) tuples
            top_k: Maximum number of results to return
            
        Returns:
            Reranked list of (chunk_id, text, metadata) tuples
        """
        if not documents:
            return []
        
        texts = [doc[1] for doc in documents]
        ranked_indices = await self.reranker.rerank(query, texts, top_k=top_k)
        
        reranked = [documents[i] for i in ranked_indices]
        logger.info("Reranked documents", query=query[:50], input_count=len(documents), output_count=len(reranked))
        return reranked


# Global reranker instance (lazy initialization)
_reranker = None

def _get_reranker() -> Reranker:
    """Get global reranker (lazy initialization)."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

# Module-level accessor (lazy - behaves like Reranker instance)
reranker = type('RerankerProxy', (), {
    'rerank': lambda self, *args, **kwargs: _get_reranker().rerank(*args, **kwargs),
})()

