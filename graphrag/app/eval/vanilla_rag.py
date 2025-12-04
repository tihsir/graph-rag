"""Vanilla RAG implementation for baseline comparison with GraphRAG."""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..core.providers import get_embedding_provider, get_llm_provider
from ..core.logging import get_logger
from ..retrieval.vector_store import _get_vector_store
from ..data.metadata_store import metadata_store

logger = get_logger(__name__)


@dataclass
class VanillaRAGResult:
    """Result from vanilla RAG query."""
    answer: str
    retrieved_chunks: List[Tuple[str, str, Dict[str, Any]]]  # (chunk_id, text, metadata)
    latencies: Dict[str, float]


class VanillaRAG:
    """
    Pure vector-based RAG without graph augmentation.
    
    This serves as a baseline for comparing against GraphRAG.
    Pipeline: Query → Embed → Vector Search → (Optional Rerank) → Generate
    """
    
    def __init__(self):
        self._embedding_provider = None
        self._llm_provider = None
        self._vector_store = None
        self._reranker = None
    
    @property
    def embedding_provider(self):
        """Lazy initialization of embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider
    
    @property
    def llm_provider(self):
        """Lazy initialization of LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider()
        return self._llm_provider
    
    @property
    def vector_store(self):
        """Lazy initialization of vector store."""
        if self._vector_store is None:
            self._vector_store = _get_vector_store()
        return self._vector_store
    
    @property
    def reranker(self):
        """Lazy initialization of reranker."""
        if self._reranker is None:
            try:
                from ..retrieval.ranker import _get_reranker
                self._reranker = _get_reranker()
            except Exception:
                self._reranker = None
        return self._reranker
    
    async def retrieve(
        self,
        query: str,
        k: int = 10,
        rerank: bool = False
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Retrieve relevant chunks using pure vector search.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            rerank: Whether to apply reranking
        
        Returns:
            List of (chunk_id, text, metadata) tuples
        """
        import time
        
        # Embed query
        embed_start = time.time()
        query_embedding = await self.embedding_provider.embed([query])
        embed_time = time.time() - embed_start
        
        # Vector search (retrieve more if reranking)
        search_start = time.time()
        search_k = k * 2 if rerank else k
        vector_results = await self.vector_store.search(query_embedding[0], top_k=search_k)
        search_time = time.time() - search_start
        
        # Load chunk texts
        chunk_ids = [r[0] for r in vector_results]
        chunks = metadata_store.get_chunks_by_ids(chunk_ids)
        
        # Build results with scores
        results = []
        score_map = {r[0]: r[1] for r in vector_results}
        for chunk in chunks:
            metadata = {
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "score": score_map.get(chunk.chunk_id, 0.0)
            }
            results.append((chunk.chunk_id, chunk.text, metadata))
        
        # Sort by score
        results.sort(key=lambda x: x[2].get("score", 0), reverse=True)
        
        # Optional reranking
        if rerank and self.reranker and results:
            rerank_start = time.time()
            results = await self.reranker.rerank(query, results, top_k=k)
            rerank_time = time.time() - rerank_start
            logger.debug("Reranking completed", rerank_time=rerank_time)
        else:
            results = results[:k]
        
        return results
    
    async def query(
        self,
        query: str,
        k: int = 10,
        rerank: bool = False,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> VanillaRAGResult:
        """
        Execute full vanilla RAG pipeline.
        
        Args:
            query: User query
            k: Number of chunks to use in context
            rerank: Whether to apply reranking
            max_tokens: Max tokens for generation
            temperature: Generation temperature
        
        Returns:
            VanillaRAGResult with answer and retrieved chunks
        """
        import time
        latencies = {}
        
        # Retrieve
        retrieve_start = time.time()
        chunks = await self.retrieve(query, k=k, rerank=rerank)
        latencies["retrieval"] = time.time() - retrieve_start
        
        if not chunks:
            return VanillaRAGResult(
                answer="I couldn't find relevant information to answer your question.",
                retrieved_chunks=[],
                latencies=latencies
            )
        
        # Build context
        context_parts = []
        for chunk_id, text, metadata in chunks:
            doc_id = metadata.get("doc_id", "unknown")
            context_parts.append(f"[Source: {doc_id}#{chunk_id}]\n{text}")
        context = "\n\n".join(context_parts)
        
        # Generate answer
        generate_start = time.time()
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain enough information, say so.
Include inline citations in the format [Source: doc#chunk] when referencing specific information."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        answer = await self.llm_provider.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        latencies["generation"] = time.time() - generate_start
        latencies["total"] = latencies["retrieval"] + latencies["generation"]
        
        return VanillaRAGResult(
            answer=answer,
            retrieved_chunks=chunks,
            latencies=latencies
        )
    
    async def retrieve_only(
        self,
        query: str,
        k: int = 10,
        rerank: bool = False
    ) -> Tuple[List[str], List[Tuple[str, str, Dict[str, Any]]]]:
        """
        Retrieve chunks without generating an answer.
        Useful for evaluation of retrieval quality.
        
        Returns:
            Tuple of (chunk_ids, full_chunks)
        """
        chunks = await self.retrieve(query, k=k, rerank=rerank)
        chunk_ids = [c[0] for c in chunks]
        return chunk_ids, chunks


# Global instance
vanilla_rag = VanillaRAG()


async def vanilla_rag_query(
    query: str,
    k: int = 10,
    rerank: bool = False,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> VanillaRAGResult:
    """Convenience function for vanilla RAG query."""
    return await vanilla_rag.query(
        query=query,
        k=k,
        rerank=rerank,
        max_tokens=max_tokens,
        temperature=temperature
    )


async def vanilla_rag_retrieve(
    query: str,
    k: int = 10,
    rerank: bool = False
) -> List[str]:
    """Convenience function to get just chunk IDs."""
    chunk_ids, _ = await vanilla_rag.retrieve_only(query, k=k, rerank=rerank)
    return chunk_ids

