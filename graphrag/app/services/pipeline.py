"""Pipeline service for orchestration: ingest → embed → extract graph."""

# Import exceptiongroup early for Python 3.9 compatibility
# This must be done before importing anyio-dependent packages
import sys
if sys.version_info < (3, 11):
    try:
        from exceptiongroup import ExceptionGroup
        # Monkey-patch into builtins for anyio compatibility
        if not hasattr(sys.modules.get('builtins', object()), 'ExceptionGroup'):
            import builtins
            builtins.ExceptionGroup = ExceptionGroup
    except ImportError:
        pass

from typing import List, Optional
import asyncio

from ..data.ingestion import ingestor, Document
from ..data.chunking import chunker, Chunk
from ..data.metadata_store import metadata_store
from ..graph.extraction import entity_extractor, relation_extractor
from ..graph.canonicalize import canonicalizer
from ..graph.store_base import GraphNode, GraphEdge
from ..core.providers import get_embedding_provider, get_llm_provider
from ..core.config import settings
from ..core.logging import get_logger
from ..retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


def get_graph_store():
    """Factory to get graph store based on config."""
    from ..graph.store_nx_pg import NetworkXPostgresStore
    from ..graph.store_neo4j import Neo4jStore
    from ..core.config import settings
    
    if settings.graph_store == "nx_pg":
        return NetworkXPostgresStore()
    elif settings.graph_store == "neo4j":
        return Neo4jStore()
    else:
        raise ValueError(f"Unknown graph store: {settings.graph_store}")


class Pipeline:
    """Orchestrates the ingestion and graph building pipeline."""

    def __init__(self):
        self._embedding_provider = None
        self._graph_store = None
        self._vector_store = None
    
    @property
    def embedding_provider(self):
        """Lazy initialization of embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider
    
    @property
    def graph_store(self):
        """Lazy initialization of graph store."""
        if self._graph_store is None:
            self._graph_store = get_graph_store()
        return self._graph_store
    
    @property
    def vector_store(self):
        """Lazy initialization of vector store."""
        if self._vector_store is None:
            from ..retrieval.vector_store import _get_vector_store
            self._vector_store = _get_vector_store()
        return self._vector_store

    async def ingest(
        self,
        paths: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Ingest documents, chunk, embed, and persist."""
        documents = await ingestor.ingest(paths=paths, urls=urls, tags=tags)
        doc_ids = []
        
        for doc in documents:
            try:
                # Store document
                metadata_store.add_document(
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    source_url=doc.source_url,
                    content=doc.content,
                    metadata=doc.metadata,
                )
                
                # Chunk document
                chunks = chunker.chunk(doc.content, doc.doc_id)
                
                # Embed and store chunks in batches
                batch_size = settings.batch_size
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    texts = [chunk.text for chunk in batch]
                    embeddings = await self.embedding_provider.embed(texts)
                    
                    for chunk, embedding in zip(batch, embeddings):
                        # Store chunk metadata
                        metadata_store.add_chunk(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            text=chunk.text,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                            section=chunk.section,
                            page=chunk.page,
                            token_count=chunk.token_count,
                            embedding=embedding,
                        )
                        
                        # Store in vector store
                        await self.vector_store.add(
                            chunk_id=chunk.chunk_id,
                            embedding=embedding,
                            metadata={
                                "doc_id": chunk.doc_id,
                                "text": chunk.text,
                                "section": chunk.section,
                                "page": chunk.page,
                            },
                        )
                
                doc_ids.append(doc.doc_id)
                logger.info("Ingested document", doc_id=doc.doc_id, chunks=len(chunks))
            except Exception as e:
                logger.error("Failed to ingest document", doc_id=doc.doc_id, error=str(e))
        
        return doc_ids

    async def rebuild_graph(self, min_conf: float = 0.4, batch_size: int = 64) -> dict:
        """Rebuild knowledge graph from all chunks.
        
        Note: This requires an LLM API key (OpenAI/Anthropic) for entity and relation extraction.
        """
        from ..data.metadata_store import ChunkModel, metadata_store
        
        # Check if LLM is available before processing
        try:
            test_provider = get_llm_provider()
            llm_available = True
        except (ValueError, Exception) as e:
            if "API key" in str(e) or "api_key" in str(e).lower():
                logger.error("LLM API key required for graph building", error=str(e))
                return {
                    "entities": 0, 
                    "relations": 0,
                    "error": "LLM API key required",
                    "message": "Graph building requires an LLM API key (OpenAI or Anthropic) for entity and relation extraction. Set OPENAI_API_KEY environment variable."
                }
            raise
        
        # Get all chunks that haven't been processed
        # For simplicity, we'll process all chunks
        session = metadata_store.get_session()
        all_chunks = session.query(ChunkModel).all()
        session.close()
        
        if not all_chunks:
            logger.warning("No chunks found in database")
            return {
                "entities": 0,
                "relations": 0,
                "message": "No chunks to process. Run ingestion first."
            }
        
        total_entities = 0
        total_relations = 0
        
        # Add a small delay between chunks to help with rate limiting
        # (rate limiter in provider handles most of this, but this helps)
        chunk_delay = 0.1  # 100ms between chunks
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            for chunk_idx, chunk in enumerate(batch):
                # Add small delay between chunks (except first in batch)
                if chunk_idx > 0:
                    await asyncio.sleep(chunk_delay)
                try:
                    # Extract entities (lazy extractor access)
                    from ..graph.extraction import _get_entity_extractor, _get_relation_extractor
                    from ..graph.canonicalize import _get_canonicalizer
                    
                    entity_ext = _get_entity_extractor()
                    relation_ext = _get_relation_extractor()
                    canonical = _get_canonicalizer()
                    
                    entities = await entity_ext.extract(
                        chunk.text, chunk.doc_id, chunk.chunk_id, chunk.start_char
                    )
                    
                    # Extract relations
                    relations = await relation_ext.extract(
                        chunk.text, entities, chunk.doc_id, chunk.chunk_id, chunk.start_char
                    )
                    
                    # Filter by confidence
                    entities = [e for e in entities if e.confidence >= min_conf]
                    relations = [r for r in relations if r.confidence >= min_conf]
                    
                    # Canonicalize entities
                    canonical_groups = await canonical.canonicalize(entities)
                    
                    # Create graph nodes and edges
                    node_map = {}
                    for canonical_name, entity_list in canonical_groups.items():
                        node_id = f"entity_{canonical_name.lower().replace(' ', '_')}"
                        
                        # Merge aliases
                        all_aliases = set()
                        for entity in entity_list:
                            all_aliases.add(entity.name)
                            all_aliases.update(entity.aliases)
                        all_aliases.discard(canonical_name)
                        
                        # Collect chunk IDs for provenance
                        chunk_ids = list(set([e.chunk_id for e in entity_list]))
                        
                        node = GraphNode(
                            node_id=node_id,
                            name=canonical_name,
                            type=entity_list[0].type,
                            aliases=list(all_aliases),
                            metadata={"chunk_ids": chunk_ids},
                        )
                        
                        await self.graph_store.add_node(node)
                        node_map[canonical_name] = node_id
                        total_entities += 1
                    
                    # Create edges
                    for relation in relations:
                        # Find canonical nodes for relation args
                        arg1_node_id = None
                        arg2_node_id = None
                        
                        for canonical_name, node_id in node_map.items():
                            # Check if relation args match canonical name or any aliases
                            canonical_lower = canonical_name.lower()
                            if relation.arg1.lower() == canonical_lower:
                                arg1_node_id = node_id
                            if relation.arg2.lower() == canonical_lower:
                                arg2_node_id = node_id
                            
                            # Also check aliases from the entity groups
                            for entity in canonical_groups[canonical_name]:
                                if relation.arg1.lower() in [a.lower() for a in entity.aliases] + [entity.name.lower()]:
                                    arg1_node_id = node_id
                                if relation.arg2.lower() in [a.lower() for a in entity.aliases] + [entity.name.lower()]:
                                    arg2_node_id = node_id
                        
                        if arg1_node_id and arg2_node_id:
                            edge_id = f"edge_{arg1_node_id}_{relation.relation}_{arg2_node_id}"
                            edge = GraphEdge(
                                edge_id=edge_id,
                                source_id=arg1_node_id,
                                target_id=arg2_node_id,
                                relation_type=relation.relation,
                                confidence=relation.confidence,
                                provenance=[{
                                    "doc_id": relation.doc_id,
                                    "chunk_id": relation.chunk_id,
                                    "char_spans": [(relation.char_start, relation.char_end)],
                                }],
                            )
                            await self.graph_store.add_edge(edge)
                            total_relations += 1
                    
                except Exception as e:
                    logger.error("Failed to process chunk for graph", chunk_id=chunk.chunk_id, error=str(e))
        
        logger.info("Graph rebuild completed", entities=total_entities, relations=total_relations)
        return {"entities": total_entities, "relations": total_relations}


# Global pipeline instance
pipeline = Pipeline()

