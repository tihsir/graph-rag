"""Vector store implementations: FAISS (default) and Qdrant."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import json

from ..core.providers import get_embedding_provider
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add(self, chunk_id: str, embedding: List[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors. Returns list of (chunk_id, score, metadata)."""
        pass

    @abstractmethod
    async def get(self, chunk_id: str) -> Optional[Tuple[List[float], dict]]:
        """Get vector and metadata by chunk_id."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store."""

    def __init__(self, index_path: Optional[str] = None, dimension: int = 1536):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")
        
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.dimension = dimension
        self.index = None
        self.id_to_index = {}
        self.index_to_id = []
        self.metadata_store = {}
        
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        import faiss
        
        if self.index_path.exists() and self.index_path.is_file():
            self.index = faiss.read_index(str(self.index_path))
            # Load metadata
            metadata_path = self.index_path.parent / f"{self.index_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    self.id_to_index = data.get("id_to_index", {})
                    self.index_to_id = data.get("index_to_id", [])
                    self.metadata_store = data.get("metadata_store", {})
            logger.info("Loaded FAISS index", path=str(self.index_path), vectors=len(self.index_to_id))
        else:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            logger.info("Created new FAISS index")

    def _save_index(self) -> None:
        """Save index to disk."""
        if self.index:
            import faiss
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            metadata_path = self.index_path.parent / f"{self.index_path.stem}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                    "metadata_store": self.metadata_store,
                }, f)
            logger.info("Saved FAISS index", path=str(self.index_path))

    async def add(self, chunk_id: str, embedding: List[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        if chunk_id in self.id_to_index:
            # Update existing
            idx = self.id_to_index[chunk_id]
            # FAISS doesn't support updates easily, so we'll add a new entry
            # In production, use IndexIDMap and maintain a deletion list
            pass
        
        # Normalize embedding for cosine similarity
        embedding_array = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        idx = len(self.index_to_id)
        self.index.add(embedding_array)
        self.id_to_index[chunk_id] = idx
        self.index_to_id.append(chunk_id)
        self.metadata_store[chunk_id] = metadata
        
        self._save_index()

    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors."""
        if len(self.index_to_id) == 0:
            return []
        
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_array)
        if norm > 0:
            query_array = query_array / norm
        
        scores, indices = self.index.search(query_array, min(top_k, len(self.index_to_id)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.index_to_id):
                chunk_id = self.index_to_id[idx]
                metadata = self.metadata_store.get(chunk_id, {})
                results.append((chunk_id, float(score), metadata))
        
        return results

    async def get(self, chunk_id: str) -> Optional[Tuple[List[float], dict]]:
        """Get vector by chunk_id (not fully supported in FAISS without IndexIDMap)."""
        if chunk_id not in self.id_to_index:
            return None
        metadata = self.metadata_store.get(chunk_id, {})
        # FAISS doesn't easily support retrieval by ID without IndexIDMap
        return None, metadata


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store."""

    def __init__(self, url: Optional[str] = None, collection_name: str = "graphrag"):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")
        
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(url=self.url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        from qdrant_client.models import Distance, VectorParams
        
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection", collection=self.collection_name)

    async def add(self, chunk_id: str, embedding: List[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        from qdrant_client.models import PointStruct
        
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload=metadata,
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )
        
        return [
            (str(result.id), result.score, result.payload or {})
            for result in results
        ]

    async def get(self, chunk_id: str) -> Optional[Tuple[List[float], dict]]:
        """Get vector by chunk_id."""
        result = self.client.retrieve(collection_name=self.collection_name, ids=[chunk_id])
        if not result:
            return None
        
        point = result[0]
        return point.vector, point.payload or {}


def get_vector_store() -> VectorStore:
    """Factory function to get vector store based on config."""
    if settings.vector_store == "faiss":
        # Get dimension based on embedding provider
        if settings.embedding_provider == "local":
            # Try to get dimension from local model
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(settings.local_embedding_model)
                dimension = model.get_sentence_embedding_dimension()
            except:
                dimension = 768  # Default for most local models (all-mpnet-base-v2)
        else:
            dimension = 1536  # Default for OpenAI embeddings
        return FAISSVectorStore(dimension=dimension)
    elif settings.vector_store == "qdrant":
        return QdrantVectorStore()
    else:
        raise ValueError(f"Unknown vector store: {settings.vector_store}")


# Global vector store instance (lazy initialization)
_vector_store = None

def _get_vector_store() -> VectorStore:
    """Get global vector store (lazy initialization)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store

# For backwards compatibility - provide as a property-like accessor
vector_store = type('VectorStoreProxy', (), {
    '__getattr__': lambda self, name: getattr(_get_vector_store(), name),
    'add': lambda self, *args, **kwargs: _get_vector_store().add(*args, **kwargs),
    'search': lambda self, *args, **kwargs: _get_vector_store().search(*args, **kwargs),
    'get': lambda self, *args, **kwargs: _get_vector_store().get(*args, **kwargs),
})()

