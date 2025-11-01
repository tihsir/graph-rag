"""Entity canonicalization with embedding-based alias merging."""

from typing import List, Dict, Set
import numpy as np

from ..core.providers import get_embedding_provider
from ..core.logging import get_logger
from .extraction import Entity

logger = get_logger(__name__)


class Canonicalizer:
    """Canonicalize entities by normalizing text and merging aliases via embeddings."""

    def __init__(self, embedding_provider=None, similarity_threshold: float = 0.85):
        self._embedding_provider_arg = embedding_provider
        self._embedding_provider = None
        self.similarity_threshold = similarity_threshold
    
    @property
    def embedding_provider(self):
        """Lazy initialization of embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = self._embedding_provider_arg or get_embedding_provider()
        return self._embedding_provider

    def normalize_text(self, text: str) -> str:
        """Normalize entity text (lowercase, strip, remove special chars)."""
        return text.lower().strip()

    async def canonicalize(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """Group entities by canonical name using embedding similarity."""
        if not entities:
            return {}
        
        # Normalize all entity names
        normalized_map = {}
        for entity in entities:
            normalized = self.normalize_text(entity.name)
            if normalized not in normalized_map:
                normalized_map[normalized] = []
            normalized_map[normalized].append(entity)
        
        # For each normalized group, check if aliases should merge groups
        canonical_groups: Dict[str, List[Entity]] = {}
        
        # Get embeddings for all unique entity names
        unique_names = list(normalized_map.keys())
        embeddings = await self.embedding_provider.embed(unique_names)
        name_to_embedding = {name: emb for name, emb in zip(unique_names, embeddings)}
        
        # Cluster by similarity
        used_names: Set[str] = set()
        for name, entity_list in normalized_map.items():
            if name in used_names:
                continue
            
            # Find similar entities
            canonical_name = name
            group = entity_list.copy()
            
            name_emb = name_to_embedding[name]
            
            for other_name, other_list in normalized_map.items():
                if other_name == name or other_name in used_names:
                    continue
                
                other_emb = name_to_embedding[other_name]
                similarity = np.dot(name_emb, other_emb) / (np.linalg.norm(name_emb) * np.linalg.norm(other_emb))
                
                if similarity >= self.similarity_threshold:
                    group.extend(other_list)
                    used_names.add(other_name)
                    # Use shorter name as canonical
                    if len(other_name) < len(canonical_name):
                        canonical_name = other_name
            
            canonical_groups[canonical_name] = group
            used_names.add(name)
        
        # Also merge based on aliases
        final_groups: Dict[str, List[Entity]] = {}
        processed = set()
        
        for canonical, group in canonical_groups.items():
            if canonical in processed:
                continue
            
            merged_group = group.copy()
            canonical_candidates = {canonical}
            
            # Collect all aliases from this group
            for entity in group:
                for alias in entity.aliases:
                    normalized_alias = self.normalize_text(alias)
                    canonical_candidates.add(normalized_alias)
            
            # Check if any other group's canonical name matches aliases
            for other_canonical, other_group in canonical_groups.items():
                if other_canonical in processed or other_canonical == canonical:
                    continue
                
                if other_canonical in canonical_candidates:
                    merged_group.extend(other_group)
                    processed.add(other_canonical)
            
            # Choose best canonical name (shortest, most common)
            best_canonical = min(canonical_candidates, key=lambda x: (len(x), -len(merged_group)))
            final_groups[best_canonical] = merged_group
            processed.add(canonical)
        
        logger.info("Canonicalized entities", input_count=len(entities), output_groups=len(final_groups))
        return final_groups


# Global canonicalizer instance (lazy initialization)
_canonicalizer = None

def _get_canonicalizer() -> Canonicalizer:
    """Get global canonicalizer (lazy initialization)."""
    global _canonicalizer
    if _canonicalizer is None:
        _canonicalizer = Canonicalizer()
    return _canonicalizer

# Module-level accessor (lazy - behaves like Canonicalizer instance)
canonicalizer = type('CanonicalizerProxy', (), {
    'normalize_text': lambda self, text: _get_canonicalizer().normalize_text(text),
    'canonicalize': lambda self, *args, **kwargs: _get_canonicalizer().canonicalize(*args, **kwargs),
    'similarity_threshold': property(lambda self: _get_canonicalizer().similarity_threshold),
})()

