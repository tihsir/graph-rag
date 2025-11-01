"""Entity and relation extraction using LLM with prompts."""

import json
import re
from typing import List, Dict, Optional
from pydantic import BaseModel

from ..core.providers import get_llm_provider
from ..core.logging import get_logger
from ..core.config import settings

logger = get_logger(__name__)


def _clean_json(json_str: str) -> str:
    """Try to fix common JSON issues in LLM responses."""
    # Remove markdown code blocks
    json_str = json_str.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    
    # Remove leading/trailing non-JSON text
    # Find first [ or { and last ] or }
    start_chars = ['[', '{']
    end_chars = [']', '}']
    
    start_idx = -1
    end_idx = -1
    
    for char in start_chars:
        idx = json_str.find(char)
        if idx != -1 and (start_idx == -1 or idx < start_idx):
            start_idx = idx
    
    for char in end_chars:
        idx = json_str.rfind(char)
        if idx != -1 and (end_idx == -1 or idx > end_idx):
            end_idx = idx
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = json_str[start_idx:end_idx+1]
    
    return json_str


def _parse_json_safely(json_str: str, chunk_id: str) -> Optional[List[Dict]]:
    """Parse JSON with multiple fallback strategies."""
    # First try: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Second try: clean and parse
    try:
        cleaned = _clean_json(json_str)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Third try: extract array from text
    try:
        # Find first [ and last ]
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start != -1 and end != -1 and end > start:
            array_str = json_str[start:end+1]
            return json.loads(array_str)
    except json.JSONDecodeError:
        pass
    
    # Fourth try: use regex to extract JSON objects
    try:
        # Find all {...} patterns and try to combine
        obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(obj_pattern, json_str, re.DOTALL)
        if matches:
            # Try to parse each object
            objects = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    objects.append(obj)
                except:
                    continue
            if objects:
                return objects
    except:
        pass
    
    logger.warning("Failed to parse JSON after all attempts", chunk_id=chunk_id, json_preview=json_str[:200])
    return None


class Entity(BaseModel):
    """Extracted entity with provenance."""

    name: str
    type: str  # person, org, location, concept, product, other
    aliases: List[str] = []
    span: List[int]  # [start, end] character positions
    confidence: float
    doc_id: str
    chunk_id: str
    char_start: int
    char_end: int


class Relation(BaseModel):
    """Extracted relation with provenance."""

    relation: str  # works_at, founded_by, part_of, similar_to, causes, references, interacts_with
    arg1: str  # entity name or ID
    arg2: str  # entity name or ID
    confidence: float
    rationale: str
    doc_id: str
    chunk_id: str
    char_start: int
    char_end: int


class EntityExtractor:
    """Extract entities from text using LLM."""

    def __init__(self, llm_provider=None):
        self._llm_provider = llm_provider
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM provider."""
        if self._llm is None:
            self._llm = self._llm_provider or get_llm_provider()
        return self._llm

    def _load_entity_prompt(self) -> str:
        """Load entity extraction prompt from file."""
        from pathlib import Path
        # Try to find prompts directory relative to this file
        current_file = Path(__file__)
        prompt_path = current_file.parent.parent / "prompts" / "entity_link.toml"
        # Fallback to hardcoded prompt if file doesn't exist
        if prompt_path.exists():
            try:
                import tomli
                with open(prompt_path, "rb") as f:
                    data = tomli.load(f)
                    return data.get("system", {}).get("prompt", "")
            except:
                pass
        # Fallback to hardcoded prompt
        return """Identify and canonicalize entities in INPUT text. Output EXACT JSON list:
[{"name": "...", "type": "person|org|location|concept|product|other",
  "aliases": ["..."], "span": [start,end], "confidence": 0..1}]
No extra prose."""

    async def extract(self, text: str, doc_id: str, chunk_id: str, char_start: int = 0) -> List[Entity]:
        """Extract entities from text."""
        prompt = f"""INPUT: {text}

{self._load_entity_prompt()}"""
        
        try:
            response = await self.llm.generate(prompt, max_tokens=500, temperature=0.0)
            # Try to parse JSON from response with error recovery
            json_str = response.strip()
            entities_data = _parse_json_safely(json_str, chunk_id)
            
            if entities_data is None:
                logger.warning("Could not parse entities JSON, skipping chunk", chunk_id=chunk_id)
                return []
            
            entities = []
            
            for e_data in entities_data:
                if "span" in e_data and len(e_data["span"]) == 2:
                    char_span_start = char_start + e_data["span"][0]
                    char_span_end = char_start + e_data["span"][1]
                else:
                    char_span_start = char_start
                    char_span_end = char_start + len(text)
                
                entity = Entity(
                    name=e_data["name"],
                    type=e_data.get("type", "other"),
                    aliases=e_data.get("aliases", []),
                    span=e_data.get("span", [0, len(text)]),
                    confidence=e_data.get("confidence", 0.5),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    char_start=char_span_start,
                    char_end=char_span_end,
                )
                entities.append(entity)
            
            return entities
        except ValueError as e:
            # API key missing - re-raise so caller can handle it
            if "API key" in str(e) or "api_key" in str(e).lower():
                raise  # Let pipeline handle this at the top level
            raise
        except Exception as e:
            logger.error("Entity extraction failed", error=str(e), chunk_id=chunk_id)
            return []


class RelationExtractor:
    """Extract relations from text using LLM."""

    def __init__(self, llm_provider=None):
        self._llm_provider = llm_provider
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM provider."""
        if self._llm is None:
            self._llm = self._llm_provider or get_llm_provider()
        return self._llm

    def _load_relation_prompt(self) -> str:
        """Load relation extraction prompt from file."""
        # Fallback to hardcoded prompt
        return """Given CHUNK and detected ENTITIES, extract binary relations.
Allowed types: works_at, founded_by, part_of, similar_to, causes, references, interacts_with.
Output JSON list:
[{"relation":"...", "arg1":"<entity_name_or_id>", "arg2":"<entity_name_or_id>",
  "confidence":0..1, "rationale":"<=30 tokens"}]
No extra prose."""

    async def extract(
        self, text: str, entities: List[Entity], doc_id: str, chunk_id: str, char_start: int = 0
    ) -> List[Relation]:
        """Extract relations from text given entities."""
        entity_list = ", ".join([f"{e.name} ({e.type})" for e in entities])
        
        prompt = f"""CHUNK: {text}

ENTITIES: {entity_list}

{self._load_relation_prompt()}"""
        
        try:
            response = await self.llm.generate(prompt, max_tokens=500, temperature=0.0)
            # Try to parse JSON from response with error recovery
            json_str = response.strip()
            relations_data = _parse_json_safely(json_str, chunk_id)
            
            if relations_data is None:
                logger.warning("Could not parse relations JSON, skipping chunk", chunk_id=chunk_id)
                return []
            
            relations = []
            
            for r_data in relations_data:
                relation = Relation(
                    relation=r_data["relation"],
                    arg1=r_data["arg1"],
                    arg2=r_data["arg2"],
                    confidence=r_data.get("confidence", 0.5),
                    rationale=r_data.get("rationale", ""),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    char_start=char_start,
                    char_end=char_start + len(text),
                )
                relations.append(relation)
            
            return relations
        except ValueError as e:
            # API key missing - re-raise so caller can handle it
            if "API key" in str(e) or "api_key" in str(e).lower():
                raise  # Let pipeline handle this at the top level
            raise
        except Exception as e:
            logger.error("Relation extraction failed", error=str(e), chunk_id=chunk_id)
            return []


# Global extractors (lazy initialization - only created when accessed)
_entity_extractor = None
_relation_extractor = None

def _get_entity_extractor() -> EntityExtractor:
    """Get global entity extractor (lazy initialization)."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor

def _get_relation_extractor() -> RelationExtractor:
    """Get global relation extractor (lazy initialization)."""
    global _relation_extractor
    if _relation_extractor is None:
        _relation_extractor = RelationExtractor()
    return _relation_extractor

# Module-level accessors (lazy)
class _LazyExtractorAccessor:
    """Lazy accessor for extractors."""
    def __call__(self):
        return _get_entity_extractor()
    
    def extract(self, *args, **kwargs):
        return _get_entity_extractor().extract(*args, **kwargs)

class _LazyRelationAccessor:
    """Lazy accessor for relation extractor."""
    def __call__(self):
        return _get_relation_extractor()
    
    def extract(self, *args, **kwargs):
        return _get_relation_extractor().extract(*args, **kwargs)

# Provide as callable objects that behave like the extractors
entity_extractor = type('EntityExtractorProxy', (), {
    'extract': lambda *args, **kwargs: _get_entity_extractor().extract(*args, **kwargs)
})()
relation_extractor = type('RelationExtractorProxy', (), {
    'extract': lambda *args, **kwargs: _get_relation_extractor().extract(*args, **kwargs)
})()

