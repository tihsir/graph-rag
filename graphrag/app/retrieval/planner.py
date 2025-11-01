"""Retrieval planner: routing and query plan generation."""

import json
from typing import Dict, List, Optional, Literal
from pathlib import Path

from ..core.providers import get_llm_provider
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class QueryPlan:
    """Query execution plan."""

    def __init__(
        self,
        route: Literal["graph", "rag", "direct"],
        rationale: str = "",
        seed_nodes: Optional[List[str]] = None,
        max_hops: int = 2,
        top_k: int = 10,
        rerank: bool = True,
    ):
        self.route = route
        self.rationale = rationale
        self.seed_nodes = seed_nodes or []
        self.max_hops = max_hops
        self.top_k = top_k
        self.rerank = rerank


class RetrievalPlanner:
    """Plans retrieval strategy based on query intent."""

    def __init__(self, llm_provider=None):
        self._llm_provider_arg = llm_provider
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM provider."""
        if self._llm is None:
            self._llm = self._llm_provider_arg or get_llm_provider()
        return self._llm

    def _load_routing_prompt(self) -> str:
        """Load routing prompt from file."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "routing.toml"
        # Try to load from file if it exists
        if prompt_path.exists():
            try:
                import tomli
                with open(prompt_path, "rb") as f:
                    data = tomli.load(f)
                    return data.get("system", {}).get("prompt", "")
            except:
                pass
        # Fallback to hardcoded prompt
        return """Classify the user query into one of:
- graph: entity-heavy or asks relationships ("how is X related to Y", "who founded", "part of").
- rag: informational/open-ended grounded in sources.
- direct: creative or subjective without source grounding.
Output EXACT JSON: {"route":"graph|rag|direct","rationale":"<≤40 tokens>"}"""

    async def plan(self, query: str, mode: Optional[Literal["auto", "graph", "rag"]] = "auto") -> QueryPlan:
        """Generate query plan from user query."""
        if mode != "auto":
            route = mode
            rationale = f"Mode explicitly set to {mode}"
            return QueryPlan(route=route, rationale=rationale)
        
        # Use LLM to route (lazy access to self.llm)
        prompt = f"""User Query: {query}

{self._load_routing_prompt()}"""
        
        try:
            # Access self.llm property which will lazily initialize
            response = await self.llm.generate(prompt, max_tokens=100, temperature=0.0)
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_str)
            route = result.get("route", "rag")
            rationale = result.get("rationale", "")
            
            # Validate route
            if route not in ["graph", "rag", "direct"]:
                route = "rag"
            
            return QueryPlan(route=route, rationale=rationale, max_hops=settings.max_hops, top_k=settings.max_context_chunks)
        except ValueError as e:
            # API key missing - default to RAG
            logger.warning("LLM provider not configured, defaulting to RAG", error=str(e))
            return QueryPlan(route="rag", rationale=f"LLM not configured: {str(e)}", max_hops=settings.max_hops, top_k=settings.max_context_chunks)
        except Exception as e:
            logger.warning("Routing failed, defaulting to RAG", error=str(e))
            return QueryPlan(route="rag", rationale="Routing failed", max_hops=settings.max_hops, top_k=settings.max_context_chunks)


# Global planner instance (lazy initialization)
_planner = None

def _get_planner() -> RetrievalPlanner:
    """Get global planner (lazy initialization)."""
    global _planner
    if _planner is None:
        _planner = RetrievalPlanner()
    return _planner

# Module-level accessor (lazy - behaves like RetrievalPlanner instance)
planner = type('PlannerProxy', (), {
    'plan': lambda self, *args, **kwargs: _get_planner().plan(*args, **kwargs),
})()

