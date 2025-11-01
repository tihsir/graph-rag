"""Tests for retrieval planner."""

import pytest
from graphrag.app.retrieval.planner import RetrievalPlanner, QueryPlan


@pytest.mark.asyncio
async def test_planner_explicit_mode():
    """Test planner with explicit mode."""
    planner = RetrievalPlanner()
    
    plan = await planner.plan("test query", mode="graph")
    assert plan.route == "graph"
    
    plan = await planner.plan("test query", mode="rag")
    assert plan.route == "rag"


@pytest.mark.asyncio
async def test_planner_auto_mode():
    """Test planner with auto mode (may require LLM)."""
    planner = RetrievalPlanner()
    
    plan = await planner.plan("test query", mode="auto")
    assert plan.route in ["graph", "rag", "direct"]

