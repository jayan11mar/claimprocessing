"""Orchestrator — multi-agent pipeline entrypoint.

Pure composition: decompose → dispatch → aggregate.
"""

from __future__ import annotations

from app.agents.aggregator import FinalAnswer, aggregate
from app.agents.decomposer import decompose
from app.agents.dispatcher import dispatch


def orchestrate(query: str) -> FinalAnswer:
    """Run the full decompose → dispatch → aggregate pipeline.

    Returns a :class:`FinalAnswer`.
    """
    subtasks = decompose(query)
    results = dispatch(subtasks)
    return aggregate(results)
