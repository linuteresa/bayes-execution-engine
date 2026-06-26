"""Shared LangGraph wiring for the Plan-and-Execute engine.

Both the CLI (``main.py``) and the web/API layers import from here so the graph is
defined exactly once.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

from core.llm import build_llm
from core.state import PlanExecuteState
from core.telemetry import span
from nodes.executor import executor_node
from nodes.planner import planner_node
from nodes.replanner import replanner_node


def should_continue(state: PlanExecuteState) -> Literal["executor", "END"]:
    """Route to executor while plan steps remain, otherwise finish."""
    if state.get("response"):
        return "END"
    if state.get("plan"):
        return "executor"
    return "END"


def build_graph(checkpointer: Optional[Any] = None):
    """Compile the Plan-and-Execute StateGraph.

    Pass a LangGraph ``checkpointer`` (e.g. a Redis/Postgres saver) to persist state
    across process restarts and enable concurrent, isolated sessions by thread id.
    """
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")
    workflow.add_conditional_edges(
        "replanner", should_continue, {"executor": "executor", "END": END}
    )
    return workflow.compile(checkpointer=checkpointer)


def run_execution_engine(
    user_input: str,
    *,
    checkpointer: Optional[Any] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full orchestration for one prompt and return a result dict."""
    model = build_llm()
    app_graph = build_graph(checkpointer=checkpointer)

    initial_state = {
        "input": user_input,
        "plan": [],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }
    config: Dict[str, Any] = {"configurable": {"model": model}}
    if thread_id is not None:
        config["configurable"]["thread_id"] = thread_id

    with span("engine.run", input=user_input, thread_id=thread_id):
        final_state = app_graph.invoke(initial_state, config=config)

    return {
        "response": final_state.get("response", "No response"),
        "steps_executed": len(final_state.get("past_steps", [])),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "past_steps": final_state.get("past_steps", []),
    }
