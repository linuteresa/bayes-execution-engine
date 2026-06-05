import os
from typing import Literal
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

from core.state import PlanExecuteState
from nodes.planner import planner_node
from nodes.executor import executor_node
from nodes.replanner import replanner_node

load_dotenv()

def should_continue(state: PlanExecuteState) -> Literal["executor", "END"]:
    """Route to executor if plan remains, otherwise finish."""
    if state.get("response"):
        return "END"
    if state.get("plan"):
        return "executor"
    return "END"

def build_graph():
    """Build the LangGraph StateGraph for Plan-and-Execute."""
    workflow = StateGraph(PlanExecuteState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")

    workflow.add_conditional_edges(
        "replanner",
        should_continue,
        {"executor": "executor", "END": END},
    )

    return workflow.compile()

def run_execution_engine(user_input: str):
    """Run the complete Plan-and-Execute orchestration."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    model = ChatAnthropic(
        api_key=api_key,
        model="claude-haiku-4-5-20251001",
        temperature=0,
    )

    app = build_graph()

    initial_state = {
        "input": user_input,
        "plan": [],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    print("\n" + "=" * 60)
    print("BAYES EXECUTION ENGINE")
    print("=" * 60)
    print(f"Input: {user_input}\n")

    final_state = app.invoke(
        initial_state,
        config={"configurable": {"model": model}},
    )

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Final Response: {final_state.get('response', 'No response')}")
    print(f"Steps Executed: {len(final_state.get('past_steps', []))}")
    print(f"Confidence Score: {final_state.get('confidence_score', 0.0):.2f}")
    print()

    return final_state

if __name__ == "__main__":
    test_input = "What are the active users in the system and which high-priority tasks are assigned?"
    run_execution_engine(test_input)
