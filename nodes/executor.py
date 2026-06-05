from core.state import PlanExecuteState
from bayesian_engine.bayes_engine import resolve_conflict

def simple_executor(task: str) -> str:
    """Simple mock execution engine for demonstration."""
    if "search" in task.lower():
        return "search_result: Found 5 relevant documents"
    elif "query" in task.lower():
        return "query_result: Retrieved 3 records with conflicting timestamps"
    elif "validate" in task.lower():
        return "validate_result: Data validation uncertain - 2 sources disagree"
    else:
        return f"executed: {task}"

def executor_node(state: PlanExecuteState, config=None) -> dict:
    """
    Execute the first task in the plan DAG.
    - Pop the first task from plan
    - Execute it and get result
    - Check if result is ambiguous (triggers Bayesian update)
    - Append (task, result) to past_steps
    - Remove task from plan
    """
    if not state.get("plan"):
        return {"response": "No tasks in plan"}

    current_task = state["plan"][0]
    result = simple_executor(current_task)

    confidence = 1.0
    if any(keyword in result.lower() for keyword in ["conflict", "uncertain", "disagree"]):
        bayes_result = resolve_conflict({
            "TaskStatus": 2,
            "DataQuality": 2,
            "ToolReliability": 3,
        })
        confidence = bayes_result["confidence"]

    new_past_steps = [(current_task, result)]

    return {
        "plan": state["plan"][1:],
        "past_steps": new_past_steps,
        "confidence_score": confidence,
    }
