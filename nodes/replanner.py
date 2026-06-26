import re

from langchain_core.prompts import ChatPromptTemplate

from core.json_utils import extract_json
from core.state import PlanExecuteState


def _normalize_step(step: str) -> str:
    return re.sub(r"\s+", " ", step).strip().lower()


def _filter_repeated_steps(steps: list[str], completed_tasks: set[str]) -> list[str]:
    """Remove completed and duplicate steps while preserving order."""
    filtered = []
    seen = set()
    for step in steps:
        if not isinstance(step, str):
            continue
        normalized = _normalize_step(step)
        if not normalized or normalized in completed_tasks or normalized in seen:
            continue
        filtered.append(step)
        seen.add(normalized)
    return filtered

def create_replanner(model):
    """Create a replanner that decides to either continue executing or finish."""
    replanner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task replanner. Given the original input, the current execution plan, and the steps completed so far, decide whether to:
1. Continue with the next execution steps (return a Plan with updated steps)
2. Provide a final response (return a Response)

Return ONLY valid JSON in ONE of these formats:

For continuing: {{"action": "plan", "steps": ["step1", "step2"]}}
For finishing: {{"action": "response", "response": "your final answer here"}}

If all necessary information has been gathered or the goal is achieved, return response.
Otherwise, return plan with only the next uncompleted steps to execute.
Never include a step that already appears in the completed steps list."""),
        ("user", """Original input: {input}
Current plan remaining: {plan}
Steps completed: {past_steps}

Decide: continue (plan) or finish (response)?""")
    ])

    replanner = replanner_prompt | model
    return replanner

def replanner_node(state: PlanExecuteState, config):
    """
    Replan or finish based on progress.
    - If action is Response, set final response
    - If action is Plan, update the plan
    """
    model = config.get("configurable", {}).get("model")
    replanner = create_replanner(model)

    past_steps = state.get("past_steps", [])
    completed_tasks = {_normalize_step(task) for task, _ in past_steps}
    current_plan = _filter_repeated_steps(state.get("plan", []), completed_tasks)
    past_steps_str = "\n".join([f"- {task}: {result}" for task, result in past_steps])

    response = replanner.invoke({
        "input": state["input"],
        "plan": str(current_plan),
        "past_steps": past_steps_str or "(none)",
    })

    response_text = response.content if hasattr(response, "content") else str(response)
    data = extract_json(response_text)

    if not isinstance(data, dict):
        # No parseable decision. If work remains, keep going; otherwise summarise.
        if current_plan:
            return {"plan": current_plan}
        return {"response": _fallback_summary(past_steps)}

    action = data.get("action", "response")

    if action == "response":
        return {"response": data.get("response") or _fallback_summary(past_steps)}

    # action == "plan"
    next_steps = _filter_repeated_steps(data.get("steps", []), completed_tasks)
    if next_steps:
        return {"plan": next_steps}
    if current_plan:
        return {"plan": current_plan}
    return {"response": _fallback_summary(past_steps)}


def _fallback_summary(past_steps) -> str:
    """If the model never emits a clean final response, surface the gathered results
    instead of an opaque error so the run still returns something useful."""
    if not past_steps:
        return "No results were produced."
    return " ".join(result for _, result in past_steps).strip()
