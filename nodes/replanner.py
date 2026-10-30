import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from core.json_utils import extract_json
from core.runtime import ModelOutputError, bounded_model, check_deadline, positive_int
from core.schemas import Act, PlanAction
from core.state import PlanExecuteState

#: A step below this confidence is re-checked once, in code, before the model decides.
LOW_CONFIDENCE = 0.5
RECHECK_PREFIX = "Re-check: "


def _normalize_step(step: str) -> str:
    return re.sub(r"\s+", " ", step).strip().lower()


def _max_steps() -> int:
    return positive_int("MAX_STEPS", 10, maximum=100)


def _filter_repeated_steps(steps: list[str], completed_tasks: set[str]) -> list[str]:
    """Remove completed and duplicate steps while preserving order.

    A completed step with its old result tacked on ("task: result...") is still that
    step; small models produce it when asked to redo work, and it looped forever.
    """
    filtered = []
    seen = set()
    for step in steps:
        if not isinstance(step, str):
            continue
        normalized = _normalize_step(step)
        if not normalized or normalized in completed_tasks or normalized in seen:
            continue
        if any(normalized.startswith(f"{done}:") for done in completed_tasks):
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
Never include a step that already appears in the completed steps list.

Each completed step shows a confidence between 0 and 1. If a step's confidence is below {threshold},
either add a differently-worded step that re-checks it, or say clearly in your final response
that this part of the answer is uncertain."""),
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
    check_deadline(config)

    past_steps = state.get("past_steps", [])
    confidences = state.get("step_confidences", [])
    completed_tasks = {_normalize_step(task) for task, _ in past_steps}
    current_plan = _filter_repeated_steps(state.get("plan", []), completed_tasks)

    if len(past_steps) >= _max_steps():
        return {"response": f"Stopped after {len(past_steps)} steps without a final answer. "
                            f"Results so far: {_fallback_summary(past_steps)}"}

    # Hard rule: a low-confidence step gets one reworded retry, whatever the model thinks --
    # unless every sample refused, since asking again can't produce data the model lacks.
    evidence = state.get("step_evidence", [])
    all_refused = (
        bool(evidence) and len(evidence) == len(past_steps) and evidence[-1]["TaskStatus"] == 4
    )
    if (
        past_steps
        and len(confidences) == len(past_steps)
        and confidences[-1] < LOW_CONFIDENCE
        and not all_refused
    ):
        last_task = past_steps[-1][0]
        if not last_task.startswith(RECHECK_PREFIX):
            return {"plan": [RECHECK_PREFIX + last_task] + current_plan}

    payload = {
        "input": state["input"],
        "plan": str(current_plan),
        "past_steps": _format_past_steps(past_steps, confidences) or "(none)",
        "threshold": LOW_CONFIDENCE,
    }
    settings = config.get("configurable", {})
    for _attempt in range(positive_int("MODEL_OUTPUT_ATTEMPTS", 2, maximum=3)):
        replanner = create_replanner(bounded_model(settings.get("model"), settings.get("deadline")))
        response = replanner.invoke(payload)
        text = response.content if hasattr(response, "content") else str(response)
        try:
            data = extract_json(text) if isinstance(text, str) else None
            decision = Act.model_validate(data).root
            break
        except ValidationError:
            payload["input"] = state["input"] + (
                '\nReturn JSON with action="plan" and a string steps array, '
                'or action="response" and a nonempty string response.'
            )
    else:
        raise ModelOutputError("Replanner did not return a valid decision")

    if not isinstance(decision, PlanAction):
        return {"response": decision.response}

    next_steps = _filter_repeated_steps(decision.steps, completed_tasks)
    if next_steps:
        return {"plan": next_steps}
    if current_plan:
        return {"plan": current_plan}
    return {"response": _fallback_summary(past_steps)}


def _format_past_steps(past_steps, step_confidences) -> str:
    """One line per completed step, tagged with its confidence when one was recorded."""
    lines = []
    for i, (task, result) in enumerate(past_steps):
        line = f"- {task}: {result}"
        if i < len(step_confidences):
            line += f" (confidence: {step_confidences[i]:.2f})"
        lines.append(line)
    return "\n".join(lines)


def _fallback_summary(past_steps) -> str:
    """If the model never emits a clean final response, surface the gathered results
    instead of an opaque error so the run still returns something useful."""
    if not past_steps:
        return "No results were produced."
    # A re-check supersedes the attempt it re-checked; identical results appear once.
    latest: dict[str, str] = {}
    for task, result in past_steps:
        latest[task.removeprefix(RECHECK_PREFIX)] = result
    return " ".join(dict.fromkeys(r.strip() for r in latest.values() if r.strip()))
