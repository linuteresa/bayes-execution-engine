from langchain_core.prompts import ChatPromptTemplate
from core.schemas import Act
from core.state import PlanExecuteState

def create_replanner(model):
    """Create a replanner that decides to either continue executing or finish."""
    replanner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task replanner. Given the original input, the current execution plan, and the steps completed so far, decide whether to:
1. Continue with the next execution steps (return a Plan with updated steps)
2. Provide a final response (return a Response)

If all necessary information has been gathered or the goal is achieved, return a Response.
Otherwise, return a Plan with the next steps to execute."""),
        ("user", """Original input: {input}
Current plan remaining: {plan}
Steps completed: {past_steps}

Decide: continue (Plan) or finish (Response)?""")
    ])

    replanner = replanner_prompt | model.with_structured_output(Act)
    return replanner

def replanner_node(state: PlanExecuteState, config):
    """
    Replan or finish based on progress.
    - If action is Response, set final response
    - If action is Plan, update the plan
    """
    model = config.get("configurable", {}).get("model")
    replanner = create_replanner(model)

    past_steps_str = "\n".join([f"- {task}: {result}" for task, result in state.get("past_steps", [])])

    response = replanner.invoke({
        "input": state["input"],
        "plan": str(state.get("plan", [])),
        "past_steps": past_steps_str or "(none)",
    })

    action = response.action

    if hasattr(action, 'response'):
        return {"response": action.response}
    else:
        return {"plan": action.steps}
