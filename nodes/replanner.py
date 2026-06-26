import json

from langchain_core.prompts import ChatPromptTemplate

from core.state import PlanExecuteState

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
Otherwise, return plan with the next steps to execute."""),
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

    past_steps_str = "\n".join([f"- {task}: {result}" for task, result in state.get("past_steps", [])])

    response = replanner.invoke({
        "input": state["input"],
        "plan": str(state.get("plan", [])),
        "past_steps": past_steps_str or "(none)",
    })

    try:
        # Extract content from AIMessage if needed
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
        else:
            return {"response": "Unable to parse replanner output"}

        action = data.get("action", "response")

        if action == "response":
            return {"response": data.get("response", "Completed")}
        else:  # action == "plan"
            return {"plan": data.get("steps", [])}
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"[REPLANNER ERROR] Failed to parse: {e}")
        return {"response": "Error processing replanner output"}
