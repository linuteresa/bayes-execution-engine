from langchain_core.prompts import ChatPromptTemplate

from core.json_utils import extract_json


def create_planner(model):
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert orchestrator architecture planner. Analyze the objective and generate a complete, step-by-step sequential plan in JSON format.
Do not execute any tools. Break down the task into discrete, logical operations.

Return ONLY valid JSON with this structure:
{{"steps": ["step1", "step2", "step3"]}}"""),
        ("user", "{input}")
    ])

    planner = planner_prompt | model
    return planner

def planner_node(state, config):
    model = config.get("configurable", {}).get("model")
    planner = create_planner(model)

    response = planner.invoke({"input": state["input"]})

    response_text = response.content if hasattr(response, "content") else str(response)
    data = extract_json(response_text)
    steps = data.get("steps", []) if isinstance(data, dict) else []
    if not steps:
        print(f"[PLANNER] No parseable steps in model output: {response_text[:200]!r}")

    return {"plan": steps}