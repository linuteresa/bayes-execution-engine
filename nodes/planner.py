import json

from langchain_core.prompts import ChatPromptTemplate

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

    try:
        # Extract content from AIMessage if needed
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            steps = data.get("steps", [])
        else:
            steps = []
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[PLANNER ERROR] Failed to parse: {e}")
        steps = []

    return {"plan": steps}