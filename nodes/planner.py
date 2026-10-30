from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from core.json_utils import extract_json
from core.runtime import ModelOutputError, bounded_model, positive_int
from core.schemas import Plan


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
    settings = config.get("configurable", {})
    for attempt in range(positive_int("MODEL_OUTPUT_ATTEMPTS", 2, maximum=3)):
        planner = create_planner(bounded_model(settings.get("model"), settings.get("deadline")))
        instruction = state["input"]
        if attempt:
            instruction += '\nReturn a JSON object with a nonempty "steps" array of strings.'
        response = planner.invoke({"input": instruction})
        text = response.content if hasattr(response, "content") else str(response)
        try:
            data = extract_json(text) if isinstance(text, str) else None
            return {"plan": Plan.model_validate(data).steps}
        except ValidationError:
            continue
    raise ModelOutputError("Planner did not return a valid nonempty plan")
