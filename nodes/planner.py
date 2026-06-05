from langchain_core.prompts import ChatPromptTemplate
from core.schemas import Plan

def create_planner(model):
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert orchestrator architecture planner. Analyze the objective and generate a complete, step-by-step sequential plan. Do not execute any tools. Break down the task into discrete, logical operations that an execution engine can process one by one."),
        ("user", "{input}")
    ])
    
    planner = planner_prompt | model.with_structured_output(Plan)
    return planner

def planner_node(state, config):
    model = config.get("configurable", {}).get("model")
    planner = create_planner(model)
    
    response = planner.invoke({"input": state["input"]})
    
    return {"plan": response.steps}