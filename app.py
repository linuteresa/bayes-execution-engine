import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from core.state import PlanExecuteState
from nodes.planner import planner_node
from nodes.executor import executor_node
from nodes.replanner import replanner_node

load_dotenv()

app = Flask(__name__)

def should_continue(state: PlanExecuteState):
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
    base_url = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url=base_url,
        api_key="not-needed",
        temperature=0,
    )

    app_graph = build_graph()

    initial_state = {
        "input": user_input,
        "plan": [],
        "past_steps": [],
        "response": "",
        "confidence_score": 1.0,
    }

    final_state = app_graph.invoke(
        initial_state,
        config={"configurable": {"model": model}},
    )

    return {
        "response": final_state.get('response', 'No response'),
        "steps_executed": len(final_state.get('past_steps', [])),
        "confidence_score": final_state.get('confidence_score', 0.0),
        "past_steps": final_state.get('past_steps', []),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = run_execution_engine(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
