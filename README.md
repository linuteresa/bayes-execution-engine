This project bridges the gap between traditional software engineering and advanced data science. It proves that you do not just know how to write a prompt; you know how to build the underlying infrastructure that makes AI safe, predictable, and mathematically sound for enterprise deployment.

The biggest problem in enterprise AI right now is agent reliability.  Standard AI agents (using the typical "ReAct" loop) are notoriously brittle. 

If you ask them a complex question, they guess an action, look at the result, and guess again. When they encounter conflicting data or missing information, they panic, hallucinate, or get stuck in infinite API-calling loops.

Plan-and-Execute (The DAG): Before taking a single action, your system forces the AI to map out the entire solution as a Directed Acyclic Graph (DAG). It decouples the "thinking" from the "doing."

Bayesian Conflict Resolution: When the execution engine inevitably hits conflicting data, it does not rely on the LLM's text-generation to guess the right answer. Instead, it pauses the graph and runs a mathematical Bayesian update—calculating exact posterior probabilities across a perfectly allocated 125-state conditional probability matrix—to logically deduce the correct path forward.

Also this features an option to run the code on a terminal or on the browser.

## Bayes Execution Engine

- A high-level "Plan-and-Execute" multi-agent orchestration engine built using LangGraph and MCP.
- It is designed to prevent infinite loops when the AI faces conflicting data.
- It uses mathematical Bayesian probability to resolve conflicts instead of letting the LLM simply agree with the prompt.

## How It Works

### 1. The Manager

- LangGraph agents pass a `state` object back and forth.
- `PlanExecuteState` tracks the user's input, the current plan, `past_steps`, a final response, and `confidence_score`.
- `schemas.py` uses Pydantic to ensure the LLM strictly outputs data in the correct format for Plan and Response.

### 2. The Planner

- The planner prompts the LLM to act as an "orchestrator architecture planner".
- It breaks the goal down into a logical, sequential list of discrete operations instead of doing the work immediately.
- It returns the steps as a JSON array, which populates the `plan` field in the state.

### 3. The Executor and the Bayesian Engine

- The executor takes the first step from the plan and runs it through a `simple_executor`.
- If the result string contains words like "conflict", "uncertain", or "disagree", the system intercepts the flow.
- Instead of asking the LLM to guess what to do next, it calls `resolve_conflict` from `bayes-engine.py`.
- The engine uses a 124-discrete Bayesian Network evaluating `TaskStatus`, `DataQuality`, and `ToolReliability`.
- It maps 5 semantic states for each variable: `CERTAIN`, `HIGH`, `MEDIUM`, `LOW`, and `AMBIGUOUS`.
- With 3 parent variables and 5 possible states each, there are exactly 125 environmental combinations.
- The code creates a Conditional Probability Table for the Outcome variable.
- For every one of those 125 combinations, it generates a strict mathematical probability distribution across 5 possible outcomes.
- `np.random.dirchlet` is used so the probability of every column always sums to 1.
- After that, the executor removes completed tasks from the plan, logs them in `past_steps`, and updates the overall `confidence_score`.

### 4. The Replanner

- After a step is executed, the state moves to a replanner node.
- It looks at the original input, the steps completed so far, and the remaining plan.
- If the system has gathered enough information to answer the user's prompt, it returns a final Response JSON.
- If there is more work to be done, it can update the plan and return Plan JSON, keeping the execution loop going.