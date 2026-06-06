Engineered a deterministic Plan-and-Execute multi-agent orchestration engine using LangGraph and MCP, integrating Bayesian probabilistic modeling (pgmpy) to eliminate hallucination loops and resolve data conflicts in automated execution pipelines.

This project bridges the gap between traditional software engineering and advanced data science. It proves that you do not just know how to write a prompt; you know how to build the underlying infrastructure that makes AI safe, predictable, and mathematically sound for enterprise deployment.

The biggest problem in enterprise AI right now is agent reliability.

Standard AI agents (using the typical "ReAct" loop) are notoriously brittle. If you ask them a complex question, they guess an action, look at the result, and guess again. When they encounter conflicting data or missing information, they panic, hallucinate, or get stuck in infinite API-calling loops.

Plan-and-Execute (The DAG): Before taking a single action, your system forces the AI to map out the entire solution as a Directed Acyclic Graph (DAG). It decouples the "thinking" from the "doing."

Bayesian Conflict Resolution: When the execution engine inevitably hits conflicting data, it does not rely on the LLM's text-generation to guess the right answer. Instead, it pauses the graph and runs a mathematical Bayesian update—calculating exact posterior probabilities across a perfectly allocated 125-state conditional probability matrix—to logically deduce the correct path forward.

Also this features an option to run the code on a terminal or on the browser.