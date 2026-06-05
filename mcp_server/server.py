from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bayesian-engine")

MOCK_DATABASE = {
    "user_001": {"name": "Alice", "status": "active"},
    "user_002": {"name": "Bob", "status": "inactive"},
    "user_003": {"name": "Charlie", "status": "active"},
    "task_001": {"title": "Review PR", "priority": "high"},
    "task_002": {"title": "Update docs", "priority": "medium"},
}

@mcp.tool()
def query_database(query: str) -> str:
    """Query the local mock database with a simple string search."""
    results = []
    for key, value in MOCK_DATABASE.items():
        if query.lower() in key.lower() or any(
            query.lower() in str(v).lower() for v in value.values()
        ):
            results.append(f"{key}: {value}")

    if results:
        return "Found records: " + "; ".join(results)
    else:
        return "No matching records in database"

@mcp.tool()
def resolve_conflict_tool(variable1: str, variable2: str, variable3: str) -> str:
    """Tool to resolve conflicting data using Bayesian inference."""
    from bayesian_engine.bayes_engine import resolve_conflict

    evidence = {
        "TaskStatus": int(variable1) % 5,
        "DataQuality": int(variable2) % 5,
        "ToolReliability": int(variable3) % 5,
    }

    result = resolve_conflict(evidence)
    return f"Conflict resolved: {result['state']} confidence (probability: {result['confidence']:.2f})"

if __name__ == "__main__":
    mcp.run()
