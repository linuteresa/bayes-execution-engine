"""Flask UI for the Bayes Execution Engine (synchronous, for local demos).

For production / long-running LLM calls use the async FastAPI service in
``service/api.py`` instead, which queues work and avoids HTTP timeouts.
"""

from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from core.graph import run_execution_engine

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    try:
        return jsonify(run_execution_engine(question))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
