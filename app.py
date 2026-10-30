"""Flask UI for the Bayes Execution Engine (synchronous, for local demos).

For production / long-running LLM calls use the async FastAPI service in
``service/api.py`` instead, which queues work and avoids HTTP timeouts.
"""

from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from core.graph import run_execution_engine
from core.runtime import MAX_PROMPT_CHARS_CEILING, positive_int

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", max_prompt_chars=positive_int(
        "MAX_PROMPT_CHARS", 16000, maximum=MAX_PROMPT_CHARS_CEILING,
    ))


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("question"), str):
        return jsonify({"error": "Provide a JSON object with a string question"}), 400
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    max_chars = positive_int("MAX_PROMPT_CHARS", 16000, maximum=MAX_PROMPT_CHARS_CEILING)
    if len(question) > max_chars:
        return jsonify({"error": f"Question exceeds {max_chars} characters"}), 400
    try:
        result = run_execution_engine(question)
        return jsonify(result), 503 if result.get("error") else 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
