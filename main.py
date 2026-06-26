"""CLI entrypoint for the Bayes Execution Engine."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from core.graph import run_execution_engine

load_dotenv()


def main(user_input: str) -> None:
    print("\n" + "=" * 60)
    print("BAYES EXECUTION ENGINE")
    print("=" * 60)
    print(f"Input: {user_input}\n")

    result = run_execution_engine(user_input)

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Final Response: {result['response']}")
    print(f"Steps Executed: {result['steps_executed']}")
    print(f"Confidence Score: {result['confidence_score']:.2f}")
    print()


if __name__ == "__main__":
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What are the active users in the system and which high-priority tasks are assigned?"
    )
    main(prompt)
