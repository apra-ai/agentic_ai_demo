from __future__ import annotations

import sys

from config import load_settings
from graph import build_initial_state, create_graph


def main() -> None:
    settings = load_settings()
    question = _read_question()

    app = create_graph(settings)
    result = app.invoke(build_initial_state(question, max_iterations=settings.max_iterations))

    print("\n=== Final Answer ===")
    print(result.get("final_answer", "No final answer produced."))

    print("\n=== Used Tools ===")
    used_tools = list(dict.fromkeys(result.get("used_tools", [])))
    print(", ".join(used_tools) if used_tools else "No tools were used.")

    print("\n=== Intermediate Steps ===")
    steps = result.get("intermediate_steps", [])
    if not steps:
        print("No intermediate steps recorded.")
        return

    for index, step in enumerate(steps, start=1):
        print(f"{index}. {step}")


def _read_question() -> str:
    cli_question = " ".join(sys.argv[1:]).strip()
    if cli_question:
        return cli_question

    prompt = input("Question: ").strip()
    if not prompt:
        raise ValueError("Please provide a question either as CLI argument or interactive input.")
    return prompt


if __name__ == "__main__":
    main()
