from __future__ import annotations

import sys

from config import load_settings
from graph import build_initial_state, create_graph


def main() -> None:
    settings = load_settings()
    question = _read_question()

    app = create_graph(settings)
    result = app.invoke(build_initial_state(question, max_iterations=settings.max_iterations))

    print("\n=== Plan ===")
    for index, step in enumerate(result.get("plan", []), start=1):
        print(f"{index}. {step}")

    print("\n=== Final Answer ===")
    print(result.get("final_answer", "No final answer produced."))

    print("\n=== Used Tools ===")
    used_tools = list(dict.fromkeys(result.get("used_tools", [])))
    print(", ".join(used_tools) if used_tools else "No tools were used.")

    print("\n=== Memory ===")
    _print_section(result.get("memory_log", []), "No memory snapshots recorded.")

    print("\n=== Reasoning ===")
    _print_section(result.get("reasoning_log", []), "No reasoning steps recorded.")

    print("\n=== Decision Log ===")
    _print_section(result.get("decision_log", []), "No decisions recorded.")

    print("\n=== Intermediate Steps ===")
    _print_section(result.get("intermediate_steps", []), "No intermediate steps recorded.")


def _read_question() -> str:
    cli_question = " ".join(sys.argv[1:]).strip()
    if cli_question:
        return cli_question

    prompt = input("Question: ").strip()
    if not prompt:
        raise ValueError("Please provide a question either as CLI argument or interactive input.")
    return prompt


def _print_section(items: list[str], empty_message: str) -> None:
    if not items:
        print(empty_message)
        return

    for index, item in enumerate(items, start=1):
        print(f"{index}. {item}")


if __name__ == "__main__":
    main()
