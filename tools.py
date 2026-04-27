from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, tool


_wikipedia = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1800)
_ALLOWED_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_ALLOWED_CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}
_ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}
_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


@tool
def search_tool(query: str) -> str:
    """Search Wikipedia for factual background information about a topic."""
    if not query.strip():
        return "Search query was empty."

    try:
        result = _wikipedia.run(query)
    except Exception as exc:  # pragma: no cover - depends on external service
        return f"Search failed: {exc}"

    return result or "No search results found."


@tool
def calculator_tool(expression: str) -> str:
    """Safely evaluate a mathematical expression such as '(107.4 - 74.9) / 2'."""
    if not expression.strip():
        return "Calculator input was empty."

    try:
        parsed = ast.parse(expression, mode="eval")
        result = _evaluate_math_expression(parsed.body)
    except Exception as exc:
        return f"Calculation failed: {exc}"

    return str(result)


def create_document_retrieval_tool(docs_dir: Path) -> BaseTool:
    @tool("document_retrieval_tool")
    def document_retrieval_tool(query: str) -> str:
        """Search local .txt files in the docs directory and return the most relevant passages."""
        if not docs_dir.exists():
            return f"Docs directory not found: {docs_dir}"

        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        matches: list[tuple[int, str, str]] = []

        for path in sorted(docs_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8")
            score = sum(content.lower().count(term) for term in query_terms)
            if score == 0 and path.stem.lower() not in query.lower():
                continue
            preview = " ".join(content.split())[:500]
            matches.append((score, path.name, preview))

        if not matches:
            return "No matching documents found in /docs."

        matches.sort(key=lambda item: item[0], reverse=True)
        snippets = [
            f"Source: {name}\nSnippet: {preview}"
            for _, name, preview in matches[:3]
        ]
        return "\n\n".join(snippets)

    return document_retrieval_tool


def build_tools(docs_dir: Path) -> list[BaseTool]:
    return [search_tool, calculator_tool, create_document_retrieval_tool(docs_dir)]


def _evaluate_math_expression(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTANTS:
        return _ALLOWED_CONSTANTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        left = _evaluate_math_expression(node.left)
        right = _evaluate_math_expression(node.right)
        return _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        operand = _evaluate_math_expression(node.operand)
        return _ALLOWED_UNARY_OPERATORS[type(node.op)](operand)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_name = node.func.id
        if function_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{function_name}' is not allowed.")
        args = [_evaluate_math_expression(argument) for argument in node.args]
        return _ALLOWED_FUNCTIONS[function_name](*args)

    raise ValueError("Only simple mathematical expressions are allowed.")
