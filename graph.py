from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import Settings
from tools import build_tools


class AgentState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    intermediate_steps: Annotated[list[str], operator.add]
    tool_outputs: Annotated[list[dict[str, str]], operator.add]
    used_tools: Annotated[list[str], operator.add]
    reasoning_log: Annotated[list[str], operator.add]
    decision_log: Annotated[list[str], operator.add]
    memory_log: Annotated[list[str], operator.add]
    final_answer: str
    iterations: int
    max_iterations: int


SYSTEM_PROMPT = """You are a single research assistant that works in a ReAct-style loop.
Decide whether a tool is needed before answering.
Use tools for current facts, calculations, or local document lookup.
Use short, high-precision search queries such as company names or document names.
Once you have the needed facts, stop searching and provide the answer.
If the user asks for a difference, average, or other math, call the calculator tool.
Your final answer must be plain text only.
Never output XML, JSON, <function_calls>, or pseudo tool markup.
Always explain uncertainty briefly if evidence is incomplete."""


def create_graph(settings: Settings):
    tools = build_tools(settings.docs_dir)
    tool_map = {tool.name: tool for tool in tools}

    llm = AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> AgentState:
        iterations = state.get("iterations", 0) + 1
        response = llm_with_tools.invoke(_build_agent_messages(state))

        if response.tool_calls and iterations >= state.get("max_iterations", settings.max_iterations):
            forced_response = _finalize_with_observations(llm, state)
            return {
                "messages": [forced_response],
                "intermediate_steps": ["Agent: reached the iteration limit and produced a final answer."],
                "reasoning_log": [
                    "Reasoning: enough evidence had been collected, so the loop was stopped at the iteration limit and summarized into a final answer."
                ],
                "decision_log": [
                    "Decision: stop the loop because the maximum number of iterations was reached."
                ],
                "memory_log": [_build_memory_snapshot(state, final_answer=_message_to_text(forced_response), iteration=iterations)],
                "iterations": iterations,
                "final_answer": _message_to_text(forced_response),
            }

        agent_text = _message_to_text(response)
        if response.tool_calls:
            tool_names = ", ".join(call["name"] for call in response.tool_calls)
            agent_step = f"Agent: decided to call tool(s): {tool_names}"
        else:
            agent_step = f"Agent: {agent_text}"

        update: AgentState = {
            "messages": [response],
            "intermediate_steps": [agent_step],
            "reasoning_log": [_describe_agent_reasoning(state, response)],
            "decision_log": _describe_agent_decisions(state, response),
            "memory_log": [_build_memory_snapshot(state, next_action=_describe_next_action(response), iteration=iterations)],
            "iterations": iterations,
        }

        if not response.tool_calls and _looks_like_tool_markup(agent_text):
            forced_response = _finalize_with_observations(llm, state)
            update["messages"] = [forced_response]
            update["intermediate_steps"] = ["Agent: converted invalid pseudo tool markup into a final plain-text answer."]
            update["reasoning_log"] = [
                "Reasoning: the previous model output was not a valid final answer, so the system reformulated it from the stored observations."
            ]
            update["decision_log"] = [
                "Decision: ignore invalid pseudo tool markup and force a final plain-text answer."
            ]
            update["memory_log"] = [_build_memory_snapshot(state, final_answer=_message_to_text(forced_response), iteration=iterations)]
            update["final_answer"] = _message_to_text(forced_response)
            return update

        if not response.tool_calls:
            update["final_answer"] = agent_text
            update["memory_log"] = [_build_memory_snapshot(state, final_answer=agent_text, iteration=iterations)]

        return update

    def tools_node(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {}

        tool_messages: list[ToolMessage] = []
        tool_outputs: list[dict[str, str]] = []
        used_tools: list[str] = []
        intermediate_steps: list[str] = []
        reasoning_log: list[str] = []
        decision_log: list[str] = []
        memory_log: list[str] = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = tool_map[tool_name]

            try:
                result = tool.invoke(tool_call["args"])
            except Exception as exc:  # pragma: no cover - defensive fallback
                result = f"Tool execution failed: {exc}"

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
            tool_outputs.append({"tool": tool_name, "output": str(result)})
            used_tools.append(tool_name)
            intermediate_steps.append(
                f"Tool {tool_name}: input={tool_call['args']} | output={str(result)[:200]}"
            )
            reasoning_log.append(_describe_tool_reasoning(tool_name, str(result)))
            decision_log.append(_describe_tool_execution(tool_name, tool_call["args"]))
            memory_log.append(
                _build_memory_snapshot(
                    state,
                    stored_tool=tool_name,
                    stored_output=str(result),
                    iteration=state.get("iterations", 0),
                )
            )

        return {
            "messages": tool_messages,
            "tool_outputs": tool_outputs,
            "used_tools": used_tools,
            "intermediate_steps": intermediate_steps,
            "reasoning_log": reasoning_log,
            "decision_log": decision_log,
            "memory_log": memory_log,
        }

    def should_continue(state: AgentState) -> str:
        if state.get("final_answer"):
            return END

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_initial_state(question: str, max_iterations: int = 5) -> AgentState:
    return {
        "question": question,
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "plan": _build_initial_plan(question),
        "intermediate_steps": [],
        "tool_outputs": [],
        "used_tools": [],
        "reasoning_log": [
            "Reasoning: the system starts by analyzing the question and deciding whether external information, local documents, or calculation are required."
        ],
        "decision_log": [
            "Decision: begin in the agent node and evaluate whether a tool call is necessary."
        ],
        "memory_log": [
            _build_memory_snapshot(
                {"question": question, "tool_outputs": [], "used_tools": []},
                next_action="agent analysis",
                iteration=0,
            )
        ],
        "iterations": 0,
        "max_iterations": max_iterations,
    }


def _message_to_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    return str(content).strip()


def _build_agent_messages(state: AgentState) -> list[BaseMessage]:
    messages = list(state["messages"])
    search_count = sum(1 for item in state.get("tool_outputs", []) if item["tool"] == "search_tool")
    question = state.get("question", "").lower()

    if search_count >= 2:
        reminder = (
            "You already have multiple search observations. "
            "Do not call search_tool again unless the current evidence is clearly insufficient. "
            "If the user asked for a difference, average, total, or comparison, either use calculator_tool now or answer directly."
        )
        if any(keyword in question for keyword in ["difference", "durchschnitt", "average", "sum", "compare", "vergleich"]):
            reminder += " Prefer calculator_tool over another search."
        messages.append(SystemMessage(content=reminder))

    return messages


def _finalize_with_observations(llm: AzureChatOpenAI, state: AgentState) -> AIMessage:
    observations = state.get("tool_outputs", [])
    observation_lines = [
        f"- {item['tool']}: {item['output']}"
        for item in observations
    ] or ["- No tool outputs were recorded."]

    final_prompt = [
        SystemMessage(
            content=(
                "You are writing the final answer for a research assistant. "
                "Use the observations below as evidence and respond in plain text only. "
                "Do not call tools and do not output XML, JSON, or markup."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                "Available observations:\n"
                + "\n".join(observation_lines)
                + "\n\nProvide a concise answer. If the evidence is insufficient, say so explicitly."
            )
        ),
    ]
    return llm.invoke(final_prompt)


def _looks_like_tool_markup(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ["<function_calls>", "<invoke name=", "<parameter name="])


def _build_initial_plan(question: str) -> list[str]:
    question_lower = question.lower()
    plan = ["Analyse the question and identify which facts or numbers are needed."]

    if any(keyword in question_lower for keyword in ["docs/", ".txt", "document", "dokument", "datei"]):
        plan.append("Retrieve relevant passages from local documents in the docs folder.")

    if any(keyword in question_lower for keyword in ["which", "welche", "compare", "vergleich", "more", "mehr", "less", "weniger"]):
        plan.append("Collect evidence for the relevant entities before answering.")

    if any(keyword in question_lower for keyword in ["difference", "differenz", "berechne", "average", "durchschnitt", "sum", "gesamt"]):
        plan.append("Use the calculator tool to compute the requested value from the collected numbers.")

    plan.append("Synthesize the observations into a concise final answer with a short uncertainty note if needed.")
    return plan


def _describe_agent_reasoning(state: AgentState, response: AIMessage) -> str:
    if response.tool_calls:
        tool_names = ", ".join(call["name"] for call in response.tool_calls)
        return f"Reasoning: the current state does not yet support a reliable final answer, so the agent requests {tool_names}."

    if state.get("tool_outputs"):
        return "Reasoning: the collected observations are sufficient to stop the loop and formulate the final answer."

    return "Reasoning: the question can be answered directly without external tools."


def _describe_agent_decisions(state: AgentState, response: AIMessage) -> list[str]:
    if response.tool_calls:
        return [
            f"Decision: call {tool_call['name']} next."
            for tool_call in response.tool_calls
        ]

    if state.get("tool_outputs"):
        return ["Decision: stop tool use and generate the final answer from the available evidence."]

    return ["Decision: answer directly because no external tool is required."]


def _describe_tool_reasoning(tool_name: str, result: str) -> str:
    preview = _truncate_text(result, 120)
    return f"Reasoning: the observation from {tool_name} was added to working memory for the next agent step: {preview}"


def _describe_tool_execution(tool_name: str, tool_args: Any) -> str:
    return f"Decision: execute {tool_name} with input {tool_args}."


def _describe_next_action(response: AIMessage) -> str:
    if response.tool_calls:
        return "tool execution"
    return "final answer"


def _build_memory_snapshot(
    state: AgentState | dict[str, Any],
    *,
    next_action: str | None = None,
    stored_tool: str | None = None,
    stored_output: str | None = None,
    final_answer: str | None = None,
    iteration: int,
) -> str:
    question = str(state.get("question", ""))
    used_tools = state.get("used_tools", [])
    observations = state.get("tool_outputs", [])

    parts = [
        f"Iteration {iteration}",
        f"question='{_truncate_text(question, 90)}'",
        f"observations={len(observations)}",
        f"used_tools={list(dict.fromkeys(used_tools))}",
    ]

    if stored_tool:
        parts.append(f"stored={stored_tool}")
    if stored_output:
        parts.append(f"latest_observation='{_truncate_text(stored_output, 110)}'")
    if next_action:
        parts.append(f"next='{next_action}'")
    if final_answer:
        parts.append(f"final_answer='{_truncate_text(final_answer, 110)}'")

    return " | ".join(parts)


def _truncate_text(value: str, max_length: int) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3] + "..."
