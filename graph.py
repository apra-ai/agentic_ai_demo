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
    intermediate_steps: Annotated[list[str], operator.add]
    tool_outputs: Annotated[list[dict[str, str]], operator.add]
    used_tools: Annotated[list[str], operator.add]
    final_answer: str
    iterations: int
    max_iterations: int


SYSTEM_PROMPT = """You are a single research assistant that works in a ReAct-style loop.
Decide whether a tool is needed before answering.
Use tools for current facts, calculations, or local document lookup.
When you have enough information, respond with a concise final answer.
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
        response = llm_with_tools.invoke(state["messages"])

        if response.tool_calls and iterations >= state.get("max_iterations", settings.max_iterations):
            forced_response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You have reached the maximum number of tool iterations. "
                            "Provide the best possible final answer with the available evidence."
                        )
                    ),
                    *state["messages"],
                ]
            )
            return {
                "messages": [forced_response],
                "intermediate_steps": ["Agent: reached the iteration limit and produced a final answer."],
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
            "iterations": iterations,
        }

        if not response.tool_calls:
            update["final_answer"] = agent_text

        return update

    def tools_node(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {}

        tool_messages: list[ToolMessage] = []
        tool_outputs: list[dict[str, str]] = []
        used_tools: list[str] = []
        intermediate_steps: list[str] = []

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

        return {
            "messages": tool_messages,
            "tool_outputs": tool_outputs,
            "used_tools": used_tools,
            "intermediate_steps": intermediate_steps,
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
        "intermediate_steps": [],
        "tool_outputs": [],
        "used_tools": [],
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
