import json
import re

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from utils.models import LLM
from utils.prompts import Q_SUGGESTION_TEMPLATE, RAG_TEMPLATE, SUMMARY_TEMPLATE
from utils.tools import TOOLS

load_dotenv()

METADATA_PATTERN = re.compile(r"Metadata:\s*({[^}]+})")


class AgentState(MessagesState):
    """State class for managing conversation and suggested questions.

    This class extends MessagesState to track the state of an agent's conversation,
    including suggested follow-up questions and used documents.

    Attributes:
        suggested_question (str): The next question suggested by the agent to
            continue the conversation flow.
        used_docs (list[dict]): List of documents that have been referenced or
            used during the conversation. Each document is represented as a
            dictionary containing document metadata.

    Inherits:
        MessagesState: Base class for managing conversation message state.

    """

    suggested_question: str
    used_docs: list[dict]


async def manage_system_prompt(state: AgentState):
    """Manages system prompts by initializing or maintaining conversation context.

    This node ensures there is always a system prompt present in the conversation.
    If no system prompt exists, it creates one with a default summary template.

    Args:
        state (AgentState): Current conversation state containing:
            - messages: List of conversation messages
            - suggested_question: Optional suggested follow-up
            - used_docs: List of referenced documents

    Returns:
        dict: Updated state with managed system messages
            - messages: List including system prompt
    """
    messages = state["messages"]
    system_messages = filter_messages(messages, include_types=[SystemMessage])

    if not system_messages:
        formatted_template = RAG_TEMPLATE.format(summary="No hay resumen previo.")
        messages.insert(0, SystemMessage(content=formatted_template))

    return {"messages": messages, "suggested_question": "", "used_docs": []}


async def model(state: AgentState):
    """Processes messages through LLM to generate responses with tool support.

    This node handles the core conversation by passing messages through an LLM
    with bound tools for enhanced capabilities like document search and retrieval.

    Args:
        state (AgentState): Current conversation state

    Returns:
        dict: LLM response state
            - messages: List containing the LLM's response
    """
    llm_with_tools = LLM.bind_tools(TOOLS)
    response = await llm_with_tools.with_config({"run_name": "agent_answer"}).ainvoke(
        state["messages"]
    )
    return {"messages": response}


async def pending_tool_calls(state: AgentState):
    """Check if latest AI message contains tool calls and mark for processing.

    Args:
        state (AgentState): Current conversation state

    Returns:
        str: Next node to process - "tools" or "clean_messages"

    Raises:
        TypeError: If last message is not an AIMessage
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        last_message.name = "tool_message"
        return "tools"
    return "clean_messages"


async def clean_messages(state: AgentState):
    """Remove tool-related messages from conversation history and extract document metadata.

    Args:
        state (AgentState): Current conversation state

    Returns:
        dict: List of messages to remove and used documents
    """
    tool_messages = filter_messages(
        state["messages"],
        include_names=["tool_message"],
        include_types=[ToolMessage],
    )

    used_docs = []
    for msg in tool_messages:
        metadata_match = METADATA_PATTERN.search(msg.content)

        if metadata_match:
            try:
                metadata = json.loads(metadata_match.group(1))
                used_docs.append(
                    {
                        "Nombre del documento": metadata.get("title", ""),
                        "Fuente": metadata.get("source", ""),
                    }
                )
            except json.JSONDecodeError:
                continue

    messages_to_remove = [
        RemoveMessage(id=msg.id) for msg in tool_messages if msg.id is not None
    ]

    return {"messages": messages_to_remove, "used_docs": used_docs}


async def suggest_question(state: AgentState) -> AgentState:
    """Generate follow-up question based on last conversation exchange.

    Args:
        state (AgentState): Current conversation state

    Returns:
        dict: Generated follow-up question
    """
    relevant_messages = state["messages"][-2:]

    formatted_prompt = Q_SUGGESTION_TEMPLATE.format(
        user_input=relevant_messages[0].content,
        bot_response=relevant_messages[1].content,
    )

    response = await LLM.with_config({"run_name": "q_suggestion"}).ainvoke(
        formatted_prompt
    )
    response.name = "suggested_question"
    return {"suggested_question": response.content}


async def check_message_count(state: AgentState):
    """Determine next node based on conversation length.

    Args:
        state (AgentState): Current conversation state

    Returns:
        str: Next node - "suggest_question" or "summarize_conversation"
    """
    messages = filter_messages(
        state["messages"],
        include_types=[HumanMessage, AIMessage],
    )

    if len(messages) < 6:
        return "suggest_question"
    return "summarize_conversation"


async def summarize_conversation(state: AgentState):
    """Generate conversation summary and update system message.

    Args:
        state (AgentState): Current conversation state

    Returns:
        dict: Messages to remove after summarization
    """
    system_message = filter_messages(state["messages"], include_types=[SystemMessage])[
        0
    ]

    messages_to_summarize = filter_messages(
        state["messages"],
        exclude_types=[SystemMessage],
    )

    formatted_conversation = "\n\n".join(
        f"{'USUARIO' if isinstance(msg, HumanMessage) else 'BOT'}: {msg.content}"
        for msg in messages_to_summarize
    )

    formatted_prompt = SUMMARY_TEMPLATE.format(conversation=formatted_conversation)
    response = await LLM.ainvoke(formatted_prompt)

    formatted_template = RAG_TEMPLATE.format(summary=response.content)
    system_message.content = formatted_template

    messages_to_remove = [
        RemoveMessage(id=msg.id) for msg in messages_to_summarize if msg.id is not None
    ]

    return {"messages": messages_to_remove}


async def join_nodes(state: AgentState):
    """Pass through node for graph completion.

    Args:
        state (AgentState): Current conversation state

    Returns:
        AgentState: Unmodified state
    """
    return state


agent_builder = StateGraph(AgentState)
agent_builder.add_node(manage_system_prompt)
agent_builder.add_node(model)
agent_builder.add_node("tools", ToolNode(tools=TOOLS))
agent_builder.add_node(clean_messages)
agent_builder.add_node(suggest_question)
agent_builder.add_node(summarize_conversation)
agent_builder.add_node(join_nodes)

agent_builder.set_entry_point("manage_system_prompt")
agent_builder.add_edge("manage_system_prompt", "model")

agent_builder.add_conditional_edges(
    "model",
    pending_tool_calls,
    {"tools": "tools", "clean_messages": "clean_messages"},
)

agent_builder.add_edge("tools", "model")

agent_builder.add_conditional_edges(
    "clean_messages",
    check_message_count,
    {
        "summarize_conversation": "summarize_conversation",
        "suggest_question": "suggest_question",
    },
)

agent_builder.add_edge("suggest_question", "join_nodes")
agent_builder.add_edge("summarize_conversation", "join_nodes")
agent_builder.set_finish_point("join_nodes")

agent_graph = agent_builder.compile(checkpointer=MemorySaver()).with_config(
    {"run_name": "Agente AquaChile"}
)
