from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    trim_messages,
)

from aquagraph.utils.models import LLM
from aquagraph.utils.prompts import (
    Q_SUGGESTION_TEMPLATE,
    RAG_TEMPLATE,
    SUMMARY_TEMPLATE,
)
from aquagraph.utils.state import AgentState
from aquagraph.utils.tools import TOOLS


async def manage_system_prompt(state: AgentState):
    messages = state["messages"]
    set_prompt = messages and isinstance(messages[0], SystemMessage)

    if not set_prompt:
        formatted_template = RAG_TEMPLATE.format(summary="No hay resumen previo.")
        messages = [SystemMessage(content=formatted_template)] + messages

    messages.append(HumanMessage(content=state["user_input"]))
    return {"messages": messages}


async def model(state: AgentState):
    messages = state["messages"]
    llm_with_tools = LLM.bind_tools(TOOLS)
    response = await llm_with_tools.with_config({"run_name": "agent_answer"}).ainvoke(
        messages
    )
    return {"messages": [response], "response": response.content}


def pending_tool_calls(state: AgentState):
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        last_message.name = "tool_message"
        return "tools"
    return "clean_messages"


def clean_messages(state: AgentState):
    tool_messages = filter_messages(
        state["messages"],
        include_names=["tool_message"],
        include_types=[ToolMessage],
    )

    messages_to_remove = [
        RemoveMessage(id=msg.id) for msg in tool_messages if msg.id is not None
    ]

    return {"messages": messages_to_remove}


async def suggest_question(state: AgentState) -> AgentState:
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


def check_message_count(state: AgentState):
    messages = filter_messages(
        state["messages"],
        include_types=[HumanMessage, AIMessage],
    )

    if len(messages) < 6:
        return "end"
    return "summarize_conversation"


async def summarize_conversation(state: AgentState):
    system_message = state["messages"][0]

    # Instead of manual filtering, use trim_messages
    qa_messages = trim_messages(
        state["messages"],
        max_tokens=2,  # Keep last 2 messages
        token_counter=len,  # Count messages instead of tokens
        strategy="last",
        start_on=[HumanMessage, AIMessage],  # Only keep Human and AI messages
        include_system=False,  # Don't include system message since we handle it separately
    )

    # Get messages to remove (all except system and trimmed messages)
    messages_to_remove = [
        msg
        for msg in state["messages"]
        if msg not in [system_message, *qa_messages]
        and not isinstance(msg, RemoveMessage)
    ]

    # Create conversation string from messages to be removed
    formatted_conversation = "\n".join(
        f"{'USER' if isinstance(msg, HumanMessage) else 'BOT'}: {msg.content}"
        for msg in messages_to_remove
    )

    # Generate new summary only if there are messages to summarize
    if formatted_conversation:
        response = await LLM.with_config(config={"llm_temperature": 0.2}).ainvoke(
            SUMMARY_TEMPLATE.format(conversation=formatted_conversation)
        )
        # Update system message with new summary
        system_message.content = RAG_TEMPLATE.format(summary=response.content)

    # Return messages to remove (all except system and last Q&A pair)
    return {"messages": [RemoveMessage(id=msg.id) for msg in messages_to_remove]}
