import asyncio
import time
import uuid

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_utils import get_agent, set_tracer


async def answer_question(
    agent, config, question, agent_thoughts_placeholder, response_placeholder
):
    full_response = ""
    full_tool_output = ""

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        version="v2",
    ):
        event_type = event["event"]
        if event_type == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                full_response += content
                response_placeholder.markdown(full_response + "▌")
        if event_type == "on_chain_end":
            print(event["name"])
            if event["name"] == "LangGraph":
                time.sleep(1)
                agent_thoughts_placeholder.update(
                    label="🎣 Respuesta generada.",
                    expanded=False,
                    state="complete",
                )
        elif event_type == "on_tool_start":
            tool_name = event["name"]
            query = event["data"].get("input")["query"]
            if tool_name == "sharepoint_retriever":
                agent_thoughts_placeholder.markdown(
                    f"- 🐟🔎 Consultando *{query}* en **sharepoint**..."
                )
        elif event_type == "on_tool_end":
            output = event["data"].get("output")
            if output:
                agent_thoughts_placeholder.markdown(
                    "- 🎏 Creo haber encontrado textos relevantes:"
                )
                full_tool_output += output.content
                agent_thoughts_placeholder.text_area(
                    "Contexto",
                    help="La IA utiliza este contexto para generar la respuesta. \
                                Estos textos provienen de una variedad de reglamentos y documentos generales de la universidad.",
                    value=full_tool_output,
                    disabled=True,
                )

    return full_response


if __name__ == "__main__":
    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "question" not in st.session_state:
        st.session_state.question = ""

    config = {
        "callbacks": [set_tracer("Sharepoint Q&A")],
        "configurable": {"thread_id": st.session_state.thread_id},
    }
    agent = get_agent()
    msgs = StreamlitChatMessageHistory(key="message_history")
    st.title("AquaChile Sharepoint Docs Q&A")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if question := st.chat_input():
        msgs.add_user_message(question)
        st.chat_message("human").markdown(question)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            agent_thoughts_placeholder = st.status("🤔 Pensando...", expanded=False)
            ai_answer = asyncio.run(
                answer_question(
                    agent,
                    config,
                    question,
                    agent_thoughts_placeholder,
                    response_placeholder,
                )
            )
        msgs.add_ai_message(ai_answer)
