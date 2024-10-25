import asyncio
import uuid

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_utils import get_agent


async def answer_question(agent, config, question, response_placeholder):
    full_response = ""

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        version="v2",
        include_types=["chat_model"],
    ):
        event_type = event["event"]
        if event_type == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                full_response += content
                response_placeholder.markdown(full_response + "▌")

    return full_response


if __name__ == "__main__":
    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "question" not in st.session_state:
        st.session_state.question = ""

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
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
            ai_answer = asyncio.run(
                answer_question(agent, config, question, response_placeholder)
            )
        msgs.add_ai_message(ai_answer)
