import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_utils import get_agent

if __name__ == "__main__":
    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "results" not in st.session_state:
        st.session_state.results = None
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.title("AquaChile Doc Q&A")
    agent = get_agent()

    if question := st.chat_input():
        st.session_state.results = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

    if not st.session_state.results:
        st.stop()

    for message_str in st.session_state.results["messages"]:
        if message_str.content.strip() != "":
            if isinstance(message_str, HumanMessage):
                with st.chat_message("human"):
                    st.markdown(message_str.content)
            elif isinstance(message_str, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(message_str.content)
