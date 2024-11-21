import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from aquagraph.graph import agent_graph  # noqa: E402


async def answer_question(question, response_placeholder):
    agent = agent_graph
    full_response = ""
    suggested_question = ""

    try:
        async for event in agent.astream(
            {"messages": [HumanMessage(content=question)]},
            config=st.session_state.configurable,
            stream_mode="messages",
        ):
            langgraph_node = event[1].get("langgraph_node", "lol")

            if langgraph_node == "model":
                full_response += event[0].content
                response_placeholder.markdown(full_response + "▌")

            if langgraph_node == "suggest_question":
                suggested_question += event[0].content

        st.session_state.suggested_question = suggested_question
        st.session_state.user_question = ""
        return full_response
    except Exception as e:
        st.error(f"Error: {e}")
        return


def submit_question(question):
    st.session_state.user_question = question


if __name__ == "__main__":
    st.set_page_config(page_title="Agente AquaChile", page_icon="🐟")
    if "configurable" not in st.session_state:
        st.session_state.configurable = {
            "configurable": {"thread_id": str(uuid.uuid4())}
        }
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "suggested_question" not in st.session_state:
        st.session_state.suggested_question = None
    if "msgs" not in st.session_state:
        st.session_state.msgs = StreamlitChatMessageHistory(key="msgs")

    st.markdown(
        """
    <style>
    .element-container:has(style){
        display: none;
    }
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
        border: none;
        background: none;
        font-style: italic;
        text-align: left;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if len(st.session_state.msgs.messages) == 0:
        st.chat_message("ai").markdown("""
**👋¡Hola!**
Soy un asistente virtual. Estoy aquí para responder tus dudas sobre reglamentos y \
documentos del área de *Riesgo corporativo* de AquaChile. Pregunta lo que necesites y te ayudaré a encontrar la información.
""")

    for msg in st.session_state.msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if question := st.chat_input(placeholder="Escribe tu pregunta..."):
        submit_question(question)

    if st.session_state.user_question != "":
        st.session_state.msgs.add_user_message(
            HumanMessage(content=st.session_state.user_question)
        )
        st.chat_message("human").markdown(st.session_state.user_question)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            ai_answer = asyncio.run(
                answer_question(st.session_state.user_question, response_placeholder)
            )
            st.session_state.msgs.add_ai_message(AIMessage(content=ai_answer))

    if st.session_state.suggested_question:
        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        st.button(
            f"✨ {st.session_state.suggested_question}",
            on_click=submit_question,
            args=(st.session_state.suggested_question,),
        )
