import asyncio
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from utils.langchain_utils import get_agent, q_suggestion_chain

load_dotenv()


async def answer_question(question, agent_thoughts_placeholder, response_placeholder):
    question_suggestion_chain = q_suggestion_chain()
    agent = get_agent()
    full_response = ""
    full_tool_output = ""

    try:
        async for event in agent.astream_events(
            {"input": question},
            config={
                "metadata": {"conversation_id": st.session_state.conversation_id},
            },
            version="v2",
        ):
            event_type = event["event"]
            if event_type == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
            if event_type == "on_chain_end":
                if event["name"] == "AquaChile Agent":
                    agent_thoughts_placeholder.update(
                        label="🎣 Respuesta generada.",
                        expanded=False,
                        state="complete",
                    )
                    await asyncio.sleep(0.1)
                    last_two_messages = st.session_state.memory.buffer[-2:]
                    st.session_state.suggested_question = (
                        question_suggestion_chain.invoke(last_two_messages)
                    )
            elif event_type == "on_tool_start":
                tool_name = event["name"]
                query = event["data"].get("input")["query"]
                if tool_name == "sharepoint_retriever":
                    agent_thoughts_placeholder.markdown(
                        f"- 🐟🔎 Consultando *{query}* en los **documentos**..."
                    )
            elif event_type == "on_tool_end":
                output = event["data"].get("output")
                if output:
                    agent_thoughts_placeholder.markdown("- 🎏 Textos relevantes:")
                    cleaned_output = re.sub(
                        r'\{\s*"source":.*?\}', "", output, flags=re.DOTALL
                    )
                    full_tool_output += cleaned_output
                    agent_thoughts_placeholder.text_area(
                        "Contexto",
                        help="La IA utiliza este contexto para generar la respuesta. \
                                    Estos textos provienen de una variedad de reglamentos de la empresa.",
                        value=full_tool_output,
                        disabled=True,
                    )
    except Exception as e:
        st.error(f"Error: {e}")
        return

    st.session_state.user_question = ""


def submit_question(question):
    st.session_state.user_question = question


if __name__ == "__main__":
    st.set_page_config(page_title="Agente AquaChile", page_icon="🐟")
    if "msgs" not in st.session_state:
        st.session_state.msgs = StreamlitChatMessageHistory(key="msgs")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            chat_memory=st.session_state.msgs,
            return_messages=True,
        )
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "suggested_question" not in st.session_state:
        st.session_state.suggested_question = None

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
        st.chat_message("human").markdown(st.session_state.user_question)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            agent_thoughts_placeholder = st.status("🤔 Pensando...", expanded=False)
            ai_answer = asyncio.run(
                answer_question(
                    st.session_state.user_question,
                    agent_thoughts_placeholder,
                    response_placeholder,
                )
            )
    if st.session_state.suggested_question:
        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        st.button(
            f"✨ {st.session_state.suggested_question}",
            on_click=submit_question,
            args=(st.session_state.suggested_question,),
        )
