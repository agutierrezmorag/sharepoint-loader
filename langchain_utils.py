import streamlit as st
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.simple import Tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

load_dotenv()

LLM = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    max_retries=2,
)
EMBEDDINGS_MODEL = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    model="text-embedding-3-small",
)

SYS_PROMPT = """
Eres un asistente virtual que quiere ayudar a los empleados de la empresa AquaChile a encontrar \
información relevante sobre documentos subidos a Sharepoint y a responder preguntas sobre ellos. \
Mantén la conversación en un tono amigable y profesional. Empieza y termina cada conversación \
con un emoji de un pez 🐟.
"""


def _get_tools(retriever: BaseRetriever) -> list[Tool]:
    retriever_tool = create_retriever_tool(
        retriever,
        "sharepoint_retriever",
        "Busca y retorna información de documentos subidos a Sharepoint en base a una consulta.",
    )
    return [retriever_tool]


def _get_retriever():
    retriever = AzureAISearchRetriever(
        content_key="content",
        top_k=3,
        index_name="sharepoint-index",
    )
    return retriever


@st.cache_resource
def get_agent():
    retriever = _get_retriever()
    tools = _get_tools(retriever)

    agent = create_react_agent(
        LLM,
        tools,
        state_modifier=SYS_PROMPT,
        checkpointer=st.session_state.memory,
    )
    return agent
