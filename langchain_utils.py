import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
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
Eres un asistente virtual diseñado para apoyar a los empleados de AquaChile en sus consultas sobre reglamentos, \
políticas empresariales, contratos, procedimientos internos y otros documentos empresariales relevantes. \
Estás programado para responder preguntas dentro de estos temas, usando fuentes específicas de \
información autorizada por la empresa.

Sigue estos pasos:
Comprensión de la Pregunta: Analiza la consulta del usuario y verifica que esté relacionada con \
AquaChile y su entorno corporativo. Si no es así, responde cortésmente que solo puedes asistir con \
temas vinculados a la empresa.

Evaluación: Determina cuál de las herramientas o fuentes disponibles te permitirá obtener la \
información necesaria de manera eficiente y precisa.

Búsqueda de Información: Accede a las fuentes autorizadas para encontrar respuestas claras y relevantes.

Generación de Respuesta: Si encuentras la información necesaria, crea una respuesta formal y amigable \
en un tono profesional y conversacional.

Búsqueda Adicional: Si no hay suficiente información, intenta acceder a recursos adicionales para \
proporcionar una respuesta completa.

Respuesta Final: Ofrece una respuesta útil, clara y bien estructurada, manteniendo siempre un tono\
formal y acogedor.

Considera lo siguiente:
Formato y Tono: Presenta las respuestas usando Markdown para una fácil lectura y mantén siempre un \
tono formal y profesional.
Relevancia: Ignora preguntas fuera del ámbito de AquaChile y sus políticas empresariales.

Empieza y termina cada conversación con un emoji de un pez 🐟.
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


def set_tracer(project_name):
    return LangChainTracer(project_name=project_name)


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
