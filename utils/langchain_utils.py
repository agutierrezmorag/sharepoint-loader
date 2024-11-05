import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks.tracers import LangChainTracer
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools.simple import Tool
from langchain_openai import AzureChatOpenAI

load_dotenv()


@st.cache_resource(show_spinner=False)
def _get_llm():
    return AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        max_retries=2,
    )


@st.cache_resource(show_spinner=False)
def _get_tools() -> list[Tool]:
    retriever = AzureAISearchRetriever(
        content_key="content",
        top_k=5,
        index_name="sharepoint-index",
    )

    document_prompt = PromptTemplate.from_template(
        "Fuente: {page_content}\n{metadata}\n ==="
    )
    retriever_tool = create_retriever_tool(
        retriever,
        "document_retriever",
        "Busca y retorna información de documentos reglamentales de AquaChile.",
        document_prompt=document_prompt,
    )
    return [retriever_tool]


def set_tracer(project_name):
    return LangChainTracer(project_name=project_name)


@st.cache_resource(show_spinner=False)
def _get_agent_prompt():
    return hub.pull("alvgutierrez/aquachile-rag-agent")


@st.cache_resource(show_spinner=False)
def q_suggestion_chain():
    llm = _get_llm()
    prompt = hub.pull("alvgutierrez/aquachile-q-suggestion")

    chain = (
        {"input": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    ).with_config({"run_name": "Suggested Question"})
    return chain


def get_agent():
    llm = _get_llm()
    tools = _get_tools()
    prompt = _get_agent_prompt()

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        max_iterations=5,
        max_execution_time=30.0,
        return_intermediate_steps=True,
    ).with_config({"run_name": "AquaChile Agent"})
    return agent_executor
