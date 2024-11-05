from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph.message import MessagesState

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

retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=3,
    index_name="sharepoint-index",
)

sharepoint_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="sharepoint_retriever",
    description="Busca y retorna información de documentos subidos a Sharepoint en base a una consulta.",
)
