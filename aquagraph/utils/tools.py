import os

from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

from aquagraph.utils.retriever import CustomAzureAISearchRetriever

AZURE_AI_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

document_prompt = PromptTemplate.from_template("""
Nombre del documento: {title}
Fuente: {source}
Pagina: {page}
Contenido:{page_content}
===
""")

retriever = CustomAzureAISearchRetriever(
    content_key="content",
    top_k=5,
    index_name=AZURE_AI_SEARCH_INDEX_NAME,
)

retriever_tool = create_retriever_tool(
    retriever,
    name="aquachile-retriever",
    description="search and retrieve information from AquaChile's knowledge base",
    document_prompt=document_prompt,
)

TOOLS = [retriever_tool]
