from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.prompts import PromptTemplate

document_prompt = PromptTemplate.from_template(
    "Metadata: {metadata} \nFuente:{page_content}\n ==="
)

retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=5,
    index_name="sharepoint-index",
)

retriever_tool = create_retriever_tool(
    retriever,
    name="aquachile-retriever",
    description="search and retrieve information from AquaChile's knowledge base",
    document_prompt=document_prompt,
)

TOOLS = [retriever_tool]
