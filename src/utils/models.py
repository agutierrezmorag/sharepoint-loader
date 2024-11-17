import os

from langchain_openai import AzureChatOpenAI

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

LLM = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    model=AZURE_OPENAI_MODEL,
    max_tokens=1000,
    temperature=0.7,
    streaming=True,
)
