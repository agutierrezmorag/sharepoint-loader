import os

from dotenv import load_dotenv
from langchain_core.runnables import ConfigurableField
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

LLM = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    model_version=AZURE_OPENAI_MODEL,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    max_tokens=1000,
    temperature=0.7,
    streaming=True,
).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)
