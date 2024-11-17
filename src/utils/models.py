from langchain_openai import ChatOpenAI

LLM = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1000,
    temperature=0.7,
    streaming=True,
)
