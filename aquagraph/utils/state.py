from typing import TypedDict

from langgraph.graph import MessagesState


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    response: str
    suggested_question: str
    used_docs: list[dict]


class AgentState(MessagesState, InputState, OutputState):
    pass
