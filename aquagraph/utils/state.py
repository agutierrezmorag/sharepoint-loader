from typing import TypedDict

from langgraph.graph import MessagesState


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    suggested_question: str


class AgentState(MessagesState, InputState, OutputState):
    pass
