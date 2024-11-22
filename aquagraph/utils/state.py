from typing import TypedDict

from langgraph.graph import MessagesState


class InputState(TypedDict):
    """Type definition for conversation input state.

    Attributes:
        user_input (str): The raw input text from the user
    """

    user_input: str


class OutputState(TypedDict):
    """Type definition for conversation output state.

    Attributes:
        suggested_question (str): Generated follow-up question suggestion
    """

    suggested_question: str


class AgentState(MessagesState, InputState, OutputState):
    """Combined state type for the conversation agent.

    Inherits from:
        MessagesState: Base message handling state
        InputState: User input state
        OutputState: Generated output state
    """

    pass
