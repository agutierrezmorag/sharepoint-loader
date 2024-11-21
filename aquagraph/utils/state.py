from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State class for managing conversation and suggested questions.

    This class extends MessagesState to track the state of an agent's conversation,
    including suggested follow-up questions and used documents.

    Attributes:
        suggested_question (str): The next question suggested by the agent to
            continue the conversation flow.
        used_docs (list[dict]): List of documents that have been referenced or
            used during the conversation. Each document is represented as a
            dictionary containing document metadata.

    Inherits:
        MessagesState: Base class for managing conversation message state.

    """

    suggested_question: str = ""
    used_docs: list[dict] = []
