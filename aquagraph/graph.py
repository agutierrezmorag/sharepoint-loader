import os
import sys

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from aquagraph.utils.nodes import (
    clean_messages,
    manage_system_prompt,
    model,
    pending_tool_calls,
    suggest_question,
    summarize_conversation,
)
from aquagraph.utils.state import AgentState, InputState, OutputState
from aquagraph.utils.tools import TOOLS

agent_builder = StateGraph(AgentState, input=InputState, output=OutputState)
agent_builder.add_node(manage_system_prompt)
agent_builder.add_node(model)
agent_builder.add_node("tools", ToolNode(tools=TOOLS))
agent_builder.add_node(clean_messages)
agent_builder.add_node(suggest_question)
agent_builder.add_node(summarize_conversation)

agent_builder.set_entry_point("manage_system_prompt")
agent_builder.add_edge("manage_system_prompt", "model")

agent_builder.add_conditional_edges(
    "model",
    pending_tool_calls,
    {"tools": "tools", "clean_messages": "clean_messages"},
)

agent_builder.add_edge("tools", "model")

agent_builder.add_edge("clean_messages", "suggest_question")
agent_builder.add_edge("clean_messages", "summarize_conversation")

agent_builder.add_edge("suggest_question", END)
agent_builder.add_edge("summarize_conversation", END)

agent_graph = agent_builder.compile(
    checkpointer=MemorySaver(),
).with_config({"run_name": "Agente AquaChile"})
