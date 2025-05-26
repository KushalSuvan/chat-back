
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState

from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

class Article(BaseModel):
    title: str
    description: str


class AgentState(MessagesState):
    title: Optional[str] = None
    description: Optional[str] = None
    final_response: Optional[Article] = None


from langchain_core.tools import tool

@tool
def extract_title(title: str) -> str:
    """Extract a title from the given user query or content. Make it a short essence of what the content is about in less than 6 words"""
    return title

@tool
def extract_description(description: str) -> str:
    """Extract a detailed description from the user query or content. Make it four to five lines of description of what the content is about"""
    return description


from langchain.chat_models import init_chat_model

tools = [extract_title, extract_description]
model = init_chat_model("llama3-8b-8192", model_provider="groq")
model_with_tools = model.bind_tools(tools, tool_choice="any")

from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)

def apply_tool_result(state: AgentState):
    last_msg = state["messages"][-1]
    print(last_msg)
    tool_call = last_msg.name
    result = state["messages"][-1].content  # the result from the tool call

    updated = {}
    if tool_call == "extract_title":
        updated["title"] = result
    elif tool_call == "extract_description":
        updated["description"] = result

    return updated


def construct_article(state: AgentState):
    if state["title"] and state["description"]:
        article = Article(title=state["title"], description=state["description"])
        return {"final_response": article}
    return {}



async def call_model(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    system_message = {"role": "system", "content": "You are a helpful assistant."}

    messages = [system_message] + state["messages"] 

    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]


    # if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    #     return "tools"

    if not ("title" in state) or not ("description" in state):
        return "tools"  # ask to extract description
    else:
        return "finish"

    return "tools"


from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("apply_result", apply_tool_result)
workflow.add_node("finish", construct_article)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "finish": "finish"
    }
)

workflow.add_edge("tools", "apply_result")
workflow.add_edge("apply_result", "agent")
workflow.add_edge("finish", END)

graph = workflow.compile()

import asyncio
from langchain_core.messages import HumanMessage

# A single user query
user_input = "Om shanti. Whether it is this world of sinful souls or that world of pure, charitable souls, the world that is mentioned refers to the type of souls. It is only because there is sorrow here that people call out. No one in that world of pure, charitable souls calls out to be taken elsewhere. You children understand that it is not a pundit, sannyasi or scholar who is telling you all of this. This one himself says: I"

# Initial agent state with the user message
initial_state = {
    "messages": [HumanMessage(content=user_input)]
}

# Run the graph
async def run_once(state):
    final_response = await graph.ainvoke(state)
    # print("Final Output:")
    # print(final_response["final_response"])
    return {
        'title': final_response['final_response'].title,
        'description': final_response['final_response'].description
    }

async def stream_run(state):
    final_response = None
    async for step in graph.astream(state):
        # Each `step` is a dict with node outputs
        print("🔹 Step output:")
        for key, value in step.items():
            # print(f"Node: {key}")
            if isinstance(value, dict):
                # for k, v in value.items():
                    # print(f"  {k}: {v}")
                final_response = value
            # else:
                # print(f"  Output: {value}")
        # print("---------")
    return {
        'title': final_response['final_response'].title,
        'description': final_response['final_response'].description
    }

# Run the async function
if __name__=="__main__": print(asyncio.run(run_once()))
