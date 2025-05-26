
from contextlib import contextmanager
import os
import sys
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import ToolNode

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver


def get_retreive(vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    return retrieve



def get_query_or_respond(llm, tools):
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        try:
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except:
            query_or_respond(state)

    return query_or_respond




def get_tools(tools):
    tools = ToolNode(tools)
    return tools


def get_generate(llm):
    async def generate(state: MessagesState):
        """Generate answer."""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(
            message.content if hasattr(message, "content") else str(message)
            for message in tool_messages
        )

        system_message_content = (
            "You are an assistant for question-answering tasks."
            "Talk like a pirate"
            "First ask about company name about which the user wants to inquire about."
            "Proceed only if you have that company's information."
            "List all products you know about."
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = await llm.ainvoke(prompt)
        return {"messages": [response]}
    
    return generate



def get_graph(llm, vector_store):
    retrieve = get_retreive(vector_store)
    query_or_respond = get_query_or_respond(llm, [retrieve])
    tools = get_tools([retrieve])
    generate = get_generate(llm)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph