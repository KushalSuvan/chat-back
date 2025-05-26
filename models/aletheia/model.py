from dotenv import load_dotenv

load_dotenv()

from models.eunoia.graph import get_graph


class Aletheia:

    def __init__(self, llm, embedder, vector_store):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph = get_graph(llm, vector_store)

    async def __call__(self, input_message, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        async for message, metadata in self.graph.astream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="messages",
            config=config
        ):
            yield message.content if metadata['langgraph_node'] != 'tools' else ''