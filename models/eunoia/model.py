from dotenv import load_dotenv

load_dotenv()

from models.eunoia.graph import get_graph


class Eunoia:

    def __init__(self, llm, embedder, vector_store):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph = get_graph(llm, vector_store) 

    async def __call__(self, input_message, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        print(thread_id)
        async for message, metadata in self.graph.astream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="messages",
            config=config
        ):
            yield message.content if metadata['langgraph_node'] != 'tools' else ''


    
    async def get_history(self, thread_id):
        """Get entire chat history for a given thread_id."""
        config = {"configurable": {"thread_id": thread_id}}
        states = self.graph.get_state(config)
        print("got states")
        import pprint
        history = []
        for message in states.values['messages']:
            if message.type == 'tool' or message.content == '': continue
            history.append(str(message.content))
        # pprint.pprint(f"STATE: {history}", indent=2, width=40)
        return {'history': history}
