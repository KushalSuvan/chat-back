import asyncio

from models.eunoia import Eunoia
from models.document_loader import web_loader

from langchain_cohere import CohereEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore

import sys
import inspect


llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vector_store = InMemoryVectorStore(embeddings)

docs = {
    "mavic": [
        "https://www.dji.com/global/support/product/mavic-4-pro",
        "https://www.dji.com/global/support/product/mavic-3-pro"
    ]
}

for company, urls in docs.items():
    web_loader(urls, vector_store, company)

model = Eunoia(llm, embeddings, vector_store)

async def query_and_print(input_string: str, config: str):
    count = 0
    async for chunk in model(input_string, config):
        if count == 0: 
            count+=1
            continue
        print(chunk, end='', flush=True)
        await asyncio.sleep(0.05)
    print()

async def main():
    config = {"configurable": {"thread_id": "abc123"}}
    await query_and_print("Hi", config)
    await query_and_print("My name is KushaSuvan", config)
    await query_and_print("I wanted to inquire about the company Mavic 3", config)


if __name__=="__main__":
    asyncio.run(main())