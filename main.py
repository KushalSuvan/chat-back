import asyncio
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Dict, Optional
import uvicorn

from models import Eunoia
from models.eunoia.summarizer import run_once as summarizer
from models.document_loader import web_loader
from utils.login.helpers import User



from langchain_cohere import CohereEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

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

async def query(input_string: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    async for chunk in model(input_string, config):
        yield chunk
        await asyncio.sleep(0.05)


app = FastAPI()

sessions: Dict[str, Dict] = {}


@app.websocket("/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    print ("GOT REQUEST FOR STREAMING !!!")

    await websocket.accept()
    if session_id not in sessions:
        sessions[session_id] = {"history": []}

    try:
        while True:
            data = await websocket.receive_text()
            message = data.strip()
            if not message:
                continue

            # Call your model with message + session_id
            async for token in model(message, session_id):  # Assumes generator or iterable
                await websocket.send_text(token)

            # Optionally store full message + response in session history
            sessions[session_id]["history"].append({
                "user": message,
                "bot": "STREAMED"
            })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")


@app.get("/chat/{session_id}")
async def get_chat(session_id: str):
    print("GOT REQUEST FOR CHAT HISTORY !!!")

    # Get full history from LangGraph memory
    try:
        messages = await model.get_history(session_id)  # `model` is your Eunoia instance

        # Format messages into a simplified role-based structure
        formatted = [
            {"role": "user" if "message" in msg else "assistant", "content": msg}
            for msg in messages['history']
        ]

        long_chat = ""
        for msg in messages["history"]: long_chat+=msg
        summ_state = {
            "messages": [HumanMessage(content=long_chat)]
        }
        result = await summarizer(summ_state)
        print(f'RESULT: {result}')
        result['history'] = formatted

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve chat history: {str(e)}"}
        )
    
@app.post("/company/{company}")
async def get_chat(session_id: str):
    print("GOT REQUEST FOR CHAT HISTORY !!!")

    # Get full history from LangGraph memory
    try:
        messages = await model.get_history(session_id)  # `model` is your Eunoia instance

        # Format messages into a simplified role-based structure
        formatted = [
            {"role": "user" if "message" in msg else "assistant", "content": msg}
            for msg in messages['history']
        ]

        long_chat = ""
        for msg in messages["history"]: long_chat+=msg
        summ_state = {
            "messages": [HumanMessage(content=long_chat)]
        }
        result = await summarizer(summ_state)
        print(f'RESULT: {result}')
        result['history'] = formatted

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve chat history: {str(e)}"}
        )

# Endpoint to end a session
# @app.delete("/end_session/{session_id}")
# async def end_session(session_id: str):
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")
#     del sessions[session_id]
#     return {"message": f"Session {session_id} ended"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


