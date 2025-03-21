from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List, Optional

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Format messages for Gemini API
        history = []
        for msg in request.messages:
            if msg.role == "user":
                history.append({"role": "user", "parts": [msg.content]})
            else:
                history.append({"role": "model", "parts": [msg.content]})
        
        # Start a chat session
        chat_session = model.start_chat(history=history[:-1] if history else [])
        
        # Get the last user message
        last_message = request.messages[-1].content if request.messages else ""
        
        # Generate response
        response = chat_session.send_message(last_message)
        return ChatResponse(response=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
