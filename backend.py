from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from typing import List, Optional, Dict, Any
import uuid
from pathlib import Path
import logging

load_dotenv()

# Configure the Gemini API using the newer library's client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],  # Adjust for production!
    allow_headers=["*"],
)

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Enhanced session storage that tracks documents and message history
chat_sessions = {}  # session_id -> chat object
session_documents = {}  # session_id -> list of document parts
session_history = {}  # session_id -> list of messages

logging.basicConfig(level=logging.INFO)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
        
        chat_session = chat_sessions[session_id]
        user_message_content = request.messages[-1].content if request.messages else ""
        
        # Prepare message parts: first any documents in context, then the new message
        message_parts = []
        
        # Add any documents associated with this session
        if session_documents.get(session_id):
            message_parts.extend(session_documents[session_id])
        
        # Add the user's text message
        message_parts.append(types.Part.from_text(text=user_message_content))
        
        # Send the message with all context
        if len(message_parts) == 1:
            # Just a text message, no documents
            response = chat_session.send_message(user_message_content)
        else:
            # Message with document context
            response = chat_session.send_message(message_parts)
        
        # Store in history
        session_history[session_id].append({"role": "user", "content": user_message_content})
        session_history[session_id].append({"role": "assistant", "content": response.text})
        
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        logging.exception("Error in /chat endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_document", response_model=ChatResponse)
async def upload_document(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    file_path = None
    try:
        session_id = session_id or str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"File saved to {file_path} ({len(content)} bytes)")
        
        # Determine MIME type
        mime_type = "application/pdf"  # default
        if file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        logging.info(f"Using MIME type: {mime_type} for {file.filename}")
        
        # Read file data
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Create document part
        document_part = types.Part.from_bytes(
            data=file_data,
            mime_type=mime_type
        )
        
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
        
        # Store document for future context
        session_documents[session_id].append(document_part)
        
        # Create the message with document and prompt
        message_parts = [document_part, types.Part.from_text(text=prompt)]
        
        # Send the message
        chat_session = chat_sessions[session_id]
        response = chat_session.send_message(message_parts)
        
        # Store in history
        session_history[session_id].append({
            "role": "user", 
            "content": f"[Uploaded document: {file.filename}] {prompt}"
        })
        session_history[session_id].append({"role": "assistant", "content": response.text})
        
        # Clean up file
        if file_path:
            file_path.unlink(missing_ok=True)
        
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        if file_path:
            file_path.unlink(missing_ok=True)
        logging.exception("Error in upload_document endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    if session_id not in session_history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": session_history[session_id]}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)