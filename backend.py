from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.genai import types
from dotenv import load_dotenv
import os
from typing import List, Optional
import tempfile
import uuid
from pathlib import Path
import json

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware for Streamlit to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for temp files if it doesn't exist
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store active chat sessions
chat_sessions = {}

# Data models
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
        
        # Format messages for Gemini API
        history = []
        for msg in request.messages:
            if msg.role == "user":
                history.append({"role": "user", "parts": [msg.content]})
            else:
                history.append({"role": "model", "parts": [msg.content]})
        
        # Initialize or retrieve chat session
        if session_id not in chat_sessions:
            # Initialize Gemini model - using gemini-2.0-flash as specified
            model = genai.GenerativeModel('gemini-2.0-flash')
            chat_sessions[session_id] = model.start_chat(history=history[:-1] if history else [])
        
        chat_session = chat_sessions[session_id]
        
        # Get the last user message
        last_message = request.messages[-1].content if request.messages else ""
        
        # Generate response
        response = chat_session.send_message(last_message)
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_with_document", response_model=ChatResponse)
async def chat_with_document(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    try:
        session_id = session_id or str(uuid.uuid4())
        
        # Create a temporary file to store the uploaded document
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Determine the MIME type based on file extension
        mime_type = "application/pdf"
        if file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        # Initialize Gemini model - using gemini-2.0-flash as specified
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Pass the document directly to Gemini using its native document understanding
        document_part = types.Part.from_bytes(
            data=file_path.read_bytes(),
            mime_type=mime_type
        )
        
        # Send the document along with the prompt - Gemini will handle the PDF processing internally
        response = model.generate_content([document_part, prompt])
        
        # Store the session for future interactions
        chat_sessions[session_id] = model.start_chat(
            history=[
                {"role": "user", "parts": [f"I've uploaded a document. {prompt}"]},
                {"role": "model", "parts": [response.text]}
            ]
        )
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)
        
        return ChatResponse(response=response.text, session_id=session_id)
        
    except Exception as e:
        # Clean up in case of error
        if 'file_path' in locals():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
