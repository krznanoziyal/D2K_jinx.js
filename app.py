import streamlit as st
import httpx
import json
from typing import List, Dict
import time
import os

st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

# Set up the page structure
st.title("📊 Financial Statement Analysis Chat")
st.markdown("Upload your financial documents (PDFs, CSVs, Excel) and ask questions to get instant analysis.")

# Add instructions for the user
with st.expander("How to use this app"):
    st.markdown("""
    **Getting Started:**
    1. Upload a financial document using the sidebar (supports PDF, CSV, Excel)
    2. Ask a question about the document in the chat
    3. For the initial question after uploading, provide context like: "Analyze this financial statement" or "What are the key financial metrics?"
    
    **Example Questions:**
    - "Analyze this financial statement and provide key metrics"
    - "What is the company's current ratio?"
    - "Calculate profitability ratios based on this data"
    - "What are the trends in revenue growth?"
    - "Identify any red flags in this financial statement"
    """)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "document_name" not in st.session_state:
    st.session_state.document_name = None

# Function to handle regular chat
def process_chat(prompt):
    try:
        with st.spinner("Thinking..."):
            # Prepare chat history for the API
            chat_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
            
            # Add the current message
            chat_messages.append({"role": "user", "content": prompt})
            
            # Call the API
            response = httpx.post(
                "http://127.0.0.1:8000/chat",
                json={
                    "messages": chat_messages,
                    "session_id": st.session_state.session_id
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                return data["response"]
            else:
                st.error(f"Error: {response.text}")
                return f"Error communicating with backend: {response.status_code}"
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Function to handle document upload and chat
def process_document_chat(file, prompt):
    try:
        with st.spinner("Processing document and analyzing..."):
            # Create a multipart form
            files = {"file": (file.name, file.getvalue(), "application/octet-stream")}
            form_data = {
                "prompt": prompt,
                "session_id": st.session_state.session_id if st.session_state.session_id else "",
            }
            
            # Call the API
            response = httpx.post(
                "http://127.0.0.1:8000/chat_with_document",
                files=files,
                data=form_data,
                timeout=120.0  # Longer timeout for document processing
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.document_uploaded = True
                st.session_state.document_name = file.name
                return data["response"]
            else:
                st.error(f"Error: {response.text}")
                return f"Error processing document: {response.status_code}"
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Document upload section
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a financial document (PDF, CSV, Excel)",
        type=["pdf", "csv", "xlsx", "xls"],
        help="Upload a financial document to analyze"
    )
    
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' is ready for analysis")
    
    if st.session_state.document_uploaded:
        st.info(f"Currently analyzing: {st.session_state.document_name}")
    
    st.header("About")
    st.markdown("This is a prototype for an AI-driven Financial Statement Analysis Platform.")
    st.markdown("You can chat with the AI or upload documents for analysis.")
    
    # Add a button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.document_uploaded = False
        st.session_state.document_name = None
        st.experimental_rerun()

# Chat input section
prompt = st.chat_input("Ask a question about financial statements...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process file if it was uploaded alongside the prompt
    if uploaded_file and not st.session_state.document_uploaded:
        response_text = process_document_chat(uploaded_file, prompt)
    else:
        response_text = process_chat(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
