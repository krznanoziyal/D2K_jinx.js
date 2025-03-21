import streamlit as st
import httpx
from typing import List, Dict

st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Statement Analysis Chat")
st.markdown("Ask questions about financial statements or upload documents for analysis.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about financial statements..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare the chat history for the API request
    chat_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
    
    # Send message to backend
    with st.spinner("Thinking..."):
        try:
            response = httpx.post(
                "http://127.0.0.1:8000/chat",
                json={"messages": chat_messages},
                timeout=30.0
            )
            
            if response.status_code == 200:
                response_data = response.json()
                assistant_response = response_data["response"]
            else:
                assistant_response = f"Error: {response.text}"
        except Exception as e:
            assistant_response = f"Error connecting to backend: {str(e)}"
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("This is a prototype for an AI-driven Financial Statement Analysis Platform.")
    st.markdown("Currently in basic chat mode - document upload functionality coming soon.")
    
    st.header("Instructions")
    st.markdown("1. Type a question in the chat box")
    st.markdown("2. The AI will respond with relevant information")
    st.markdown("3. Future versions will support document upload and analysis")
