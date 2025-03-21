from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

class LangChainHandler:
    """Placeholder for future LangChain operations.
    
    For now, we are not using document splitting, vectorstores, etc.
    """
    def __init__(self):
        # Use gemini-2.0-flash for consistency
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        
    # Future methods for multi-step prompts can be added here.
