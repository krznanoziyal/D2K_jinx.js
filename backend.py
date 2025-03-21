from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiChat:
    def __init__(self):
        # Initialize the Gemini model with API key from environment variables
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.7,
        )
        
        # Setup conversation memory
        self.memory = ConversationBufferMemory()
        
        # Create conversation chain
        self.conversation = ConversationChain(
            llm=self.model,
            memory=self.memory,
            verbose=True
        )
    
    def get_response(self, user_input):
        """Get a response from the Gemini model"""
        try:
            response = self.conversation.predict(input=user_input)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
