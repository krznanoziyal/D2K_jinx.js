from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
import pandas as pd

# Load environment variables
load_dotenv()

# Configure environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class LangChainHandler:
    """Handler for LangChain operations"""
    
    def __init__(self):
        # Always use gemini-2.0-flash as specified
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def create_financial_analysis_chain(self):
        """Create a conversation chain specialized for financial analysis"""
        prompt_template = """You are a financial analyst expert specializing in analyzing financial statements.
        You provide clear, detailed responses based on the financial data presented.
        
        Always structure your response with clear sections and bullet points where appropriate.
        Calculate key financial ratios using the data you see in the document.
        
        Current conversation:
        {history}
        Human: {input}
        AI:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=prompt_template
        )
        
        chain = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=True
        )
        
        return chain
    
    # Financial analysis prompts can be added here as needed
    def get_financial_analysis_prompt(self):
        """Return a prompt for financial statement analysis"""
        return """Analyze the following financial document:
        
        1. Extract key financial metrics:
           - Total Revenue
           - Net Income
           - Total Assets
           - Total Liabilities
           - Equity
           
        2. Calculate the following financial ratios:
           - Current Ratio (Current Assets / Current Liabilities)
           - Debt-to-Equity Ratio (Total Debt / Total Equity)
           - Return on Assets (Net Income / Total Assets)
           - Return on Equity (Net Income / Shareholder's Equity)
           - Profit Margin (Net Income / Revenue)
           
        3. Provide a brief analysis of the company's financial health.
        
        4. Highlight any red flags or areas of concern.
        
        5. Suggest potential areas for improvement.
        
        Format your response with clear sections and bullet points."""

    def process_document(self, file_path, file_type):
        """Process a document for retrieval-based QA"""
        try:
            # Create document based on file type
            if file_type.lower() == "pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_type.lower() == "csv":
                loader = CSVLoader(file_path)
                documents = loader.load()
            else:
                # For other file types we'll implement later
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vectorstore
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Create retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(),
                memory=self.memory,
                return_source_documents=True
            )
            
            return qa_chain
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise
    
    # Additional methods to be implemented as needed

# Placeholder for future functionality
def analyze_financial_ratios(financial_data):
    """Analyze financial ratios based on extracted data"""
    # This will be implemented in future iterations
    pass
