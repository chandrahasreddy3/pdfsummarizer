import streamlit as st
import logging
import os
import requests
from datetime import datetime

# Import frontend components
from frontend.document_manager import DocumentManager
from frontend.chat_interface import ChatInterface
from config import BACKEND_URL, GEMINI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Project Knowledge Transfer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_environment():
    """Check if required environment variables are set."""
    missing_vars = []
    
    if not GEMINI_API_KEY:
        missing_vars.append("GEMINI_API_KEY")
    
    return missing_vars

def render_header():
    """Render the application header."""
    st.title("🤖 Project Knowledge Transfer Chatbot")
    st.markdown("""
    **Welcome to your AI-powered project knowledge assistant!**
    
    Upload your project documents (PDFs, Markdown, or text files) and ask questions about your project. 
    The chatbot uses advanced RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents.
    """)

def render_sidebar():
    """Render the sidebar with application information."""
    with st.sidebar:
        # Application info
        st.header("📋 About")
        st.markdown("""
        **Features:**
        - 📄 Multi-format document upload
        - 🔍 Semantic search across documents
        - 💬 Natural language Q&A
        - 📚 Source attribution
        """)
        
        st.markdown("""
        **Supported Formats:**
        - PDF (.pdf)
        - Markdown (.md)
        - Text (.txt)
        """)

def main():
    """Main application function."""
    # Render header
    render_header()
    
    # Check environment
    missing_vars = check_environment()
    if missing_vars:
        st.error("⚠️ Application not properly configured. Please check the sidebar for setup instructions.")
        render_sidebar()
        return
    
    # Initialize components
    try:
        document_manager = DocumentManager()
        chat_interface = ChatInterface()
    except Exception as e:
        st.error(f"❌ Failed to initialize application components: {str(e)}")
        st.markdown("Please check your environment configuration and try again.")
        render_sidebar()
        return
    
    # Main application layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Document management section
        document_manager.render_upload_section()
    
    with col2:
        # Chat interface section
        try:
            document_count = document_manager.get_document_count()
            chat_interface.render_chat_interface(document_count)
        except Exception as e:
            st.error(f"❌ Chat interface error: {str(e)}")
            st.markdown("Please check if the backend server is running and try again.")
    
    # Render sidebar
    render_sidebar()

def run_instructions():
    """Display instructions if the app is not running properly."""
    st.error("🚫 Application Startup Error")
    st.markdown("""
    ## Setup Instructions
    
    ### 1. Environment Variables
    Set your Gemini API key:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```
    
    ### 2. Start the Backend
    ```bash
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    
    ### 3. Start the Frontend
    ```bash
    streamlit run app.py --server.port 5000
    ```
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        run_instructions()
    