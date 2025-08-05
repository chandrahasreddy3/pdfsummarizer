import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Project Knowledge Transfer Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Constants
BACKEND_URL = "http://localhost:8000"

def test_backend_connection():
    """Test if backend is accessible."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def get_documents():
    """Get list of documents from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def upload_document(uploaded_file):
    """Upload a single document to backend."""
    try:
        files = [('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))]
        response = requests.post(f"{BACKEND_URL}/upload-documents", files=files, timeout=30)
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

def ask_question(question, session_id="default"):
    """Ask a question to the chatbot."""
    try:
        data = {
            "query": question,
            "session_id": session_id,
            "max_results": 5
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=data, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        return False, response.text
    except Exception as e:
        return False, str(e)

# Main app
st.title("🤖 Project Knowledge Transfer Chatbot")
st.markdown("Upload documents and ask questions about them!")

# Backend status check
is_connected, status_info = test_backend_connection()

if is_connected:
    st.success(f"✅ Backend connected successfully!")
    if isinstance(status_info, dict):
        st.info(f"Documents in database: {status_info.get('services', {}).get('document_count', 0)}")
else:
    st.error(f"❌ Backend connection failed: {status_info}")
    st.stop()

# Create two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 Document Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'md', 'pdf'],
        help="Supported formats: .txt, .md, .pdf"
    )
    
    if uploaded_file is not None:
        if st.button("Upload Document", type="primary"):
            with st.spinner("Uploading..."):
                success, result = upload_document(uploaded_file)
                if success:
                    st.success("✅ Document uploaded successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result}")
    
    # Display current documents
    st.subheader("📚 Current Documents")
    documents = get_documents()
    
    if documents:
        for doc in documents:
            st.write(f"**{doc['filename']}**")
            st.caption(f"Chunks: {doc['chunk_count']} | Type: {doc['file_type']}")
            st.divider()
    else:
        st.info("No documents uploaded yet")

with col2:
    st.subheader("💬 Chat")
    
    if not documents:
        st.warning("⚠️ Upload documents first to start chatting")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"- **{source['filename']}** (Page {source.get('page', 'N/A')})")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    success, response = ask_question(prompt)
                    
                    if success:
                        answer = response.get("answer", "No answer provided")
                        sources = response.get("sources", [])
                        
                        st.markdown(answer)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.write(f"- **{source['filename']}** (Page {source.get('page', 'N/A')})")
                    else:
                        error_msg = f"❌ Error: {response}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.session_state.get("messages"):
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()