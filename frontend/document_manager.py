import streamlit as st
import requests
from typing import List, Optional
import logging
from datetime import datetime

from config import BACKEND_URL

logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self):
        self.backend_url = BACKEND_URL
        
    def render_upload_section(self):
        """Render the document upload section."""
        st.subheader("📄 Document Upload")
        
        # File upload widget
        uploaded_files = st.file_uploader(
            "Upload project documents",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            help="Supported formats: .txt, .md, .pdf (max 10MB each)"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            upload_button = st.button("📤 Upload Documents", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear All Documents", type="secondary")
        
        # Handle upload
        if upload_button and uploaded_files:
            with st.spinner("Processing documents..."):
                success = self._upload_documents(uploaded_files)
                if success:
                    st.rerun()
        
        # Handle clear
        if clear_button:
            with st.spinner("Clearing documents..."):
                success = self._clear_documents()
                if success:
                    st.success("All documents cleared successfully!")
                    st.rerun()
        
        # Display current documents
        self._display_current_documents()
    
    def _upload_documents(self, uploaded_files) -> bool:
        """Upload documents to backend."""
        try:
            files = []
            for uploaded_file in uploaded_files:
                files.append(
                    ('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                )
            
            response = requests.post(
                f"{self.backend_url}/upload-documents",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                documents = response.json()
                st.success(f"✅ Successfully uploaded {len(documents)} documents!")
                
                # Show details
                with st.expander("📋 Upload Details"):
                    for doc in documents:
                        st.write(f"**{doc['filename']}**")
                        st.write(f"- Chunks: {doc['chunk_count']}")
                        st.write(f"- Type: {doc['file_type']}")
                        st.write("---")
                
                return True
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Upload failed: {error_detail}")
                return False
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend server. Please ensure the backend is running on port 8000.")
            return False
        except requests.exceptions.Timeout:
            st.error("❌ Upload timed out. Please try with smaller files or check your connection.")
            return False
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            st.error(f"❌ Upload error: {str(e)}")
            return False
    
    def _clear_documents(self) -> bool:
        """Clear all documents from backend."""
        try:
            response = requests.delete(f"{self.backend_url}/documents", timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Clear failed: {error_detail}")
                return False
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend server.")
            return False
        except Exception as e:
            logger.error(f"Clear error: {str(e)}")
            st.error(f"❌ Clear error: {str(e)}")
            return False
    
    def _display_current_documents(self):
        """Display currently processed documents."""
        try:
            response = requests.get(f"{self.backend_url}/documents", timeout=5)
            
            if response.status_code == 200:
                documents = response.json()
                
                if documents:
                    st.subheader(f"📚 Current Documents ({len(documents)})")
                    
                    for doc in documents:
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"**{doc['filename']}**")
                                st.caption(f"Uploaded: {doc['upload_time'][:19]}")
                            
                            with col2:
                                st.metric("Chunks", doc['chunk_count'])
                            
                            with col3:
                                st.write(f"**{doc['file_type']}**")
                            
                            st.divider()
                else:
                    st.info("📭 No documents uploaded yet. Upload some documents to get started!")
        
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ Cannot connect to backend to show current documents.")
        except Exception as e:
            logger.error(f"Display documents error: {str(e)}")
            st.warning(f"⚠️ Error loading documents: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the number of currently uploaded documents."""
        try:
            response = requests.get(f"{self.backend_url}/documents", timeout=5)
            if response.status_code == 200:
                documents = response.json()
                return len(documents)
        except:
            pass
        return 0
