import streamlit as st
import requests
from typing import List, Dict, Any
import uuid
import logging
from datetime import datetime

from config import BACKEND_URL

logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self):
        self.backend_url = BACKEND_URL
        
        # Initialize session state
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'waiting_for_response' not in st.session_state:
            st.session_state.waiting_for_response = False
    
    def render_chat_interface(self, document_count: int):
        """Render the main chat interface."""
        st.subheader("💬 Chat with Your Documents")
        
        # Show status
        if document_count == 0:
            st.warning("⚠️ No documents uploaded. Please upload documents first to start chatting.")
            return
        
        st.success(f"✅ Ready to chat! {document_count} document(s) loaded.")
        
        # Chat controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary"):
                self._clear_chat_history()
                st.rerun()
        
        # Chat history display
        self._display_chat_history()
        
        # Chat input
        self._render_chat_input()
    
    def _display_chat_history(self):
        """Display the chat history."""
        if not st.session_state.chat_history:
            st.info("👋 Ask me anything about your uploaded documents!")
            return
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                        if 'timestamp' in message:
                            st.caption(f"⏰ {message['timestamp']}")
                
                elif message['role'] == 'assistant':
                    with st.chat_message("assistant"):
                        st.write(message['content'])
                        
                        # Show sources if available
                        if message.get('sources'):
                            with st.expander("📚 Sources"):
                                for source in message['sources']:
                                    st.write(f"- {source}")
                        
                        if 'timestamp' in message:
                            st.caption(f"⏰ {message['timestamp']}")
    
    def _render_chat_input(self):
        """Render the chat input area."""
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask a question about your documents:",
                placeholder="e.g., What is the project architecture? What are the main requirements?",
                height=100,
                disabled=st.session_state.waiting_for_response
            )
            
            col1, col2 = st.columns([6, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "Send 📤",
                    type="primary",
                    disabled=st.session_state.waiting_for_response
                )
        
        # Handle form submission
        if submit_button and user_input.strip():
            self._process_user_message(user_input.strip())
    
    def _process_user_message(self, message: str):
        """Process user message and get AI response."""
        # Add user message to history
        user_msg = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.chat_history.append(user_msg)
        
        # Set waiting state
        st.session_state.waiting_for_response = True
        
        # Show loading message
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Send request to backend
                    response = self._send_chat_request(message)
                    
                    if response:
                        # Add assistant response to history
                        assistant_msg = {
                            'role': 'assistant',
                            'content': response['response'],
                            'sources': response.get('sources', []),
                            'confidence': response.get('confidence', 0.0),
                            'has_context': response.get('has_context', False),
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        }
                        st.session_state.chat_history.append(assistant_msg)
                    else:
                        # Add error message
                        error_msg = {
                            'role': 'assistant',
                            'content': "❌ Sorry, I encountered an error processing your request. Please try again.",
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        }
                        st.session_state.chat_history.append(error_msg)
                
                except Exception as e:
                    logger.error(f"Chat processing error: {str(e)}")
                    error_msg = {
                        'role': 'assistant',
                        'content': f"❌ Error: {str(e)}",
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.chat_history.append(error_msg)
        
        # Reset waiting state
        st.session_state.waiting_for_response = False
        
        # Rerun to update the interface
        st.rerun()
    
    def _send_chat_request(self, message: str) -> Dict[str, Any]:
        """Send chat request to backend."""
        try:
            payload = {
                "message": message,
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(
                f"{self.backend_url}/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Chat request failed: {error_detail}")
                return {}
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend server. Please ensure the backend is running on port 8000.")
            return {}
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
            return {}
        except Exception as e:
            logger.error(f"Chat request error: {str(e)}")
            return {}
    
    def _clear_chat_history(self):
        """Clear the chat history."""
        try:
            # Clear backend history
            response = requests.delete(
                f"{self.backend_url}/chat-history/{st.session_state.session_id}",
                timeout=5
            )
            
            # Clear frontend history regardless of backend response
            st.session_state.chat_history = []
            
            if response.status_code == 200:
                st.success("✅ Chat history cleared!")
            else:
                st.warning("⚠️ Chat history cleared locally, but backend clear may have failed.")
        
        except requests.exceptions.ConnectionError:
            # Clear locally even if backend is unreachable
            st.session_state.chat_history = []
            st.warning("⚠️ Chat history cleared locally (backend unreachable).")
        except Exception as e:
            logger.error(f"Clear chat history error: {str(e)}")
            st.session_state.chat_history = []
            st.warning(f"⚠️ Chat history cleared locally. Error: {str(e)}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            'session_id': st.session_state.session_id,
            'message_count': len(st.session_state.chat_history),
            'last_activity': datetime.now().isoformat() if st.session_state.chat_history else None
        }
