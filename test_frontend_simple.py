#!/usr/bin/env python3
"""
Simple test to check if frontend components can initialize
"""
import streamlit as st
import sys
import traceback

st.set_page_config(page_title="Frontend Test", layout="wide")

st.title("Frontend Components Test")

try:
    st.write("Testing basic imports...")
    from config import BACKEND_URL, GEMINI_API_KEY
    st.success(f"✅ Config imported successfully - Backend URL: {BACKEND_URL}")
    
    st.write("Testing frontend components...")
    from frontend.document_manager import DocumentManager
    from frontend.chat_interface import ChatInterface
    st.success("✅ Frontend imports successful")
    
    st.write("Testing component initialization...")
    doc_manager = DocumentManager()
    chat_interface = ChatInterface()
    st.success("✅ Components initialized successfully")
    
    st.write("Testing backend connectivity...")
    import requests
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success(f"✅ Backend is reachable: {response.json()}")
        else:
            st.error(f"❌ Backend returned status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Backend connection failed: {e}")
    
    st.markdown("---")
    st.write("**If all tests pass, the main app should work correctly.**")
    
except Exception as e:
    st.error(f"❌ Error occurred: {e}")
    st.code(traceback.format_exc())
    st.markdown("---")
    st.write("**This error is preventing the main app from loading.**")