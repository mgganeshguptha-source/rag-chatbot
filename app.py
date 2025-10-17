import streamlit as st
import os
from typing import List, Dict, Optional

from drive_service import GoogleDriveService
from rag_pipeline import RAGPipeline
from config import Config

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot for Google Drive",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize services
@st.cache_resource
def get_config():
    return Config()

@st.cache_resource
def get_drive_service(folder_id):
    return GoogleDriveService(folder_id)

@st.cache_resource
def get_rag_pipeline(api_key):
    return RAGPipeline(api_key)

config = get_config()
drive_service = get_drive_service(config.google_drive_folder_id)
rag_pipeline = get_rag_pipeline(config.gemini_api_key)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_session():
    """Initialize and load documents from Google Drive"""
    try:
        # Load documents from Google Drive
        with st.spinner("Loading documents from Google Drive..."):
            documents = drive_service.load_documents()
        
        if not documents:
            st.error("⚠️ No documents found in the specified Google Drive folder. Please check the folder ID and permissions.")
            return False
        
        # Initialize RAG pipeline with documents
        with st.spinner(f"Processing {len(documents)} documents..."):
            rag_pipeline.initialize_with_documents(documents)
        
        st.session_state.initialized = True
        st.success(f"✅ Loaded {len(documents)} documents from Google Drive.")
        return True
        
    except Exception as e:
        st.error(f"❌ Error initializing: {str(e)}")
        st.info("💡 **Google Drive API Setup Required:**\n\n1. Go to [Google Cloud Console](https://console.developers.google.com)\n2. Enable the Google Drive API for your project\n3. Make sure your service account has access to the folder")
        return False

# Main UI
st.title("🤖 RAG Chatbot for Google Drive")
st.markdown("""
Ask questions about your documents stored in Google Drive. The chatbot will search through your documents and provide relevant answers.
""")

# Check for configuration
if not config.is_configured:
    st.error("❌ Missing required configuration. Please set the following environment variables:")
    if not config.gemini_api_key:
        st.code("GEMINI_API_KEY")
    if not config.google_service_account_key:
        st.code("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not config.google_drive_folder_id:
        st.code("GOOGLE_DRIVE_FOLDER_ID")
    st.stop()

# Auto-initialize on first run
if not st.session_state.initialized:
    with st.spinner("Initializing chatbot..."):
        if not initialize_session():
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_pipeline.generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("🤖 RAG Chatbot powered by Google Gemini AI | Built on Replit")
