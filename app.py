import streamlit as st
import os
import sys
from typing import List, Dict, Optional

print("=" * 50, file=sys.stderr)
print("STARTUP DEBUG: app.py loading...", file=sys.stderr)
print("=" * 50, file=sys.stderr)

try:
    print("DEBUG: Importing config...", file=sys.stderr)
    from config import Config
    print("DEBUG: Config imported successfully", file=sys.stderr)
    
    print("DEBUG: Importing drive_service...", file=sys.stderr)
    from drive_service import GoogleDriveService
    print("DEBUG: drive_service imported successfully", file=sys.stderr)
    
    print("DEBUG: Importing rag_pipeline...", file=sys.stderr)
    from rag_pipeline import RAGPipeline
    print("DEBUG: rag_pipeline imported successfully", file=sys.stderr)
    
except Exception as e:
    print(f"DEBUG: Import error: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# Page configuration
print("DEBUG: Setting page config...", file=sys.stderr)
st.set_page_config(
    page_title="RAG Chatbot for Google Drive",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)
print("DEBUG: Page config set", file=sys.stderr)

# Initialize services with error handling
@st.cache_resource
def get_config():
    print("DEBUG: get_config() called", file=sys.stderr)
    try:
        config = Config()
        print(f"DEBUG: Config created, is_configured={config.is_configured}", file=sys.stderr)
        return config
    except Exception as e:
        print(f"DEBUG: Error in get_config: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

@st.cache_resource
def get_drive_service(folder_id):
    print(f"DEBUG: get_drive_service() called with folder_id={folder_id}", file=sys.stderr)
    try:
        service = GoogleDriveService(folder_id)
        print("DEBUG: GoogleDriveService created successfully", file=sys.stderr)
        return service
    except Exception as e:
        print(f"DEBUG: Error in get_drive_service: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

@st.cache_resource
def get_rag_pipeline(api_key, use_extended_knowledge=True):
    print(f"DEBUG: get_rag_pipeline() called", file=sys.stderr)
    try:
        pipeline = RAGPipeline(api_key, use_extended_knowledge=use_extended_knowledge)
        print("DEBUG: RAGPipeline created successfully", file=sys.stderr)
        return pipeline
    except Exception as e:
        print(f"DEBUG: Error in get_rag_pipeline: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

print("DEBUG: Creating config...", file=sys.stderr)
config = get_config()
print(f"DEBUG: Config created, proceeding to drive service...", file=sys.stderr)

print("DEBUG: Creating drive service...", file=sys.stderr)
drive_service = get_drive_service(config.google_drive_folder_id)
print("DEBUG: Drive service created, proceeding to RAG pipeline...", file=sys.stderr)

print("DEBUG: Creating RAG pipeline...", file=sys.stderr)
rag_pipeline = get_rag_pipeline(config.gemini_api_key, config.use_extended_knowledge)
print("DEBUG: RAG pipeline created successfully", file=sys.stderr)

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

print("DEBUG: app.py loaded successfully", file=sys.stderr)
