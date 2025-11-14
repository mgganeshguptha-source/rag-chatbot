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
    
    print("DEBUG: Importing web_content_service...", file=sys.stderr)
    from web_content_service import WebContentService
    print("DEBUG: web_content_service imported successfully", file=sys.stderr)
    
except Exception as e:
    print(f"DEBUG: Import error: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# Page configuration
print("DEBUG: Setting page config...", file=sys.stderr)
st.set_page_config(
    page_title="Ganesh's RAG Chatbot for Google Drive",
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
st.title("🤖 Ganesh's RAG Chatbot for Google Drive")
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
                # Detect URLs in the query
                urls = WebContentService.detect_urls(prompt)
                web_content = None
                
                # Fetch web content if URLs found
                if urls:
                    with st.spinner(f"Fetching content from {len(urls)} web link(s)..."):
                        web_content = WebContentService.fetch_all_urls(urls)
                        if web_content:
                            st.info(f"🔗 Retrieved content from {len(web_content)} web source(s)")
                
                # Generate response with web content
                response = rag_pipeline.generate_response(prompt, external_web_content=web_content)
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
import streamlit as st
import os
import sys
import socket
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
    
    print("DEBUG: Importing rag_pipeline_postgres...", file=sys.stderr)
    from rag_pipeline_postgres import RAGPipelinePostgres
    print("DEBUG: rag_pipeline_postgres imported successfully", file=sys.stderr)
    
    print("DEBUG: Importing web_content_service...", file=sys.stderr)
    from web_content_service import WebContentService
    print("DEBUG: web_content_service imported successfully", file=sys.stderr)
    
    print("DEBUG: Importing auth_service...", file=sys.stderr)
    from auth_service import AuthService
    print("DEBUG: auth_service imported successfully", file=sys.stderr)
    
except Exception as e:
    print(f"DEBUG: Import error: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# Page configuration
print("DEBUG: Setting page config...", file=sys.stderr)
st.set_page_config(
    page_title="Ganesh's RAG Chatbot for Google Drive",
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
    print(f"DEBUG: get_rag_pipeline() called (using PostgreSQL + pgvector)", file=sys.stderr)
    try:
        pipeline = RAGPipelinePostgres(api_key, use_extended_knowledge=use_extended_knowledge)
        print("DEBUG: RAGPipelinePostgres created successfully (query-only mode)", file=sys.stderr)
        return pipeline
    except Exception as e:
        print(f"DEBUG: Error in get_rag_pipeline: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

def check_database_status(rag_pipeline):
    """Check if database has embeddings (PostgreSQL mode - no document loading)
    
    Note: This is NOT cached so it always reflects current database state.
    Database stat queries are fast (~1ms), so no caching needed.
    """
    print(f"DEBUG: check_database_status() called", file=sys.stderr)
    try:
        # Get database statistics (fast COUNT query)
        stats = rag_pipeline.embedding_store.get_stats()
        total_chunks = stats.get('total_chunks', 0)
        total_docs = stats.get('total_documents', 0)
        
        print(f"📊 Database stats: {total_docs} documents, {total_chunks} chunks", file=sys.stderr)
        
        if total_chunks == 0:
            print("⚠️ Database is empty. Run the embedding pipeline to populate it.", file=sys.stderr)
            return {
                'status': 'empty',
                'total_docs': 0,
                'total_chunks': 0
            }
        
        print(f"✅ Database ready with {total_docs} documents and {total_chunks} chunks", file=sys.stderr)
        return {
            'status': 'ready',
            'total_docs': total_docs,
            'total_chunks': total_chunks
        }
        
    except Exception as e:
        error_msg = f"Error checking database: {str(e)}"
        print(f"❌ {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'error': error_msg
        }

# Services will be initialized lazily in the main app flow
print("DEBUG: Module initialization complete", file=sys.stderr)

def initialize_session(rag_pipeline):
    """Initialize session by checking database status (PostgreSQL mode)"""
    with st.spinner("Checking database..."):
        db_status = check_database_status(rag_pipeline)
    
    if db_status['status'] == 'error':
        st.error(f"❌ {db_status['error']}")
        st.info("💡 **Database Connection Error** - Check your PostgreSQL connection settings.")
        return False
    elif db_status['status'] == 'empty':
        st.warning("⚠️ Database is empty - no embeddings found.")
        st.info("""
        💡 **To populate the database with embeddings:**
        
        **Option 1: GitHub Actions (Recommended)**
        1. Push this repo to GitHub
        2. The workflow will run automatically on schedule
        3. Or manually trigger it from Actions tab
        
        **Option 2: Manual Pipeline**
        ```bash
        python embed_pipeline.py
        ```
        
        **Note:** The chatbot will still work using general AI knowledge, but won't have access to your Drive documents until embeddings are generated.
        """)
        
        # Add refresh button to check if pipeline has populated the database
        if st.button("🔄 Refresh Database Status"):
            # Clear all session state to force fresh check
            st.session_state.initialized = False
            if 'database_empty' in st.session_state:
                del st.session_state.database_empty
            st.rerun()
        
        st.session_state.initialized = True
        st.session_state.database_empty = True
        return True
    else:
        st.session_state.initialized = True
        st.session_state.database_empty = False  # Explicitly clear empty flag
        st.success(f"✅ Database ready with {db_status['total_docs']} documents ({db_status['total_chunks']} chunks)")
        return True

def main():
    """Main application flow"""
    print("DEBUG: main() function called", file=sys.stderr)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    # Get configuration
    print("DEBUG: Getting config...", file=sys.stderr)
    config = get_config()
    print(f"DEBUG: Config loaded, is_configured={config.is_configured}", file=sys.stderr)
    
    # Initialize authentication service
    auth_service = AuthService()
    
    # Authentication check (only if OAuth providers are configured)
    if auth_service.is_any_provider_configured():
        if not auth_service.is_authenticated():
            # User not logged in - show login page
            user_info = auth_service.show_login_page()
            if user_info:
                # Login successful
                st.session_state.user_info = user_info
                st.success(f"✅ Welcome, {user_info.get('name')}!")
                st.rerun()
            st.stop()
        
        # User is authenticated - show user profile in sidebar
        auth_service.show_user_profile()
    else:
        # No OAuth configured - show info message in sidebar
        with st.sidebar:
            st.info("💡 **Optional**: Add Google/GitHub authentication by configuring OAuth credentials. See [OAUTH_SETUP.md](https://github.com/your-repo/OAUTH_SETUP.md) for instructions.")
    
    # Main UI
    st.title("🤖 Ganesh's RAG Chatbot for Google Drive")
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
    
    # Initialize RAG pipeline (lazy, cached) - Drive service not needed in PostgreSQL mode
    print("DEBUG: Initializing RAG pipeline...", file=sys.stderr)
    rag_pipeline = get_rag_pipeline(config.gemini_api_key, config.use_extended_knowledge)
    print("DEBUG: RAG pipeline initialized successfully", file=sys.stderr)
    
    # Auto-initialize on first run (just checks database status)
    if not st.session_state.initialized:
        with st.spinner("Initializing chatbot..."):
            if not initialize_session(rag_pipeline):
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
                    # Detect URLs in the query
                    urls = WebContentService.detect_urls(prompt)
                    web_content = None
                    
                    # Fetch web content if URLs found
                    if urls:
                        with st.spinner(f"Fetching content from {len(urls)} web link(s)..."):
                            web_content = WebContentService.fetch_all_urls(urls)
                            if web_content:
                                st.info(f"🔗 Retrieved content from {len(web_content)} web source(s)")
                    
                    # Generate response with web content
                    response = rag_pipeline.generate_response(prompt, external_web_content=web_content)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.divider()
    st.caption("🤖 RAG Chatbot powered by Google Gemini AI | Built on Replit")

# Run main application
print("DEBUG: Calling main()...", file=sys.stderr)
main()
print("DEBUG: app.py loaded successfully", file=sys.stderr)
