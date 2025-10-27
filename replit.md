# RAG Chatbot for Google Drive

## Overview

This is a Retrieval-Augmented Generation (RAG) chatbot application that enables users to ask questions about documents stored in Google Drive. The system retrieves relevant document chunks and uses Google's Gemini AI to generate contextual answers. Built with Streamlit for the web interface, it provides an intuitive chat experience with source attribution, automatic document loading, and extended knowledge fallback for questions beyond Drive documents.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **UI Pattern**: Chat-based interface with message history
- **Session Management**: Streamlit session state for maintaining conversation context and initialization status
- **Caching Strategy**: Resource-level caching (`@st.cache_resource`) for expensive operations like service initialization and document loading

### Backend Architecture
- **RAG Pipeline**: Keyword-based document retrieval with AI-powered response generation and web content integration
- **Text Processing**: Document chunking with configurable overlap (1000 char chunks, 100 char overlap) for optimal context retrieval
- **Search Method**: Keyword matching with similarity scoring to identify relevant document chunks
- **Web Content Integration**: 
  - URL Detection: Regex-based detection of URLs in user queries (max 2 URLs per query) AND automatic detection of URLs in Drive document chunks
  - URL Discovery Threshold: Scans all relevant Drive chunks with score > 0.2 to find URLs, even if chunks aren't highly relevant by keywords
  - Content Fetching: HTTP requests with timeout (10s) and size limits (1MB)
  - HTML Extraction: BeautifulSoup-based content parsing and cleaning
  - Security: Domain blocking list and content-type validation
  - Smart Integration: Users don't need to provide URLs - chatbot automatically finds and fetches URLs mentioned in Drive documents when relevant to the query
- **Smart Routing**: Multi-source response generation:
  - Drive documents with high relevance (score > 0.4) → Drive-based answers with 📄 attribution
  - URLs detected in query → Fetch and combine web content with Drive context, show 🔗 attribution
  - No relevant Drive/web content → Automatic fallback to Gemini's general knowledge with 🌐 note
- **Extended Knowledge**: Configurable fallback to general AI knowledge when Drive and web lack relevant information
- **Response Generation**: Context-aware answer synthesis combining Drive chunks, web content, and/or general knowledge

### Configuration Management
- **Environment-based Config**: Centralized `Config` class managing all environment variables
- **Validation**: Automatic validation of required configuration on startup
- **Configuration Requirements**:
  - Gemini API key for AI responses
  - Google Service Account credentials (JSON format)
  - Google Drive folder ID for document source

### Authentication & Authorization
- **Google Drive Access**: Service account-based authentication using OAuth2
- **Credentials Storage**: Service account JSON credentials stored as environment variable
- **Permissions Model**: Read-only access scope (`drive.readonly`) for security
- **Drive Folder Sharing**: Requires service account email to have access to target folder

### Session Management
- **Timeout Mechanism**: Configurable session timeout (default 5 minutes)
- **Automatic Cleanup**: Background thread for expired session removal
- **Activity Tracking**: Last activity timestamp and expiration time per session
- **Thread Safety**: Lock-based synchronization for concurrent session access

## External Dependencies

### APIs & Services
- **Google Drive API v3**: Document retrieval from specified Drive folders
- **Google Gemini AI API**: Natural language understanding and response generation
  - Model: `gemini-2.0-flash-exp` (experimental model with extended capabilities)
  - Purpose: Converting retrieved context into coherent answers and providing general knowledge fallback
  - Retry Logic: 3 attempts with exponential backoff (1s, 2s, 4s) for handling rate limits and temporary unavailability

### Authentication Services
- **Google OAuth2 Service Account**: Server-to-server authentication for Drive access
- **Scopes**: `https://www.googleapis.com/auth/drive.readonly`

### Python Libraries
- **Streamlit**: Web application framework for the chat interface
- **google-api-python-client**: Google Drive API integration
- **google-auth**: OAuth2 authentication handling
- **google-genai**: Gemini AI client library
- **pypdf**: PDF text extraction
- **pytesseract**: OCR engine wrapper for image text extraction
- **Pillow (PIL)**: Python Imaging Library for image processing
- **requests**: HTTP client for web content fetching
- **beautifulsoup4**: HTML parsing and content extraction

### Data Storage
- **In-Memory Storage**: Document chunks and session data stored in application memory
- **Session State**: Streamlit session state for user conversation history
- **No Persistent Database**: Stateless architecture requiring reinitialization on restart

### Document Processing
- **Text Extraction**: Multi-format document processing
  - TXT: Direct UTF-8/Latin-1 decoding
  - PDF: pypdf library for text-based PDFs
  - JPG/JPEG: pytesseract OCR for image text extraction with defensive error handling
- **Supported File Types**: .txt, .pdf (text-based), .jpg/.jpeg
- **Unsupported**: Image-based/scanned PDFs (no OCR for PDFs yet)
- **OCR Safety Features**: Production-ready defensive programming to handle problematic images gracefully
  - File size validation (max 20MB) - skips extremely large images
  - Image verification before OCR processing using PIL Image.verify()
  - Dimension checks (max 10000x10000 pixels) - prevents memory issues
  - Format conversion for problematic image modes (converts to RGB/L)
  - Complete resource cleanup on all code paths (closes file handles properly)
  - Graceful skip-and-continue behavior - corrupted/problematic images are logged and skipped without crashing the entire document loading process
  - Separate error handling for validation vs OCR operations for granular failure reporting

### Required Environment Variables
1. `GEMINI_API_KEY`: API key for Google Gemini AI service
2. `GOOGLE_SERVICE_ACCOUNT_KEY`: Full JSON credentials for service account (stored as string)
3. `GOOGLE_DRIVE_FOLDER_ID`: Target folder ID containing documents to query
4. `USE_EXTENDED_KNOWLEDGE`: Optional (default: "true") - Enable/disable extended knowledge fallback