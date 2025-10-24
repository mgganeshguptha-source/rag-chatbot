# RAG Chatbot for Google Drive

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on documents stored in Google Drive. Built with Streamlit and powered by Google Gemini AI.

## Features

- 🤖 **Intelligent Q&A**: Ask questions about your Google Drive documents and get AI-powered answers
- 📁 **Google Drive Integration**: Securely connects to your specified Google Drive folder
- 🔍 **Smart Document Search**: Uses keyword-based retrieval to find relevant information
- 💬 **Context-Aware Responses**: Generates answers based on document content using Gemini AI
- 🎯 **Source Attribution**: Shows which documents were used to generate each answer (📄 for Drive, 🔗 for web links, 🌐 for general knowledge)
- 🔗 **Web Content Integration**: Automatically detects URLs in your questions and fetches content from those web pages to enhance answers
- 🌐 **Extended Knowledge**: Automatically falls back to Gemini's general knowledge when Drive documents don't contain the answer
- ⚡ **Auto-Initialize**: Automatically loads documents on startup
- 🔄 **Retry Logic**: Automatic retry mechanism for handling temporary API failures

## Prerequisites

1. **Google Cloud Project** with Google Drive API enabled
2. **Service Account** with JSON credentials
3. **Gemini API Key** from Google AI Studio
4. **Google Drive Folder** with read permissions for the service account

## Setup Instructions

### 1. Google Cloud Setup

#### Enable Google Drive API
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services > Library**
4. Search for "Google Drive API"
5. Click **Enable**

#### Create Service Account
1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > Service Account**
3. Fill in the service account details and click **Create**
4. Grant the service account "Viewer" role (or appropriate permissions)
5. Click **Done**

#### Download Service Account Key
1. Click on the created service account
2. Go to the **Keys** tab
3. Click **Add Key > Create New Key**
4. Select **JSON** format
5. Click **Create** (the JSON file will download automatically)
6. **Copy the entire contents** of this JSON file - you'll need it for the environment variables

### 2. Google Drive Setup

1. Create or identify a folder in Google Drive containing your documents
2. Share this folder with your service account email (found in the JSON credentials file, looks like `your-service-account@your-project.iam.gserviceaccount.com`)
3. Give the service account **Viewer** permissions
4. Copy the **Folder ID** from the URL (e.g., if URL is `https://drive.google.com/drive/folders/1AbC123XyZ`, the ID is `1AbC123XyZ`)

### 3. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Copy the generated API key

### 4. Environment Variables

Set the following environment variables in your Replit Secrets or `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
GOOGLE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"..."}
USE_EXTENDED_KNOWLEDGE=true  # Optional: Enable/disable extended knowledge (default: true)
```

**Note**: For `GOOGLE_SERVICE_ACCOUNT_KEY`, paste the **entire JSON contents** from the service account key file you downloaded.

**Extended Knowledge**: When enabled (default), the chatbot automatically falls back to Gemini's general knowledge if your Drive documents don't contain relevant information. Set to `false` to restrict answers to Drive documents only.

### 5. Run the Application

```bash
streamlit run app.py --server.port 5000
```

The application will:
1. Automatically load documents from your Google Drive folder
2. Process and chunk the documents
3. Initialize the chatbot interface
4. Be ready to answer questions!

## Supported File Types

✅ **Currently supports:**
- `.txt` files (plain text)
- `.pdf` files (text-based PDFs)
- `.jpg/.jpeg` files (images with text via OCR)

⚠️ **Note**: Image-based/scanned PDFs are not yet supported. Only text-based PDFs work currently.

## How It Works

1. **Document Loading**: Connects to Google Drive and retrieves all supported files (TXT, PDF, JPG) from the specified folder
2. **Text Extraction**: 
   - TXT files: Direct UTF-8/Latin-1 decoding
   - PDF files: Text extraction using pypdf library
   - JPG files: OCR text extraction using pytesseract
3. **Text Chunking**: Splits documents into manageable chunks with overlap for better context
4. **Query Processing**: When you ask a question:
   - Searches Drive documents for relevant chunks using keyword matching
   - Automatically detects URLs in your question (up to 2 URLs)
   - Fetches and extracts content from detected web pages
5. **Smart Routing**: 
   - If Drive documents contain relevant information (relevance score > 0.4), uses Drive context
   - If URLs detected, fetches and combines web content with Drive context
   - If both Drive and web lack relevant information, falls back to Gemini's general knowledge
6. **Response Generation**: Sends combined context (Drive + Web) to Gemini AI (with automatic retry on temporary failures) to generate an accurate answer
7. **Source Attribution**: Shows which sources were used:
   - 📄 for Drive sources
   - 🔗 for web links
   - 🌐 for general knowledge
   - Responses can combine multiple source types

## Architecture

```
User Question
    ↓
Keyword-based Retrieval (finds relevant document chunks)
    ↓
Relevance Scoring (filters chunks with score > 0.4)
    ↓
    ├─── Drive has relevant info? ──YES──> Use Drive context
    │                                      ↓
    │                                   Gemini AI (Drive-based answer)
    │                                      ↓
    │                                   Response with 📄 Drive attribution
    │
    └─── No relevant Drive info? ──YES──> Use general knowledge
                                           ↓
                                        Gemini AI (general knowledge)
                                           ↓
                                        Response with 🌐 note
```

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration management
├── drive_service.py          # Google Drive API integration
├── rag_pipeline.py           # RAG implementation
├── session_manager.py        # Session management (legacy)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # This file
```

## Troubleshooting

### Google Drive API Error
**Error**: `Google Drive API has not been used in project...`

**Solution**: 
1. Go to https://console.developers.google.com
2. Enable the Google Drive API for your project
3. Wait a few minutes for the changes to propagate

### No Documents Found
**Error**: `No documents found in the specified Google Drive folder`

**Solution**:
1. Verify the folder ID is correct
2. Ensure the folder contains supported files (TXT, PDF, or JPG)
3. Check that the service account has access to the folder
4. Make sure the folder is shared with the service account email

### Authentication Error
**Error**: `Invalid JSON format in GOOGLE_SERVICE_ACCOUNT_KEY`

**Solution**:
1. Ensure you copied the **entire** JSON contents from the service account key file
2. The JSON should start with `{` and end with `}`
3. Do not modify the JSON structure

### Gemini API Overload Error
**Error**: `503 UNAVAILABLE - The model is overloaded`

**Solution**:
- The application automatically retries up to 3 times with exponential backoff (1s, 2s, 4s delays)
- If the error persists, wait a few moments and try again
- The `gemini-2.0-flash-exp` model is used with built-in retry logic

## Development Roadmap

### Phase 1 (MVP) ✅
- [x] Streamlit chat interface
- [x] Google Drive integration
- [x] RAG pipeline with Gemini AI
- [x] Text file support (.txt)
- [x] Source attribution

### Phase 2 ✅
- [x] Extended knowledge retrieval (use Gemini's knowledge when Drive lacks info) ✅

### Phase 3 ✅
- [x] PDF file support ✅
- [x] Image OCR support (.jpg) ✅
- [x] Web content integration ✅

### Phase 4 (Planned)
- [ ] Document preview in UI
- [ ] Deployment to Hugging Face Spaces
- [ ] OpenAI validation agent
- [ ] "Use only Google Drive" command support
- [ ] Scanned PDF OCR support
- [ ] PNG image support

## Technologies Used

- **Frontend**: Streamlit
- **LLM**: Google Gemini AI (gemini-2.0-flash-exp with retry logic and extended knowledge)
- **Cloud Storage**: Google Drive API
- **Authentication**: Google Service Account
- **Document Processing**:
  - pypdf: PDF text extraction
  - pytesseract: OCR for image text extraction
  - Pillow (PIL): Image processing
- **Web Content Retrieval**:
  - requests: HTTP client for fetching web pages
  - BeautifulSoup4: HTML parsing and content extraction
- **Language**: Python 3.11+

## Contributing

This project is prepared for GitHub integration. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- Check the Troubleshooting section above
- Review Google Cloud and Drive API documentation
- Ensure all environment variables are correctly set

---

**Built with ❤️ on Replit**
