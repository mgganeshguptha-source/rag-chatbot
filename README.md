# RAG Chatbot for Google Drive

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on documents stored in Google Drive. Built with Streamlit and powered by Google Gemini AI.

## Features

- 🤖 **Intelligent Q&A**: Ask questions about your Google Drive documents and get AI-powered answers
- 📁 **Google Drive Integration**: Securely connects to your specified Google Drive folder
- 🔍 **Smart Document Search**: Uses keyword-based retrieval to find relevant information
- 💬 **Context-Aware Responses**: Generates answers based on document content using Gemini AI
- 🎯 **Source Attribution**: Shows which documents were used to generate each answer
- ⚡ **Auto-Initialize**: Automatically loads documents on startup

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
```

**Note**: For `GOOGLE_SERVICE_ACCOUNT_KEY`, paste the **entire JSON contents** from the service account key file you downloaded.

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

Currently supports:
- `.txt` files (plain text)

Future support planned for:
- `.pdf` files (Phase 3)
- `.jpg` images via OCR (Phase 3)

## How It Works

1. **Document Loading**: Connects to Google Drive and retrieves all `.txt` files from the specified folder
2. **Text Chunking**: Splits documents into manageable chunks with overlap for better context
3. **Query Processing**: When you ask a question, the system searches for relevant chunks using keyword matching
4. **Response Generation**: Sends the relevant context to Gemini AI (with automatic retry on temporary failures) to generate an accurate answer
5. **Source Attribution**: Shows which documents were used to answer your question

## Architecture

```
User Question
    ↓
Keyword-based Retrieval (finds relevant document chunks)
    ↓
Context Preparation (top 3 most relevant chunks)
    ↓
Gemini AI (generates answer based on context)
    ↓
Response with Source Attribution
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
2. Ensure the folder contains `.txt` files
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
- The application automatically retries up to 3 times with exponential backoff
- If the error persists, wait a few moments and try again
- The stable `gemini-1.5-flash` model is used for better reliability

## Development Roadmap

### Phase 1 (MVP) ✅
- [x] Streamlit chat interface
- [x] Google Drive integration
- [x] RAG pipeline with Gemini AI
- [x] Text file support (.txt)
- [x] Source attribution

### Phase 2 (Planned)
- [ ] Extended knowledge retrieval (use Gemini's knowledge when Drive lacks info)
- [ ] OpenAI validation agent
- [ ] "Use only Google Drive" command support

### Phase 3 (Planned)
- [ ] PDF file support
- [ ] Image OCR support (.jpg)
- [ ] Document preview in UI
- [ ] Deployment to Hugging Face Spaces

## Technologies Used

- **Frontend**: Streamlit
- **LLM**: Google Gemini AI (gemini-1.5-flash with retry logic)
- **Cloud Storage**: Google Drive API
- **Authentication**: Google Service Account
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
