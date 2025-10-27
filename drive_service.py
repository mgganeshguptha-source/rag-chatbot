import os
import json
import tempfile
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account
from text_extractors import TextExtractor


class GoogleDriveService:
    """Service for interacting with Google Drive API"""

    def __init__(self, folder_id: Optional[str] = None):
        self.folder_id = folder_id
        self.service = None
        self._initialize_service()

    def _initialize_service(self):
        """Initialize Google Drive service with authentication"""
        try:
            # Get service account credentials from environment variable
            service_account_key = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")

            if not service_account_key:
                raise ValueError(
                    "GOOGLE_SERVICE_ACCOUNT_KEY environment variable not set")

            # Parse the JSON credentials
            try:
                credentials_dict = json.loads(service_account_key)
            except json.JSONDecodeError:
                raise ValueError(
                    "Invalid JSON format in GOOGLE_SERVICE_ACCOUNT_KEY")

            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://www.googleapis.com/auth/drive.readonly'])

            # Build the service
            self.service = build('drive', 'v3', credentials=credentials)

            print("✅ Google Drive service initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Google Drive service: {str(e)}")
            raise

    def load_documents(self) -> List[Dict[str, str]]:
        """Load documents from the specified Google Drive folder (supports TXT, PDF, JPG)"""
        if not self.service:
            raise RuntimeError("Google Drive service not initialized")

        if not self.folder_id:
            raise ValueError("Google Drive folder ID not specified")

        documents = []

        try:
            # Search for supported file types (TXT, PDF, JPG/JPEG)
            supported_types = [
                "mimeType='text/plain'", "mimeType='application/pdf'",
                "mimeType='image/jpeg'", "mimeType='image/jpg'"
            ]
            type_query = " or ".join(supported_types)
            query = f"'{self.folder_id}' in parents and ({type_query}) and trashed=false"

            results = self.service.files().list(
                q=query,
                fields="files(id, name, size, modifiedTime, mimeType)",
                pageSize=100).execute()

            files = results.get('files', [])

            if not files:
                print(
                    f"⚠️ No supported files found in folder {self.folder_id}")
                print(f"   Supported types: TXT, PDF, JPG/JPEG")
                return documents

            print(f"📄 Found {len(files)} files in Google Drive folder")

            # Download and process each file
            for file_info in files:
                try:
                    file_id = file_info['id']
                    file_name = file_info['name']
                    file_type = file_info.get('mimeType', 'unknown')

                    print(f"📥 Loading: {file_name} ({file_type})")

                    # Download file content
                    file_content = self.service.files().get_media(
                        fileId=file_id).execute()

                    # Extract text based on file type
                    content = None

                    if file_type == 'text/plain':
                        # Plain text file
                        content = TextExtractor.extract_from_text(file_content)

                    elif file_type == 'application/pdf':
                        # PDF file
                        content = TextExtractor.extract_from_pdf(file_content)

                    elif file_type in ['image/jpeg', 'image/jpg']:
                        # Image file (OCR)
                        content = TextExtractor.extract_from_image(
                            file_content)

                    else:
                        print(
                            f"⚠️ Unsupported file type: {file_type}, skipping..."
                        )
                        continue

                    # Check if extraction was successful
                    if not content or not content.strip():
                        print(
                            f"⚠️ No text extracted from {file_name}, skipping..."
                        )
                        continue

                    # Add document to collection
                    documents.append({
                        'id':
                        file_id,
                        'name':
                        file_name,
                        'content':
                        content,
                        'size':
                        file_info.get('size', 0),
                        'modified_time':
                        file_info.get('modifiedTime', ''),
                        'file_type':
                        file_type
                    })

                    print(f"✅ Loaded: {file_name} ({len(content)} characters)")

                except Exception as e:
                    print(
                        f"❌ Error loading file {file_info.get('name', 'unknown')}: {str(e)}"
                    )
                    continue

            print(
                f"✅ Successfully loaded {len(documents)} documents from Google Drive"
            )
            return documents

        except Exception as e:
            print(f"❌ Error accessing Google Drive folder: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """Test the Google Drive connection"""
        if not self.service:
            return False

        try:
            # Try to get folder information
            if self.folder_id:
                folder_info = self.service.files().get(
                    fileId=self.folder_id,
                    fields="id, name, mimeType").execute()

                print(
                    f"✅ Connected to folder: {folder_info.get('name', 'Unknown')}"
                )
                return True
            else:
                # Just test API access
                self.service.files().list(pageSize=1).execute()
                print("✅ Google Drive API connection successful")
                return True

        except Exception as e:
            print(f"❌ Google Drive connection test failed: {str(e)}")
            return False

    def get_folder_info(self) -> Optional[Dict]:
        """Get information about the specified folder"""
        if not self.service or not self.folder_id:
            return None

        try:
            folder_info = self.service.files().get(
                fileId=self.folder_id,
                fields="id, name, mimeType, createdTime, modifiedTime, owners"
            ).execute()

            return folder_info

        except Exception as e:
            print(f"❌ Error getting folder info: {str(e)}")
            return None
