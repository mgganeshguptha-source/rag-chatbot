import os
from typing import Optional

class Config:
    """Configuration management for the RAG chatbot"""
    
    def __init__(self):
        self.gemini_api_key = self._get_env_var("GEMINI_API_KEY")
        self.google_service_account_key = self._get_env_var("GOOGLE_SERVICE_ACCOUNT_KEY")
        self.google_drive_folder_id = self._get_env_var("GOOGLE_DRIVE_FOLDER_ID")
        
        # Validate required configuration
        self._validate_config()
    
    def _get_env_var(self, var_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default"""
        value = os.getenv(var_name, default)
        return value.strip() if value else None
    
    def _validate_config(self):
        """Validate that required configuration is present"""
        required_vars = {
            "GEMINI_API_KEY": self.gemini_api_key,
            "GOOGLE_SERVICE_ACCOUNT_KEY": self.google_service_account_key,
            "GOOGLE_DRIVE_FOLDER_ID": self.google_drive_folder_id
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nPlease set these environment variables before running the application.")
        else:
            print("✅ All required environment variables are set")
    
    @property
    def is_configured(self) -> bool:
        """Check if all required configuration is present"""
        return all([
            self.gemini_api_key,
            self.google_service_account_key,
            self.google_drive_folder_id
        ])
    
    def get_config_status(self) -> dict:
        """Get configuration status for debugging"""
        return {
            "gemini_api_key_set": bool(self.gemini_api_key),
            "google_service_account_key_set": bool(self.google_service_account_key),
            "google_drive_folder_id_set": bool(self.google_drive_folder_id),
            "fully_configured": self.is_configured
        }
