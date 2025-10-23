"""
Text extraction utilities for different file types (PDF, JPG, etc.)
"""

import io
from typing import Optional
from pypdf import PdfReader
from PIL import Image
import pytesseract


class TextExtractor:
    """Handles text extraction from various file formats"""
    
    @staticmethod
    def extract_from_pdf(file_content: bytes) -> Optional[str]:
        """
        Extract text from PDF file content
        
        Args:
            file_content: Raw bytes of the PDF file
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Create a file-like object from bytes
            pdf_file = io.BytesIO(file_content)
            
            # Read PDF
            reader = PdfReader(pdf_file)
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            # Combine all pages
            full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                print("⚠️ PDF appears to be empty or image-based (might need OCR)")
                return None
            
            return full_text
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {str(e)}")
            return None
    
    @staticmethod
    def extract_from_image(file_content: bytes) -> Optional[str]:
        """
        Extract text from image file (JPG, JPEG, PNG) using OCR
        
        Args:
            file_content: Raw bytes of the image file
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Create a file-like object from bytes
            image_file = io.BytesIO(file_content)
            
            # Open image
            image = Image.open(image_file)
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                print("⚠️ No text detected in image (might be blank or low quality)")
                return None
            
            return text
            
        except Exception as e:
            print(f"❌ Error extracting text from image: {str(e)}")
            return None
    
    @staticmethod
    def extract_from_text(file_content: bytes) -> Optional[str]:
        """
        Extract text from plain text file
        
        Args:
            file_content: Raw bytes of the text file
            
        Returns:
            Decoded text or None if decoding fails
        """
        try:
            # Try UTF-8 first
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1
                try:
                    return file_content.decode('latin-1')
                except UnicodeDecodeError:
                    print("⚠️ Could not decode text file with UTF-8 or Latin-1")
                    return None
                    
        except Exception as e:
            print(f"❌ Error extracting text: {str(e)}")
            return None
