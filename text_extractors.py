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
        image = None
        try:
            # Validate file size (skip extremely large files that might cause issues)
            MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
            if len(file_content) > MAX_IMAGE_SIZE:
                print(f"⚠️ Image too large ({len(file_content)} bytes), skipping OCR")
                return None
            
            # Create a file-like object from bytes
            image_file = io.BytesIO(file_content)
            
            # Open and validate image
            temp_image = None
            try:
                temp_image = Image.open(image_file)
                
                # Verify image can be loaded
                temp_image.verify()
                temp_image.close()  # Close after verify
                temp_image = None
                
                # Reopen image after verify (verify closes it)
                image_file.seek(0)
                image = Image.open(image_file)
                
                # Check image dimensions (skip very large images that might cause OCR issues)
                MAX_DIMENSION = 10000
                if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
                    print(f"⚠️ Image dimensions too large ({image.width}x{image.height}), skipping OCR")
                    image.close()
                    return None
                
                # Convert to RGB if needed (some formats cause issues)
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                    
            except Exception as img_error:
                print(f"⚠️ Invalid or corrupted image file: {str(img_error)}")
                # Clean up temp_image if it's still open
                if temp_image:
                    try:
                        temp_image.close()
                    except:
                        pass
                # Clean up image if it's open
                if image:
                    try:
                        image.close()
                    except:
                        pass
                return None
            
            # Perform OCR with additional safety
            try:
                text = pytesseract.image_to_string(image)
            except Exception as ocr_error:
                print(f"⚠️ OCR failed for image: {str(ocr_error)}")
                return None
            finally:
                # Clean up
                if image:
                    try:
                        image.close()
                    except:
                        pass
            
            if not text or not text.strip():
                print("⚠️ No text detected in image (might be blank or low quality)")
                return None
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting text from image: {str(e)}")
            if image:
                try:
                    image.close()
                except:
                    pass
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
