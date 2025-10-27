import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urlparse
import sys

class WebContentService:
    """Service for detecting URLs in text and fetching web content"""
    
    MAX_URLS_PER_QUERY = 2
    MAX_CONTENT_SIZE = 1_000_000  # 1MB
    TIMEOUT_SECONDS = 10
    
    # Blocked domains for security
    BLOCKED_DOMAINS = ['localhost', '127.0.0.1', '0.0.0.0']
    
    @staticmethod
    def detect_urls(text: str) -> List[str]:
        """
        Detect URLs in text using regex.
        Returns list of unique URLs found (max MAX_URLS_PER_QUERY).
        """
        # URL regex pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        urls = re.findall(url_pattern, text)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        # Limit to MAX_URLS_PER_QUERY
        return unique_urls[:WebContentService.MAX_URLS_PER_QUERY]
    
    @staticmethod
    def is_url_allowed(url: str) -> bool:
        """Check if URL is allowed (not blocked)"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check against blocked domains
            for blocked in WebContentService.BLOCKED_DOMAINS:
                if blocked in domain:
                    return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def fetch_url_content(url: str) -> Optional[Dict[str, str]]:
        """
        Fetch content from a URL and extract readable text.
        Returns dict with 'url', 'title', and 'content' keys, or None if failed.
        """
        try:
            # Validate URL
            if not WebContentService.is_url_allowed(url):
                print(f"⚠️ Blocked URL: {url}", file=sys.stderr)
                return None
            
            print(f"🔗 Fetching content from: {url}", file=sys.stderr)
            
            # Fetch with timeout and size limit
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RAG-Chatbot/1.0)'
            }
            
            response = requests.get(
                url,
                headers=headers,
                timeout=WebContentService.TIMEOUT_SECONDS,
                stream=True
            )
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                print(f"⚠️ Unsupported content type: {content_type}", file=sys.stderr)
                return None
            
            # Read content with size limit
            content_bytes = b''
            for chunk in response.iter_content(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > WebContentService.MAX_CONTENT_SIZE:
                    print(f"⚠️ Content too large (>{WebContentService.MAX_CONTENT_SIZE} bytes)", file=sys.stderr)
                    break
            
            # Parse HTML
            soup = BeautifulSoup(content_bytes, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            
            # Limit text length
            if len(text) > 10000:
                text = text[:10000] + "..."
            
            print(f"✅ Fetched {len(text)} characters from {url}", file=sys.stderr)
            
            return {
                'url': url,
                'title': str(title).strip(),
                'content': text
            }
            
        except requests.Timeout:
            print(f"⚠️ Timeout fetching {url}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {str(e)}", file=sys.stderr)
            return None
    
    @staticmethod
    def fetch_all_urls(urls: List[str]) -> List[Dict[str, str]]:
        """
        Fetch content from multiple URLs.
        Returns list of successfully fetched content dicts.
        """
        results = []
        for url in urls:
            content = WebContentService.fetch_url_content(url)
            if content:
                results.append(content)
        
        return results
