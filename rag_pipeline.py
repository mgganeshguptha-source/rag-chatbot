import os
import re
import time
from typing import List, Dict, Optional
from google import genai
from google.genai import types

class RAGPipeline:
    """Simplified RAG pipeline for document-based Q&A using Gemini AI"""
    
    def __init__(self, gemini_api_key: str, use_extended_knowledge: bool = True):
        self.gemini_api_key = gemini_api_key
        self.gemini_client = None
        self.chunks = []  # Store all document chunks
        self.use_extended_knowledge = use_extended_knowledge  # Enable/disable general knowledge fallback
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize Gemini AI client"""
        try:
            # Initialize Gemini client
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            print("✅ Gemini AI client initialized")
            
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {str(e)}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at sentence or word boundary
            chunk_text = text[start:end]
            
            # Look for sentence endings
            last_sentence = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? ')
            )
            
            if last_sentence > chunk_size * 0.5:  # If we found a good sentence break
                end = start + last_sentence + 1
            else:
                # Look for word boundary
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.7:  # If we found a good word break
                    end = start + last_space
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter very short chunks
    
    def initialize_with_documents(self, documents: List[Dict[str, str]]):
        """Initialize the RAG pipeline with documents"""
        try:
            print(f"🔄 Initializing RAG pipeline")
            
            # Process documents and create chunks
            self.chunks = []
            
            for doc_idx, document in enumerate(documents):
                print(f"🔄 Processing document: {document['name']}")
                
                # Split document into chunks
                doc_chunks = self._chunk_text(document['content'])
                
                for chunk_idx, chunk in enumerate(doc_chunks):
                    self.chunks.append({
                        'content': chunk,
                        'document_name': document['name'],
                        'document_id': document['id'],
                        'chunk_index': chunk_idx,
                        'chunk_id': f"doc_{doc_idx}_chunk_{chunk_idx}"
                    })
            
            if not self.chunks:
                raise ValueError("No text chunks extracted from documents")
            
            print(f"📝 Generated {len(self.chunks)} text chunks")
            print(f"✅ RAG pipeline initialized with {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {str(e)}")
            raise
    
    def _simple_relevance_score(self, query: str, text: str) -> float:
        """Calculate a simple relevance score based on keyword matching"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Extract keywords from query (remove common stop words)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from'}
        query_words = [word for word in re.findall(r'\w+', query_lower) if word not in stop_words and len(word) > 2]
        
        if not query_words:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for word in query_words if word in text_lower)
        
        # Calculate score (percentage of query words found)
        score = matches / len(query_words)
        
        # Bonus for exact phrase match
        if query_lower in text_lower:
            score += 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the most relevant document chunks for a query using simple keyword matching"""
        if not self.chunks:
            raise ValueError("No documents loaded")
        
        try:
            # Calculate relevance scores
            scored_chunks = []
            for chunk in self.chunks:
                score = self._simple_relevance_score(query, chunk['content'])
                if score > 0:
                    scored_chunks.append({
                        **chunk,
                        'score': score
                    })
            
            # Sort by score
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top k
            return scored_chunks[:top_k]
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            return []
    
    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Gemini API with automatic retry logic for handling temporary failures"""
        retry_delay = 1
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt
                )
                break
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️ Gemini API temporarily unavailable (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                raise
        
        if response and response.text:
            return response.text.strip()
        return None
    
    def generate_response(self, query: str) -> str:
        """Generate a response using RAG pipeline with optional extended knowledge fallback"""
        try:
            print(f"🤔 Processing query: {query[:100]}...")
            
            # Retrieve relevant chunks from Drive documents
            relevant_chunks = self._retrieve_relevant_chunks(query, top_k=5)
            
            # Define minimum relevance threshold (stricter to avoid false positives)
            RELEVANCE_THRESHOLD = 0.4
            
            # Filter chunks by relevance threshold
            filtered_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['score'] > RELEVANCE_THRESHOLD
            ]
            
            # Check if we have high-quality Drive information
            # Require at least one chunk with good relevance score
            has_drive_info = len(filtered_chunks) > 0 and filtered_chunks[0]['score'] > 0.4
            
            if has_drive_info:
                # Use Drive documents to answer
                print(f"📄 Found {len(filtered_chunks)} relevant chunks in Drive documents")
                
                # Prepare context from relevant chunks
                context_parts = []
                sources = set()
                
                for chunk in filtered_chunks[:3]:  # Use top 3 chunks
                    context_parts.append(chunk['content'])
                    sources.add(chunk['document_name'])
                
                context = "\n\n".join(context_parts)
                source_list = ", ".join(sorted(sources))
                
                # Create prompt for Gemini with Drive context
                prompt = f"""Based on the following context from documents, please answer the user's question. If the context doesn't contain enough information to answer the question, respond with "No relevant information found in the source."

Context from documents:
{context}

User question: {query}

Please provide a helpful and accurate answer based only on the information provided in the context. If you use specific information from the context, be precise and factual."""
                
                # Generate response using Gemini
                response_text = self._call_gemini_with_retry(prompt)
                
                if not response_text:
                    return "Unable to generate response at this time. Please try again."
                
                # Check if Gemini couldn't answer from Drive context
                # If so, and extended knowledge is enabled, try general knowledge
                no_info_phrases = ["no relevant information found", "no information found", "not enough information"]
                if any(phrase in response_text.lower() for phrase in no_info_phrases):
                    if self.use_extended_knowledge:
                        print("🌐 Drive context insufficient, switching to general knowledge...")
                        # Fall back to general knowledge
                        gk_prompt = f"""Please answer the following question based on your general knowledge. Provide a clear, accurate, and helpful response.

User question: {query}

Please provide a comprehensive answer."""
                        
                        gk_response = self._call_gemini_with_retry(gk_prompt)
                        if gk_response:
                            gk_response += "\n\n*🌐 Note: This answer is based on general knowledge, as no relevant information was found in your Google Drive documents.*"
                            return gk_response
                
                # Add source attribution for Drive documents
                if len(sources) == 1:
                    response_text += f"\n\n*📄 Source: {source_list}*"
                else:
                    response_text += f"\n\n*📄 Sources: {source_list}*"
                
                return response_text
            
            else:
                # No relevant info found in Drive
                if self.use_extended_knowledge:
                    # Fall back to general knowledge
                    print("🌐 No relevant information in Drive. Using general knowledge...")
                    
                    # Create prompt for general knowledge query
                    prompt = f"""Please answer the following question based on your general knowledge. Provide a clear, accurate, and helpful response.

User question: {query}

Please provide a comprehensive answer."""
                    
                    # Generate response using Gemini's general knowledge
                    response_text = self._call_gemini_with_retry(prompt)
                    
                    if not response_text:
                        return "Unable to generate response at this time. Please try again."
                    
                    # Add attribution indicating general knowledge was used
                    response_text += "\n\n*🌐 Note: This answer is based on general knowledge, as no relevant information was found in your Google Drive documents.*"
                    
                    return response_text
                else:
                    # Extended knowledge disabled - return not found message
                    return "No relevant information found in your Google Drive documents."
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
