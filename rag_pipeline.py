import os
import re
from typing import List, Dict, Optional
from google import genai
from google.genai import types

class RAGPipeline:
    """Simplified RAG pipeline for document-based Q&A using Gemini AI"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.gemini_client = None
        self.chunks = []  # Store all document chunks
        
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
    
    def generate_response(self, query: str) -> str:
        """Generate a response using RAG pipeline"""
        try:
            print(f"🤔 Processing query: {query[:100]}...")
            
            # Retrieve relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(query, top_k=5)
            
            if not relevant_chunks:
                return "No relevant information found in the source."
            
            # Filter chunks by relevance threshold
            filtered_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['score'] > 0.1  # Minimum threshold
            ]
            
            if not filtered_chunks:
                return "No relevant information found in the source."
            
            # Prepare context from relevant chunks
            context_parts = []
            sources = set()
            
            for chunk in filtered_chunks[:3]:  # Use top 3 chunks
                context_parts.append(chunk['content'])
                sources.add(chunk['document_name'])
            
            context = "\n\n".join(context_parts)
            source_list = ", ".join(sorted(sources))
            
            # Create prompt for Gemini
            prompt = f"""Based on the following context from documents, please answer the user's question. If the context doesn't contain enough information to answer the question, respond with "No relevant information found in the source."

Context from documents:
{context}

User question: {query}

Please provide a helpful and accurate answer based only on the information provided in the context. If you use specific information from the context, be precise and factual."""
            
            # Generate response using Gemini
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            
            if not response.text:
                return "No relevant information found in the source."
            
            # Add source information
            final_response = response.text.strip()
            
            # Add source attribution
            if len(sources) == 1:
                final_response += f"\n\n*Source: {source_list}*"
            else:
                final_response += f"\n\n*Sources: {source_list}*"
            
            return final_response
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
