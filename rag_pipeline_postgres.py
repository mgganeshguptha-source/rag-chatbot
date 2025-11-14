"""
RAG Pipeline for querying PostgreSQL + pgvector (read-only mode for Streamlit app).
This version queries pre-generated embeddings from Supabase instead of generating them on startup.
"""

import os
import re
import time
from typing import List, Dict, Optional, Any
from google import genai  # type: ignore
from google.genai import types  # type: ignore
from web_content_service import WebContentService
from postgres_embedding_store import PostgresEmbeddingStore, compute_text_hash

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    print("⚠️ sentence-transformers not available (only needed for embedding generation)")


class RAGPipelinePostgres:
    """
    Query-only RAG pipeline using PostgreSQL + pgvector.
    Designed for Streamlit app - no embedding generation on startup, instant loading.
    """
    
    def __init__(self, gemini_api_key: str, use_extended_knowledge: bool = True):
        """
        Initialize RAG pipeline in query-only mode.
        
        Args:
            gemini_api_key: API key for Gemini LLM (for response generation only)
            use_extended_knowledge: Whether to fall back to general knowledge
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_client: Optional[genai.Client] = None
        self.embedding_model: Optional[Any] = None
        self.embedding_store: Optional[PostgresEmbeddingStore] = None
        self.use_extended_knowledge = use_extended_knowledge
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize Gemini AI client and PostgreSQL connection"""
        try:
            # Initialize Gemini client (for LLM generation only)
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            print("✅ Gemini AI client initialized (for LLM generation)")
            
            # Initialize PostgreSQL embedding store
            self.embedding_store = PostgresEmbeddingStore()
            print("✅ Connected to PostgreSQL database with pgvector")
            
            # Initialize embedding model for query encoding only
            if EMBEDDING_MODEL_AVAILABLE:
                print("🔄 Loading embedding model for query encoding...")
                self.embedding_model = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2',
                    device='cpu'
                )
                print("✅ Embedding model loaded (768-dim, for queries only)")
            else:
                print("⚠️ sentence-transformers not available")
                self.embedding_model = None
            
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {str(e)}")
            raise
    
    def initialize_with_documents(self, documents: List[Dict[str, str]]):
        """
        NO-OP for query-only mode.
        Documents are already embedded and stored in PostgreSQL by the pipeline.
        This method exists for API compatibility with the old implementation.
        """
        print("ℹ️  Query-only mode: Using pre-generated embeddings from PostgreSQL")
        
        # Show database stats
        if self.embedding_store:
            stats = self.embedding_store.get_stats()
            print(f"📊 Database contains {stats.get('total_chunks', 0)} chunks from {stats.get('total_documents', 0)} documents")
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query (not for documents).
        
        Args:
            query: User query text
            
        Returns:
            768-dimensional embedding vector
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding[0].tolist()
        except Exception as e:
            print(f"❌ Error generating query embedding: {str(e)}")
            raise
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant document chunks using pgvector similarity search.
        
        Args:
            query: User query text
            top_k: Number of results to return
            
        Returns:
            List of chunks with scores
        """
        if not self.embedding_store:
            raise ValueError("PostgreSQL embedding store not initialized")
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Perform similarity search in PostgreSQL
            results = self.embedding_store.cosine_similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=0.3  # Minimum similarity threshold
            )
            
            # Format results to match expected structure
            relevant_chunks = []
            for result in results:
                metadata = result.get('metadata', {})
                relevant_chunks.append({
                    'chunk_id': result['chunk_id'],
                    'document_id': result['document_id'],
                    'content': result['content'],
                    'score': result['score'],
                    'document_name': metadata.get('document_name', 'Unknown')
                })
            
            print(f"🔍 Found {len(relevant_chunks)} relevant chunks using pgvector")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Gemini API with automatic retry logic"""
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")
            
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
                
                # Check for quota exhaustion
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "Quota exceeded" in error_str:
                    print(f"⚠️ Gemini API quota exceeded. Please wait or use a different API key.")
                    raise ValueError("Gemini API quota exceeded. The free tier has daily limits. Please try again later or use an API key with higher quota.")
                
                # Check for temporary unavailability
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
    
    def generate_response(self, query: str, external_web_content: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using RAG pipeline with pgvector semantic search.
        
        Args:
            query: User query
            external_web_content: Optional web content from URLs in query
            
        Returns:
            Generated response with source attribution
        """
        try:
            print(f"🤔 Processing query: {query[:100]}...")
            
            # Retrieve relevant chunks using pgvector
            relevant_chunks = self._retrieve_relevant_chunks(query, top_k=5)
            
            # Define thresholds
            URL_DETECTION_THRESHOLD = 0.3
            RELEVANCE_THRESHOLD = 0.5
            
            # Extract URLs from relevant chunks
            drive_web_content = []
            url_candidate_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['score'] > URL_DETECTION_THRESHOLD
            ]
            
            if url_candidate_chunks:
                print("🔍 Scanning Drive documents for URLs...")
                drive_text = " ".join([chunk['content'] for chunk in url_candidate_chunks])
                drive_urls = WebContentService.detect_urls(drive_text)
                
                if drive_urls:
                    print(f"🔗 Found {len(drive_urls)} URL(s) in Drive documents")
                    drive_web_content = WebContentService.fetch_all_urls(drive_urls)
                    if drive_web_content:
                        print(f"✅ Fetched content from {len(drive_web_content)} Drive-mentioned URL(s)")
            
            # Filter chunks by relevance threshold
            filtered_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['score'] > RELEVANCE_THRESHOLD
            ]
            
            # Check if we have high-quality information
            has_drive_info = len(filtered_chunks) > 0
            has_web_info = bool(external_web_content and len(external_web_content) > 0)
            has_drive_web_info = len(drive_web_content) > 0
            
            if has_drive_info or has_web_info or has_drive_web_info:
                # Use Drive documents and/or web content to answer
                if has_drive_info:
                    print(f"📄 Found {len(filtered_chunks)} semantically relevant chunks in Drive documents")
                if has_web_info and external_web_content:
                    print(f"🔗 Found {len(external_web_content)} web sources from user query")
                if has_drive_web_info:
                    print(f"🔗 Found {len(drive_web_content)} web sources from Drive documents")
                
                # Prepare context from relevant chunks
                context_parts = []
                drive_sources = set()
                web_sources_user = []
                web_sources_drive = []
                
                # Add Drive context
                if has_drive_info:
                    context_parts.append("=== Information from Google Drive Documents ===")
                    for chunk in filtered_chunks[:3]:  # Use top 3 chunks
                        context_parts.append(chunk['content'])
                        drive_sources.add(chunk['document_name'])
                
                # Add web content from Drive-mentioned URLs
                if has_drive_web_info:
                    context_parts.append("\n=== Information from URLs mentioned in Drive Documents ===")
                    for web_item in drive_web_content:
                        context_parts.append(f"URL: {web_item['url']}\nTitle: {web_item['title']}\nContent: {web_item['content']}")
                        web_sources_drive.append(web_item['url'])
                
                # Add web content from user query
                if has_web_info and external_web_content:
                    context_parts.append("\n=== Information from URLs in Your Question ===")
                    for web_item in external_web_content:
                        context_parts.append(f"URL: {web_item['url']}\nTitle: {web_item['title']}\nContent: {web_item['content']}")
                        web_sources_user.append(web_item['url'])
                
                context = "\n\n".join(context_parts)
                
                # Create prompt for Gemini
                source_count = sum([has_drive_info, has_web_info, has_drive_web_info])
                if source_count > 1:
                    instruction = "Based on the following context from multiple sources (Google Drive documents and web links), please answer the user's question. Synthesize information from all sources where relevant."
                elif has_drive_info:
                    instruction = "Based on the following context from Google Drive documents, please answer the user's question."
                else:
                    instruction = "Based on the following context from web links, please answer the user's question."
                
                prompt = f"""{instruction}

{context}

User question: {query}

Please provide a helpful and accurate answer based on the information provided. If you use specific information from the context, be precise and factual."""
                
                # Generate response using Gemini
                response_text = self._call_gemini_with_retry(prompt)
                
                if not response_text:
                    return "Unable to generate response at this time. Please try again."
                
                # Build response with sources
                response_parts = [response_text]
                
                if drive_sources:
                    response_parts.append(f"\n\n📄 **Sources:** {', '.join(sorted(drive_sources))}")
                
                if web_sources_drive:
                    response_parts.append(f"\n🔗 **Related URLs from documents:** {', '.join(web_sources_drive[:3])}")
                
                if web_sources_user:
                    response_parts.append(f"\n🔗 **Web sources:** {', '.join(web_sources_user)}")
                
                return "".join(response_parts)
            
            else:
                # No relevant information found - use extended knowledge if enabled
                if self.use_extended_knowledge:
                    print("🌐 No relevant information in Drive/web, using Gemini's general knowledge")
                    
                    prompt = f"""The user has asked a question, but no relevant information was found in their Google Drive documents or provided web sources.

User question: {query}

Please provide a helpful answer using your general knowledge. Be clear, accurate, and concise."""
                    
                    response_text = self._call_gemini_with_retry(prompt)
                    
                    if response_text:
                        return f"{response_text}\n\n🌐 *Note: This answer is based on general knowledge, not your Drive documents.*"
                    else:
                        return "Unable to generate response at this time. Please try again."
                else:
                    return "No relevant information found in your documents. Try rephrasing your question or check if the documents contain the information you're looking for."
        
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error generating response: {error_msg}")
            return f"Error generating response: {error_msg}"
