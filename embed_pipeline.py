#!/usr/bin/env python3
"""
Standalone embedding pipeline for RAG chatbot.
Processes Google Drive documents, generates embeddings, and stores in PostgreSQL.

Usage:
    python embed_pipeline.py [--force-rebuild]

Environment Variables Required:
    - GEMINI_API_KEY: Google Gemini API key (for LLM only, not embeddings)
    - GOOGLE_SERVICE_ACCOUNT_KEY: Service account JSON credentials
    - GOOGLE_DRIVE_FOLDER_ID: Drive folder ID to process
    - SUPABASE_DATABASE_URL: PostgreSQL connection string
"""

import os
import sys
import argparse
from typing import List, Dict
from drive_service import GoogleDriveService
from postgres_embedding_store import PostgresEmbeddingStore
from embedding_store import compute_text_hash, compute_document_hash
import re

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)


class EmbeddingPipeline:
    """Standalone pipeline for processing documents and generating embeddings"""
    
    def __init__(self):
        """Initialize pipeline components"""
        print("="*60)
        print("🚀 RAG Chatbot Embedding Pipeline")
        print("="*60)
        
        # Validate environment
        self._validate_environment()
        
        # Initialize components
        print("\n📦 Initializing components...")
        self.drive_service = GoogleDriveService(
            folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        )
        
        self.embedding_store = PostgresEmbeddingStore()
        
        print("🔄 Loading sentence-transformers model...")
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device='cpu'
        )
        print("✅ Embedding model loaded (768-dim, local)")
        
        print("✅ Pipeline initialization complete\n")
    
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = [
            "GOOGLE_SERVICE_ACCOUNT_KEY",
            "GOOGLE_DRIVE_FOLDER_ID",
            "SUPABASE_DATABASE_URL"
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            sys.exit(1)
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks (same logic as rag_pipeline_vector.py)"""
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            last_sentence = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? ')
            )
            
            if last_sentence > chunk_size * 0.5:
                end = start + last_sentence + 1
            else:
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.7:
                    end = start + last_space
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def _generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings using local sentence-transformers model"""
        if not texts:
            return []
        
        print(f"  🔄 Generating {len(texts)} embeddings locally...")
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            print(f"    Batch {batch_num}/{total_batches}")
            
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            
            for embedding in batch_embeddings:
                embeddings.append(embedding.tolist())
        
        print(f"  ✅ Generated {len(embeddings)} embeddings (768-dim, 0 API calls)")
        return embeddings
    
    def process_documents(self, force_rebuild: bool = False):
        """
        Main pipeline: load documents, detect changes, generate embeddings, update DB
        
        Args:
            force_rebuild: If True, reprocess all documents regardless of changes
        """
        print("📥 Loading documents from Google Drive...")
        documents = self.drive_service.load_documents()
        print(f"✅ Loaded {len(documents)} documents\n")
        
        # Get existing document hashes from database
        if force_rebuild:
            print("🔄 Force rebuild mode: processing all documents\n")
            cached_hashes = {}
        else:
            cached_hashes = self.embedding_store.get_all_document_hashes()
            print(f"📋 Found {len(cached_hashes)} documents in database\n")
        
        # Detect changed documents
        current_hashes = {
            doc['id']: compute_document_hash(doc) 
            for doc in documents
        }
        
        changed_docs = [
            doc for doc in documents
            if force_rebuild or cached_hashes.get(doc['id']) != current_hashes[doc['id']]
        ]
        
        if not changed_docs:
            print("✅ All documents up to date. No processing needed.")
            return
        
        print(f"🔄 Processing {len(changed_docs)} changed/new documents:\n")
        
        for doc in changed_docs:
            self._process_single_document(doc, current_hashes[doc['id']])
        
        # Show final stats
        print("\n" + "="*60)
        stats = self.embedding_store.get_stats()
        print(f"📊 Pipeline Statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   Avg chunks/doc: {stats.get('avg_chunks_per_doc', 0):.1f}")
        print("="*60)
        print("✅ Pipeline execution complete!\n")
    
    def _process_single_document(self, doc: Dict[str, str], doc_hash: str):
        """Process a single document: chunk, embed, upsert"""
        doc_id = doc['id']
        doc_name = doc['name']
        content = doc['content']
        
        print(f"📄 Processing: {doc_name}")
        print(f"   Document ID: {doc_id}")
        print(f"   Content length: {len(content)} chars")
        
        # Step 1: Delete existing chunks for this document
        self.embedding_store.delete_document_chunks(doc_id)
        
        # Step 2: Chunk the document
        chunks = self._chunk_text(content)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        if not chunks:
            print("   ⚠️ No chunks created, skipping")
            return
        
        # Step 3: Generate embeddings
        embeddings = self._generate_embeddings(chunks)
        
        # Step 4: Prepare data for bulk upsert
        chunk_data = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{idx}"
            text_hash = compute_text_hash(chunk_text)
            
            chunk_data.append({
                'chunk_id': chunk_id,
                'document_id': doc_id,
                'content': chunk_text,
                'embedding': embedding,
                'metadata': {
                    'document_name': doc_name,
                    'chunk_index': idx,
                    'text_hash': text_hash
                }
            })
        
        # Step 5: Bulk upsert to database
        self.embedding_store.bulk_upsert(chunk_data)
        
        # Step 6: Update document metadata
        self.embedding_store.update_document_metadata(
            document_id=doc_id,
            document_name=doc_name,
            document_hash=doc_hash,
            chunk_count=len(chunks)
        )
        
        print(f"   ✅ Stored {len(chunk_data)} embeddings in database\n")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot Embedding Pipeline - Process documents and generate embeddings"
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild all embeddings (ignore change detection)'
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = EmbeddingPipeline()
        pipeline.process_documents(force_rebuild=args.force_rebuild)
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        if 'pipeline' in locals():
            pipeline.embedding_store.close()


if __name__ == "__main__":
    main()
