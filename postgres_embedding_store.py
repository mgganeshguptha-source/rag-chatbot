"""
PostgreSQL-based embedding store using Supabase with pgvector extension.
Replaces JSON cache with persistent database storage.
"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Any
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from psycopg2.extensions import register_adapter, AsIs
import numpy as np


def compute_text_hash(text: str) -> str:
    """Compute stable hash for text content"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def compute_document_hash(document: Dict[str, str]) -> str:
    """Compute stable fingerprint for document (id + content hash)"""
    doc_id = document.get('id', '')
    content = document.get('content', '')
    
    combined = f"{doc_id}:{compute_text_hash(content)}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]


def adapt_numpy_array(numpy_array):
    """Adapter to convert numpy arrays to PostgreSQL arrays"""
    return AsIs(f"ARRAY{numpy_array.tolist()}")


# Register numpy array adapter
register_adapter(np.ndarray, adapt_numpy_array)


class PostgresEmbeddingStore:
    """
    Persistent embedding storage using PostgreSQL + pgvector.
    Provides the same interface as JSONEmbeddingStore for compatibility.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize PostgreSQL connection.
        
        Args:
            connection_string: PostgreSQL connection URL (defaults to SUPABASE_DATABASE_URL env var)
        """
        self.connection_string = connection_string or os.getenv("SUPABASE_DATABASE_URL")
        if not self.connection_string:
            raise ValueError("SUPABASE_DATABASE_URL environment variable not set")
        
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            print("✅ Connected to Supabase PostgreSQL database")
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {str(e)}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is alive"""
        if self.conn is None or self.conn.closed:
            self._connect()
    
    def get(self, text_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached embedding by text hash.
        
        Args:
            text_hash: SHA-256 hash of the text content
            
        Returns:
            Dict with 'embedding' key if found, None otherwise
        """
        self._ensure_connection()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT embedding FROM document_embeddings WHERE chunk_id = %s",
                    (text_hash,)
                )
                result = cur.fetchone()
                if result:
                    # Convert pgvector format to list
                    embedding = result['embedding']
                    if isinstance(embedding, str):
                        # Parse string representation if needed
                        embedding = json.loads(embedding.replace('[', '').replace(']', ''))
                    return {'embedding': list(embedding)}
                return None
        except Exception as e:
            print(f"⚠️ Error retrieving embedding from PostgreSQL: {str(e)}")
            return None
    
    def upsert(self, text_hash: str, embedding: List[float], metadata: Optional[Dict] = None):
        """
        Store or update a single embedding.
        
        Args:
            text_hash: SHA-256 hash of the text content (used as chunk_id)
            embedding: 768-dim embedding vector
            metadata: Optional metadata dict
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                # For cache compatibility, store with hash as both chunk_id and document_id
                cur.execute("""
                    INSERT INTO document_embeddings 
                        (chunk_id, document_id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (chunk_id) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """, (
                    text_hash,
                    metadata.get('document_id', 'cache') if metadata else 'cache',
                    metadata.get('content', '') if metadata else '',
                    embedding,
                    json.dumps(metadata) if metadata else '{}'
                ))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"⚠️ Error upserting embedding to PostgreSQL: {str(e)}")
            raise
    
    def bulk_upsert(self, items: List[Dict[str, Any]]):
        """
        Bulk insert/update embeddings efficiently.
        
        Args:
            items: List of dicts with 'chunk_id', 'document_id', 'content', 'embedding', 'metadata'
        """
        self._ensure_connection()
        if not items:
            return
        
        try:
            with self.conn.cursor() as cur:
                # Prepare data tuples
                data = [
                    (
                        item['chunk_id'],
                        item['document_id'],
                        item['content'],
                        item['embedding'],
                        json.dumps(item.get('metadata', {}))
                    )
                    for item in items
                ]
                
                # Batch insert/update
                execute_batch(cur, """
                    INSERT INTO document_embeddings 
                        (chunk_id, document_id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (chunk_id) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """, data, page_size=100)
                
            self.conn.commit()
            print(f"✅ Bulk upserted {len(items)} embeddings to PostgreSQL")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error bulk upserting to PostgreSQL: {str(e)}")
            raise
    
    def get_all_document_hashes(self) -> Dict[str, str]:
        """
        Get all document hashes for change detection.
        
        Returns:
            Dict mapping document_id to document_hash
        """
        self._ensure_connection()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT document_id, document_hash FROM document_metadata")
                results = cur.fetchall()
                return {row['document_id']: row['document_hash'] for row in results}
        except Exception as e:
            print(f"⚠️ Error retrieving document hashes: {str(e)}")
            return {}
    
    def delete_document_chunks(self, document_id: str):
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: Document identifier
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_embeddings WHERE document_id = %s",
                    (document_id,)
                )
                deleted_count = cur.rowcount
            self.conn.commit()
            print(f"🗑️ Deleted {deleted_count} chunks for document: {document_id}")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error deleting document chunks: {str(e)}")
            raise
    
    def update_document_metadata(self, document_id: str, document_name: str, 
                                 document_hash: str, chunk_count: int):
        """
        Update document metadata after processing.
        
        Args:
            document_id: Document identifier
            document_name: Human-readable document name
            document_hash: SHA-256 hash of document content
            chunk_count: Number of chunks created
        """
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_metadata 
                        (document_id, document_name, document_hash, chunk_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id)
                    DO UPDATE SET
                        document_name = EXCLUDED.document_name,
                        document_hash = EXCLUDED.document_hash,
                        chunk_count = EXCLUDED.chunk_count,
                        updated_at = NOW()
                """, (document_id, document_name, document_hash, chunk_count))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error updating document metadata: {str(e)}")
            raise
    
    def cosine_similarity_search(self, query_embedding: List[float], 
                                 top_k: int = 5, 
                                 threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using pgvector.
        
        Args:
            query_embedding: Query vector (768-dim)
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with 'chunk_id', 'document_id', 'content', 'score', 'metadata'
        """
        self._ensure_connection()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use cosine distance operator (<=>)
                # Convert to similarity score: 1 - distance
                cur.execute("""
                    SELECT 
                        chunk_id,
                        document_id,
                        content,
                        1 - (embedding <=> %s::vector) as score,
                        metadata
                    FROM document_embeddings
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, threshold, query_embedding, top_k))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            print(f"❌ Error performing similarity search: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with total_documents, total_chunks, avg_chunks_per_doc
        """
        self._ensure_connection()
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT document_id) as total_documents,
                        COUNT(*) as total_chunks,
                        AVG(chunk_count) as avg_chunks_per_doc
                    FROM document_metadata
                """)
                stats = cur.fetchone()
                return dict(stats) if stats else {}
        except Exception as e:
            print(f"⚠️ Error retrieving stats: {str(e)}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("✅ Closed PostgreSQL connection")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
