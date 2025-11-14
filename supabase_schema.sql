-- ================================================
-- Supabase PostgreSQL Schema for RAG Chatbot
-- Vector embeddings storage with pgvector extension
-- ================================================

-- Enable pgvector extension (run this first if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables if recreating
DROP TABLE IF EXISTS document_embeddings CASCADE;
DROP TABLE IF EXISTS document_metadata CASCADE;

-- Table: document_metadata
-- Tracks document-level information and fingerprints
CREATE TABLE document_metadata (
    document_id TEXT PRIMARY KEY,
    document_name TEXT NOT NULL,
    document_hash TEXT NOT NULL,  -- SHA-256 hash of content for change detection
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: document_embeddings
-- Stores text chunks with their 768-dimensional embeddings
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,  -- Format: {document_id}_chunk_{index}
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,  -- 768-dim vectors from sentence-transformers
    metadata JSONB,  -- Flexible metadata storage (source, page, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key to document_metadata
    CONSTRAINT fk_document FOREIGN KEY (document_id) 
        REFERENCES document_metadata(document_id) 
        ON DELETE CASCADE,
    
    -- Unique constraint on chunk_id
    CONSTRAINT unique_chunk UNIQUE (chunk_id)
);

-- Indexes for performance
CREATE INDEX idx_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_embeddings_chunk_id ON document_embeddings(chunk_id);

-- Vector similarity search index (HNSW for fast approximate nearest neighbor)
CREATE INDEX idx_embeddings_vector ON document_embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index on document metadata
CREATE INDEX idx_metadata_document_hash ON document_metadata(document_hash);
CREATE INDEX idx_metadata_updated_at ON document_metadata(updated_at);

-- Function: cosine_similarity_search
-- Helper function for semantic search queries
CREATE OR REPLACE FUNCTION cosine_similarity_search(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    chunk_id TEXT,
    document_id TEXT,
    content TEXT,
    similarity float,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.chunk_id,
        e.document_id,
        e.content,
        1 - (e.embedding <=> query_embedding) as similarity,
        e.metadata
    FROM document_embeddings e
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: update_document_timestamp
-- Automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_document_metadata_timestamp
    BEFORE UPDATE ON document_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_document_embeddings_timestamp
    BEFORE UPDATE ON document_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Grant permissions (adjust based on your setup)
-- For service role: full access
-- For anon/authenticated: read-only access
ALTER TABLE document_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
CREATE POLICY "Service role has full access to metadata"
    ON document_metadata
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to embeddings"
    ON document_embeddings
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Policy: Allow public read-only access for chatbot queries
CREATE POLICY "Public read access to metadata"
    ON document_metadata
    FOR SELECT
    TO anon, authenticated
    USING (true);

CREATE POLICY "Public read access to embeddings"
    ON document_embeddings
    FOR SELECT
    TO anon, authenticated
    USING (true);

-- View: document_stats
-- Useful for monitoring and debugging
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    dm.document_id,
    dm.document_name,
    dm.chunk_count,
    dm.updated_at,
    COUNT(de.id) as actual_chunk_count,
    dm.document_hash
FROM document_metadata dm
LEFT JOIN document_embeddings de ON dm.document_id = de.document_id
GROUP BY dm.document_id, dm.document_name, dm.chunk_count, dm.updated_at, dm.document_hash;

-- Sample query: Find similar chunks
-- SELECT * FROM cosine_similarity_search(
--     (SELECT embedding FROM document_embeddings LIMIT 1),
--     0.5,
--     10
-- );
