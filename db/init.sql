CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    source       TEXT       NOT NULL,
    chunk_index  INT        NOT NULL,
    text         TEXT       NOT NULL,
    char_start   INT        NOT NULL,
    char_end     INT        NOT NULL,
    token_count  INT        NOT NULL,
    embedding    VECTOR(1024),
    created_at   TIMESTAMP  DEFAULT NOW()
);

-- ivfflat centroids are computed from existing rows at index-creation time.
-- On an empty table the index is created but ineffective; run
-- `REINDEX INDEX chunks_embedding_idx;` after bulk-inserting data.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
