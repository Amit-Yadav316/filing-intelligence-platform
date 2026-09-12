-- Chunk store: one table serving both retrieval modes.
--
-- Keeping BM25 and dense vectors in the same engine is the reason Postgres was
-- chosen over a dedicated vector database. A hybrid query needs to filter by
-- company, form and section *before* ranking, and when the lexical index, the
-- vector index and the metadata live in three systems that pre-filter becomes a
-- distributed join. Here it is a WHERE clause.

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,

    -- Provenance. Every one of these appears in an API citation.
    accession       TEXT        NOT NULL,
    cik             TEXT        NOT NULL,
    company         TEXT,
    form            TEXT        NOT NULL,
    filed           DATE        NOT NULL,
    fiscal_year     INTEGER,

    -- Structure, from the chunker.
    item_number     TEXT,
    item_title      TEXT,
    section         TEXT,
    chunk_type      TEXT        NOT NULL CHECK (chunk_type IN ('prose', 'table')),
    char_start      INTEGER     NOT NULL,
    char_end        INTEGER     NOT NULL,
    token_count     INTEGER     NOT NULL,

    text            TEXT        NOT NULL,

    -- SHA256 of the chunk text plus the embedding model name. Identical to the
    -- Redis cache key, so "is this chunk already embedded" is answerable from
    -- either side without recomputing anything.
    content_hash    TEXT        NOT NULL,

    -- Offsets are only meaningful against the text a specific parser produced,
    -- so the version travels with them. A mismatch means stale, not wrong.
    parser_version  TEXT,

    embedding       vector(384),

    -- Generated, so the lexical index cannot drift from the text it indexes.
    -- Postgres maintains it on write; nothing in application code can forget to.
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lexical retrieval.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);

-- Dense retrieval. HNSW rather than IVFFlat: it needs no training step and no
-- minimum row count, so an index built on 200 chunks behaves the same way as
-- one built on 200,000. Cosine distance matches how bge-small is trained.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Metadata pre-filtering, applied before ranking in both arms.
CREATE INDEX IF NOT EXISTS chunks_meta_idx     ON chunks (cik, form, fiscal_year);
CREATE INDEX IF NOT EXISTS chunks_section_idx  ON chunks (section);
CREATE INDEX IF NOT EXISTS chunks_accession_idx ON chunks (accession);

-- Index freshness is an SLO, and answering it must not require a sequential
-- scan over the whole corpus.
CREATE INDEX IF NOT EXISTS chunks_filed_idx    ON chunks (filed DESC);
