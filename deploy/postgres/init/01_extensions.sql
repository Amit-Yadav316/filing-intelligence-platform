-- Runs once, when the Postgres data directory is first created.
--
-- The pgvector image ships the extension binary but does not enable it in any
-- database; that is a per-database action. Enabling it here means a fresh
-- `docker compose up` produces a database the vector index can be built in,
-- with no manual step between clone and working system.

CREATE EXTENSION IF NOT EXISTS vector;

-- Used for trigram similarity on company names and ticker lookup, where exact
-- match is too brittle and full-text search is the wrong tool.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Confirms the extensions are present in the container logs, so a failed
-- enable is visible at startup rather than as a confusing error on first query.
DO $$
BEGIN
    RAISE NOTICE 'pgvector version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END
$$;
