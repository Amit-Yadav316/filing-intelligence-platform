"""Embeddings, with a content-hash cache in front of them.

The corpus gets reprocessed every time the chunker changes, and most chunks are
byte-identical across those runs. Embedding them again is pure waste, so the
cache is keyed on the **content**, not on the chunk id: a chunk whose id changed
because an earlier Item gained a paragraph still hits the cache, and a chunk
whose text changed correctly misses it.

The key includes the model name and the embedding dimension. Swapping
``bge-small`` for a different model must invalidate every entry, and a cache
that silently returned 384-dimension vectors for a 768-dimension model would
fail deep inside pgvector with an error naming neither cause.

Query prefixing
---------------
BGE retrieval models are trained asymmetrically: the *query* side expects an
instruction prefix and the *passage* side expects none. Omitting it is a silent
quality loss - nothing errors, retrieval is just worse - which is exactly the
kind of defect a project measuring its own accuracy should not ship.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger
from src.observability.metrics import EMBEDDING_CACHE, STAGE_LATENCY

log = get_logger(__name__)

# Documented instruction prefix for bge-*-en-v1.5 retrieval queries.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class CacheKeyBuilder:
    """Builds the content-hash key shared by Redis and the chunk store."""

    NAMESPACE = "emb"

    @staticmethod
    def content_hash(text: str, model: str, dim: int) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(dim).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def redis_key(cls, content_hash: str) -> str:
        return f"{cls.NAMESPACE}:{content_hash}"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    computed: int = 0
    errors: int = 0
    duplicates: int = 0
    """Texts repeated within one call, embedded once and reused."""

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "computed": self.computed,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
        }


class EmbeddingService:
    """Encodes text to vectors, checking Redis before computing anything."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: Any = None,
        redis_client: Any = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self._model = model
        self._redis = redis_client
        self._redis_ready = redis_client is not None
        self.use_cache = use_cache
        self.stats = CacheStats()

    # --- lazy resources ---------------------------------------------------
    @property
    def model(self) -> Any:
        """Loaded on first use. Importing torch costs seconds and hundreds of
        megabytes, and the chunker and store must not pay that to import."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading_embedding_model", model=self.settings.embedding_model)
            self._model = SentenceTransformer(self.settings.embedding_model, device="cpu")
        return self._model

    @property
    def redis(self) -> Any:
        if not self._redis_ready:
            try:
                import redis as redis_lib

                self._redis = redis_lib.Redis.from_url(
                    self.settings.redis_url, socket_connect_timeout=2
                )
                self._redis.ping()
            except Exception as exc:
                # A missing cache must degrade to recomputation, never to a
                # failed pipeline. It is an optimisation, not a dependency.
                log.warning("redis_unavailable", error=str(exc), effect="embedding uncached")
                self._redis = None
            self._redis_ready = True
        return self._redis

    # --- encoding ---------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,  # cosine distance assumes unit vectors
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.stats.computed += len(texts)
        return [v.tolist() for v in vectors]

    def _cache_get(self, keys: Sequence[str]) -> list[list[float] | None]:
        client = self.redis if self.use_cache else None
        if client is None or not keys:
            return [None] * len(keys)
        try:
            import numpy as np

            raw = client.mget([CacheKeyBuilder.redis_key(k) for k in keys])
            out: list[list[float] | None] = []
            for blob in raw:
                if blob is None:
                    out.append(None)
                    continue
                out.append(np.frombuffer(blob, dtype=np.float32).tolist())
            return out
        except Exception as exc:
            self.stats.errors += 1
            log.warning("cache_read_failed", error=str(exc))
            return [None] * len(keys)

    def _cache_put(self, pairs: Sequence[tuple[str, list[float]]]) -> None:
        client = self.redis if self.use_cache else None
        if client is None or not pairs:
            return
        try:
            import numpy as np

            pipe = client.pipeline()
            for content_hash, vector in pairs:
                # float32 rather than JSON: 1.5 KB per vector instead of ~9 KB,
                # which is the difference between the corpus fitting in the
                # cache and evicting itself.
                blob = np.asarray(vector, dtype=np.float32).tobytes()
                pipe.setex(
                    CacheKeyBuilder.redis_key(content_hash),
                    self.settings.redis_cache_ttl_seconds,
                    blob,
                )
            pipe.execute()
        except Exception as exc:
            self.stats.errors += 1
            log.warning("cache_write_failed", error=str(exc))

    def content_hashes(self, texts: Sequence[str]) -> list[str]:
        return [
            CacheKeyBuilder.content_hash(
                t, self.settings.embedding_model, self.settings.embedding_dim
            )
            for t in texts
        ]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages, using the cache. Returns vectors in input order."""
        if not texts:
            return []

        keys = self.content_hashes(texts)

        # Collapse repeats inside the batch before touching Redis. Filings share
        # a lot of boilerplate, so this is not a theoretical saving.
        unique: dict[str, int] = {}
        for i, key in enumerate(keys):
            unique.setdefault(key, i)
        self.stats.duplicates += len(keys) - len(unique)

        unique_keys = list(unique)
        cached = self._cache_get(unique_keys)

        resolved: dict[str, list[float]] = {}
        missing: list[str] = []
        for key, vector in zip(unique_keys, cached, strict=True):
            if vector is None:
                missing.append(key)
            else:
                resolved[key] = vector

        self.stats.hits += len(resolved)
        self.stats.misses += len(missing)

        if missing:
            with STAGE_LATENCY.labels(stage="embed").time():
                fresh = self._encode([texts[unique[k]] for k in missing])
            for key, vector in zip(missing, fresh, strict=True):
                resolved[key] = vector
            self._cache_put(list(zip(missing, fresh, strict=True)))

        EMBEDDING_CACHE.labels(outcome="hit").inc(len(resolved) - len(missing))
        EMBEDDING_CACHE.labels(outcome="miss").inc(len(missing))

        return [resolved[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        The BGE instruction prefix is applied here and nowhere else - a query
        embedded as a passage matches measurably worse, and nothing errors to
        tell you.
        """
        prefixed = f"{BGE_QUERY_PREFIX}{text}" if self._is_bge() else text
        # Queries are not cached: they are rarely repeated, and caching them
        # would evict passage vectors that are.
        with STAGE_LATENCY.labels(stage="embed_query").time():
            return self._encode([prefixed])[0]

    def _is_bge(self) -> bool:
        return "bge" in self.settings.embedding_model.lower()

    def log_stats(self, context: str = "") -> dict[str, Any]:
        payload = self.stats.as_dict()
        log.info("embedding_cache_stats", context=context, **payload)
        return payload
