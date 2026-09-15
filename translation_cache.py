"""
translation_cache.py — Step 2 (NLLB local translation backend)

Rev. 4: this cache moved from a Google-Translate-backed cache to a local
facebook/nllb-200-distilled-600M backend. The cache key now includes the
model name and a generation-settings version string in addition to
(source_text, source_language, target_language) -- this is deliberate and
important:

  - It prevents a translation produced under one set of decoding
    parameters (e.g. num_beams=4) from being silently served back after
    those parameters change (e.g. num_beams=1) -- a settings change now
    naturally misses the cache instead of returning a stale, differently-
    generated translation under the new settings' name.
  - It prevents any accidental mixing between this cache and the old
    Google-Translate-backed cache. Per the frozen Step 2 redesign, this is
    a NEW cache file (data/cache/nllb_translation_cache_v1.json), not a
    continuation of the old translation_cache_v2.json -- the old file is
    never read by this module, and even if the two files were merged by
    hand, differing model/generation-version strings in the key would
    keep every old Google-Translate record from ever being served as an
    NLLB result (or vice versa).

Note on validity, unchanged from the previous revision: caching a
translation makes it *reproducible* (the same input reliably returns the
same stored output), not *correct*. A cached translation can still be a
bad translation -- the cache guarantees the same output every time, it
says nothing about whether that output is an accurate rendering of the
source text. Translation *quality* is assessed separately (see
preprocess_v2.py's translation_quality_summary), not by this module.
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class TranslationCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        self._dirty = False
        self._hits = 0
        self._misses = 0
        self._writes = 0

    @staticmethod
    def _key(
        text: str,
        source_lang: str,
        target_lang: str,
        model_name: str = "",
        generation_version: str = "",
    ) -> str:
        """Cache key covers source text, source language, target language,
        the model that produced (or would produce) the translation, and a
        generation-settings version string. Including source_lang matters
        because the same string of text could in principle be looked up
        under two different assumed source languages, and those are not
        guaranteed to translate the same way. Including model_name and
        generation_version matters for the reasons in the module
        docstring above -- both a model swap and a decoding-parameter
        change must produce a fresh cache miss, never a stale hit.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model_name}|{generation_version}|{source_lang}->{target_lang}:{digest}"

    def get(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_name: str = "",
        generation_version: str = "",
    ) -> Optional[dict]:
        """Returns the full cached record (dict), or None on a cache miss.

        Every call to get() is counted as either a hit or a miss here --
        this is the one place that knows whether the lookup actually found
        something, so it is the only place that should update hits_this_run
        / misses_this_run. (set() only records writes; a write is not the
        same event as a miss, even though in this codebase every miss that
        goes on to succeed is followed by exactly one set() call.)
        """
        cached = self._cache.get(self._key(text, source_lang, target_lang, model_name, generation_version))
        if cached is not None:
            self._hits += 1
        else:
            self._misses += 1
        return cached

    def contains(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_name: str = "",
        generation_version: str = "",
    ) -> bool:
        """Round 12: a pure existence check that does NOT touch hits_this_
        run/misses_this_run. get()/get_text() are the right calls during
        actual translation work -- every one of those lookups is a real
        event worth counting toward this run's cache statistics. This
        method exists for read-only, pre-flight/diagnostic checks (e.g.
        preprocess_v2.compute_primary_cache_coverage()) that need to know
        whether something is already cached WITHOUT that check itself
        being counted as a lookup -- calling get() there would silently
        inflate hits_this_run by however many sentences the coverage check
        walks, ahead of process()'s own real lookups for those same
        sentences moments later, corrupting the very stats this class
        exists to report accurately.
        """
        return self._key(text, source_lang, target_lang, model_name, generation_version) in self._cache

    def get_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model_name: str = "",
        generation_version: str = "",
    ) -> Optional[str]:
        """Convenience wrapper: returns just the translated_text string."""
        record = self.get(text, source_lang, target_lang, model_name, generation_version)
        return record["translated_text"] if record is not None else None

    def set(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translated_text: str,
        model_name: str = "",
        generation_version: str = "",
        engine: str = "local_model",
        engine_version: Optional[str] = None,
    ) -> dict:
        record = {
            "source_text": text,
            "source_language": source_lang,
            "target_language": target_lang,
            "translated_text": translated_text,
            "model_name": model_name,
            "generation_version": generation_version,
            "engine": engine,
            "engine_version": engine_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cache[self._key(text, source_lang, target_lang, model_name, generation_version)] = record
        self._writes += 1
        self._dirty = True
        return record

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translate_fn,
        model_name: str = "",
        generation_version: str = "",
        engine: str = "local_model",
        engine_version: Optional[str] = None,
    ) -> str:
        """translate_fn(text) -> str ; only called on a cache miss.
        Returns the translated_text string (the full record is what gets
        persisted; use get() directly if the caller needs the metadata).
        """
        cached = self.get(text, source_lang, target_lang, model_name, generation_version)
        if cached is not None:
            return cached["translated_text"]
        translated_text = translate_fn(text)
        record = self.set(
            text,
            source_lang,
            target_lang,
            translated_text,
            model_name=model_name,
            generation_version=generation_version,
            engine=engine,
            engine_version=engine_version,
        )
        return record["translated_text"]

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "hits_this_run": self._hits,
            "misses_this_run": self._misses,
            "writes_this_run": self._writes,
        }

    def save(self) -> None:
        """Atomically persists the cache to disk: write-temp -> flush/fsync
        -> os.replace(). `os.replace` is an atomic rename on both POSIX and
        Windows, so a reader (or a process that gets killed mid-write) can
        never observe a half-written cache file -- it either sees the
        previous complete version or the new complete version, never a
        truncated/corrupt one.

        Added after the fifth external review: this is called far more
        often now (after every successful translation batch, not just once
        at the end of a multi-hour run -- see preprocess_v2.py's
        translate_texts_batched), specifically so an interrupted run's
        already-completed translations survive and are served as
        CACHE_HIT on the next run instead of being silently recomputed. A
        write failure here (disk full, permissions, a transient I/O error)
        is reported via a log warning and never raised -- a full corpus
        translation run that has been going for hours must not crash over
        a single failed cache write. `_dirty` is left True on failure, so
        the next successful save() call still persists everything
        accumulated since the last successful write; nothing already
        cached in memory is lost by a failed save, only by the process
        exiting before any later save succeeds.
        """
        if not self._dirty:
            return
        parent = os.path.dirname(self.cache_path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                logger.warning(f"[TranslationCache] could not create cache directory {parent}: {e}")
                return
        tmp_path = f"{self.cache_path}.tmp{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.cache_path)
        except OSError as e:
            logger.warning(
                f"[TranslationCache] failed to save cache to {self.cache_path}: {e}. "
                "Translations completed so far remain in memory and will be retried "
                "on the next successful save; the run continues."
            )
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return
        self._dirty = False
