# Step 2 — translation reliability fix (rev. 3)

**Rev. 3 changes, after review:** `TranslationFailureTracker` no longer
treats a `CACHE_HIT` as evidence the endpoint is healthy. Previously
`CACHE_HIT` reset `consecutive_failures` to zero, same as a real
`TRANSLATED` success — so a run that interleaved failing live calls with
cache hits (`fail, hit, fail, hit, fail, hit, ...`) could see the streak
reset on every other item and never reach `MAX_CONSECUTIVE_TRANSLATION_
FAILURES`, even though every live call was failing. Only a real
`TRANSLATED` now resets the streak; only a real `WARNING_TRANSLATION_
FAILED` extends it; `CACHE_HIT` and `SKIPPED_SAME_LANGUAGE` do neither,
because neither one is a request to the endpoint at all. Also added an
explicit `successful_translations == required_translations` check inside
`translation_validity`, alongside `failed_required_translations == 0` —
redundant given the current status set, but stated so the rule can't be
silently misread if a new status is ever added. The test harness used to
verify this patch (`test_preprocess_v2_reliability.py`, 57 checks) is now
included in this ZIP so the "N tests passed" claim can be run and checked
directly rather than taken on faith.

**Rev. 2 changes:** raised the normal request pacing from 0.35s to 0.5s;
added a 15-second global cooldown after a translation exhausts all retries
(before the *next* record starts its own fresh sequence of live requests);
fixed `TranslationCache` so `misses_this_run` reflects actual cache misses
instead of successful writes (writes are now tracked separately as
`writes_this_run`); added `software_versions` (Python, spaCy,
`ca_core_news_sm`/`es_core_news_sm`, `langdetect`, `deep-translator`) and
`hard_split_translation_chunk_count` to the report.

## What was wrong

The prior full-corpus run produced 950 responses / 6,434 sentence units with
zero structural record loss (parsing and sentence alignment worked), but the
translation stage failed repeatedly across all four required directions
(ca→es, ca→en, es→ca, es→en) on ordinary interview text. The script still
reached the end and printed a completion message even though many translated
fields were missing.

**Accurate description of that run:** the structural parsing and
sentence-alignment components executed, but Step 2 preprocessing as a whole
failed because the translation component was incomplete.

A separate manual test showed `deep_translator.GoogleTranslator` working
correctly for all four directions in isolation, so the most likely cause is a
corpus-scale reliability problem (throttling / temporary blocking under
sustained request volume), not unsupported language codes or a broken call.

## What changed in `preprocess_v2.py`

- **Retry with exponential backoff**: every live translation call
  (`_translate_with_retry`) retries up to `TRANSLATE_MAX_RETRIES` (3) times
  with exponential backoff + jitter before it's treated as a failure.
- **Request pacing**: a delay (`TRANSLATE_REQUEST_DELAY_SECONDS`, raised
  from 0.35s to **0.5s** after review) after every live (non-cached) call,
  so the corpus run doesn't hammer the endpoint back-to-back.
- **Cooldown after an exhausted retry burst**: once `_translate_with_retry`
  gives up on a single translation (all `TRANSLATE_MAX_RETRIES` attempts
  failed), `translate_text` now sleeps `TRANSLATE_FAILURE_COOLDOWN_SECONDS`
  (15s) before returning `WARNING_TRANSLATION_FAILED`, so the *next*
  record doesn't immediately start hammering a throttled endpoint again.
  The 8-consecutive-failure early-abort threshold is unchanged and still
  applies on top of this.
- **Cache discipline unchanged and confirmed**: `cache.set()` is only ever
  called after a translation succeeds — a failed/empty translation is never
  cached (verified by test).
- **Cache statistics fixed**: `TranslationCache.get()` now increments
  `misses_this_run` on an actual miss (previously it only counted hits);
  `set()` no longer double-counts as a "miss" — it increments a new,
  separate `writes_this_run`. So `misses_this_run` is now genuinely "cache
  lookups that didn't find anything," not "translations successfully
  written."
- **Early abort**: `TranslationFailureTracker` counts *consecutive live*
  translation failures — only a real `TRANSLATED` resets the streak and only
  a real `WARNING_TRANSLATION_FAILED` extends it; `CACHE_HIT` does neither,
  since a cache hit never touched the endpoint and is not evidence of
  anything about its health (fixed in rev. 3 — this used to reset the streak
  the same as a real success, which meant a run interleaving failing live
  calls with cache hits could never reach the threshold). If
  `MAX_CONSECUTIVE_TRANSLATION_FAILURES` (8) is hit, a
  `TranslationReliabilityError` aborts the run — the script stops burning
  through the rest of the corpus against an endpoint that isn't answering,
  but still writes a full, honest partial report (`translation_aborted_early:
  true`, with `abort_reason`) instead of silently continuing.
- **Preflight check** (`--preflight` or automatically at the start of a full
  run): calls all four required directions with a representative
  interview-register test sentence before touching the corpus. Prints:

  ```
  ca -> es  PASS
  ca -> en  PASS
  es -> ca  PASS
  es -> en  PASS
  ```

  A full run aborts immediately (before opening the input file) if preflight
  fails.
- **Smoke test** (`--smoke-test`): runs one Catalan-primary and one
  Spanish-primary synthetic response through the *entire* pipeline (real
  translation calls, real cleaning/splitting) and checks that
  `original_text`, `topic_text_ca`, `text_es`, and `text_en` all populate for
  both, e.g.:

  ```
  Catalan response:
    original_text       ✓
    topic_text_ca       ✓
    text_es             ✓
    text_en             ✓
  ```

  Uses a throwaway cache file that is deleted afterwards — it never touches
  `translation_cache_v2.json`.
- **Per-direction + overall counters** in the report: `required_translations`,
  `successful_translations`, `failed_required_translations` (only the four
  real cross-language directions — same-language skips don't count),
  `missing_topic_text_ca`, `missing_text_es`, `missing_text_en`.
- **`STEP2_VALID` gating**: `structural_validity` (the existing
  silent-loss check) AND `translation_validity` (zero failed required
  translations, `successful_translations == required_translations`, zero
  missing fields, no early abort) both have to be `True` for `STEP2_VALID`
  to be `True`. The "Step 2 complete" success message is
  now suppressed whenever `STEP2_VALID` is `False` — a run that reaches the
  end of the loop with missing translations logs a `Step 2 FAILED` message
  instead, never "complete".
- **Software environment record**: the report now includes
  `software_versions` — Python, spaCy, the `ca_core_news_sm` /
  `es_core_news_sm` pipeline versions (read off the loaded models; reports
  `"not_loaded"` if a stub/duck-typed preprocessor without those attributes
  is used, e.g. in tests), `langdetect`, and `deep-translator`.
- **Hard-split visibility**: `_boundary_aware_chunks()` now returns how many
  pieces its last-resort character-split fallback produced (for a single
  "sentence" longer than `MAX_TRANSLATE_CHUNK`, 4,000 chars); the total is
  reported as `hard_split_translation_chunk_count` instead of being
  invisible in the output.
- **End-of-run validation report** printed to the console (and included in
  `preprocessing_language_report.json`):

  ```
  Input responses                 = 950
  Output responses                = 950
  Duplicate response IDs          = 0
  Duplicate sentence IDs          = 0
  Unmapped sentences              = 0

  Required translations           = N
  Successful translations         = N
  Failed required translations    = 0

  Missing topic_text_ca           = 0
  Missing text_es                 = 0
  Missing text_en                 = 0

  Structural validity             = True
  Translation validity            = True
  STEP2_VALID                     = True
  ```

## This is a patch, not a full project replacement

This ZIP contains only the two files that changed: `preprocess_v2.py` and
`translation_cache.py`. It does **not** contain `preprocessing.py`,
`json_handler.py`, `data/`, `requirements.txt`, or anything else in the
project.

**Copy these two files into your existing Step 2 project, overwriting the
old versions. Do not delete your project and replace it with this ZIP's
contents** — `preprocess_v2.py` depends on the already-in-place
`Preprocessor.split_sentences_strict()` and `Preprocessor.full_preprocess_v2()`
in your current `preprocessing.py`, and on `json_handler.py`, and on your
`data/input/interviews.json`, none of which are included here because they
did not change.

## CLI

```bash
python preprocess_v2.py --preflight     # check the 4 directions, exit
python preprocess_v2.py --smoke-test    # mini end-to-end check, exit
python preprocess_v2.py                 # full run (preflight runs first automatically)
```

The script exits with status 0 only when the requested mode passed / when
`STEP2_VALID` is `True` for a full run; non-zero otherwise, so it can gate a
larger pipeline or CI step.

## What to check before trusting a full run

0. `python test_preprocess_v2_reliability.py` — offline, no network/spaCy
   needed. Confirms the reliability logic itself (retry, pacing, cooldown,
   the tracker's cache-hit-doesn't-reset-the-streak fix, early abort,
   cache hit/miss/write stats, report math) still holds after any edit to
   `preprocess_v2.py` or `translation_cache.py`. All 57 checks should pass.
1. `python preprocess_v2.py --preflight` — all four directions PASS.
2. `python preprocess_v2.py --smoke-test` — all four fields populate for
   both languages.
3. Run the full corpus. Read the printed validation report (and/or
   `data/output/preprocessing_language_report.json`). Confirm
   `STEP2_VALID = True` before treating this run as usable input to later
   stages. If it's `False`, check `translation_aborted_early` /
   `abort_reason` first, then `failed_required_translations` and the
   `missing_*` counts, then re-run once the translation endpoint is
   healthy — `translation_cache_v2.json` means everything that already
   succeeded is reused, not re-requested.
