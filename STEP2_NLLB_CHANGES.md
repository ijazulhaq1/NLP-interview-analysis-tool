# Step 2 — frozen NLLB redesign (rev. 9, patched after a sixth external code review)

This replaces the Google-Translate-backed reliability fix (rev. 1-3, see
`STEP2_RELIABILITY_CHANGES.md` for that now-retired history) with a local
`facebook/nllb-200-distilled-600M` translation backend. This is a design
change, not another reliability patch on top of the same architecture: it
removes the entire class of problem the earlier revisions were fighting
(an unversioned, intermittently-throttled web endpoint) by not depending
on a network translation service at all.

## Fixes from the external review of the previous ZIP

A careful review of the previous delivery (compiling every file, reading
`preprocess_v2.py`/`terminal.py`/`translation_cache.py` in full, and
checking actual response lengths in `data/input/interviews.json`) found
two critical issues and several smaller ones. All are fixed in this
revision:

1. **Critical -- silent truncation of long responses.** NLLB generates
   with `max_length=512` subword tokens, but this corpus's real responses
   run up to 740 words (24 responses over 400 words, 9 over 500), and
   translation happens at the whole-response level. A 740-word response
   could be translated only up to whatever fit in 512 tokens, with the
   rest silently dropped -- and the old validation never made this fatal
   (a bad length ratio only produced `translation_sanity_status=REVIEW`,
   never blocking `STEP2_VALID`). **Fixed** with boundary-aware chunking:
   `_split_into_translation_chunks()` splits a response at sentence
   boundaries into pieces of at most 150 words before translation
   (falling back to a last-resort word-boundary split only for a single
   sentence that is, by itself, over that budget), `translate_responses_
   batched()` translates all chunks (batched across responses, not just
   within one) and rejoins them in order into the one Spanish-standardized
   response the frozen methodology still calls for. Verified directly
   against this corpus's actual longest response (740 words): produces 6
   chunks of 43-144 words each, zero content lost. See "Boundary-aware
   chunking" below.
2. **Critical -- stale bridge files could bypass Step 2 entirely.** The
   previous ZIP shipped `data/output/topic_modeling_input.json`,
   `sentiment_input_ca.json`, and `fully_processed_ca.json` from the old,
   pre-NLLB pipeline. Because `_analyze_topics()`/`_analyze_sentiment()`
   only checked whether those files existed (not whether they came from a
   valid Step 2 run), selecting "2. Analyze topics" or "3. Analyze
   sentiment" as the very first action -- without ever running Step 2 --
   would have silently run against that historical content. **Fixed** two
   ways: the three files were moved out of `data/output/` into
   `data/legacy_pre_nllb_step2_outputs/` (kept, not deleted -- see that
   folder's own README for why), AND `_create_processed_versions()` now
   writes a `step2_bridge_metadata.json` sidecar (schema version +
   `step2_valid: true`) alongside the bridge files, which
   `_analyze_topics()`/`_analyze_sentiment()` now check via
   `_step2_bridge_is_fresh()` before trusting any file at those paths --
   so even a stale file reintroduced later cannot silently satisfy the
   gate again.
3. **`terminal.py` loaded BART immediately at startup.**
   `TerminalInterface.__init__()` used to construct `SentimentAnalyzer()`
   unconditionally, which loads both the nlptown sentiment model and
   `facebook/bart-large-mnli` right away -- so merely running
   `python terminal.py` started loading/downloading BART even for a
   session that only ever runs Step 2. **Fixed**: `self.analyzer` is now
   `None` until `_get_analyzer()` is actually called (from
   `_analyze_sentiment()`), matching how the NLLB components were already
   lazy.
4. **Sentiment analysis still uses English BART for the confusion
   signal**, which contradicts the frozen plan's Spanish/multilingual
   design (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`). This is correctly
   scoped as a **later Stage 5 change**, not part of Step 2 -- but running
   it unknowingly and mistaking it for the corrected pipeline would be
   worse than leaving it alone. **Fixed** by labeling it, not by
   implementing mDeBERTa here: the menu now reads "3. Analyze sentiment
   (confusion signal is legacy BART pending Stage 5 mDeBERTa)", and
   `_analyze_sentiment()` prints an explicit note before running.
5. **Round-trip diagnostic sampled the first N responses in corpus
   order**, not a representative cross-section. **Fixed** with
   `_evenly_spaced_sample()` -- a deterministic, RNG-free, evenly-spaced
   sample across the full corpus.
6. **Semantic-preservation similarity was computed one response at a
   time** (`batch_cosine_similarity([original], [translation])` inside
   the per-response loop). **Fixed**: it's now computed once, after the
   main loop, in a single batched call across every translated response.
7. **"No network dependency" was overstated.** The preflight docstring
   said "there is no network dependency left to check," which reads as a
   stronger claim than intended -- the FIRST run on a machine still needs
   network access to download the NLLB checkpoint from Hugging Face.
   **Fixed** wording, here and in `preprocess_v2.py`: "after the NLLB
   checkpoint is downloaded and cached locally, translation inference
   does not depend on an external translation service" -- the one-time
   download requirement is now stated explicitly rather than implied
   away.
8. **`sentencepiece` was not pinned** even though NLLB's tokenizer is
   SentencePiece-based. **Fixed**: added to `requirements.txt` explicitly
   rather than relying on it arriving transitively.
9. **Question-text context**: see "What context Step 2 actually preserves"
   below for an explicit, accurate statement of what is and isn't
   preserved -- this was implicitly assumed complete before and is not.

Not changed in this round, deliberately: `mDeBERTa-v3-base-mnli-xnli`
implementation (Stage 5, out of scope for Step 2), removal of the legacy
`GoogleTranslator`-based methods in `preprocessing.py` (harmless dead code
while unused; removing it is a separate, later cleanup), and the
methodology itself (Catalan -> Spanish via NLLB, one Spanish-standardized
response, sentence-level split once) -- none of that changed.

## Fixes from the second external review

A second, equally careful review of the revision above (again compiling
every file, and confirming both prior critical fixes actually work)
confirmed the NLLB redesign was "close to ready" and found 3 further
issues, one marked important. All are fixed in this revision:

1. **Important -- chunk safety was still a word-count heuristic, not a
   guarantee.** `_split_into_translation_chunks()`'s 150-word-per-chunk
   budget makes silent truncation *very unlikely* (150 words would need an
   average subword-expansion ratio above ~3.4x to reach NLLB's 512-token
   `max_length`, which real interview text does not approach), but the
   real `NLLBTranslator.translate_batch()` call still passed
   `truncation=True, max_length=512` to the tokenizer -- so an unusually
   token-dense chunk (rare words, heavy accenting) could in principle still
   be silently truncated rather than the run failing loudly. **Fixed** with
   a tokenizer-verified second pass, closing the gap the heuristic alone
   could not fully guarantee:
   - `NLLBTranslator.count_tokens(text, source_lang)` returns the actual
     NLLB subword token count for a text, using the real loaded tokenizer
     -- exactly what `generate()` would see.
   - `_enforce_real_token_budget()` is a second chunking pass, run only
     when a real `count_tokens`-backed callback is available (i.e. a real
     `NLLBTranslator`, wired in by `translate_responses_batched()` via
     `hasattr(translator, "count_tokens")` -- the offline test-double
     translators don't implement it, so they keep exercising only the
     word-heuristic path, unchanged): it re-splits, at word boundaries, any
     chunk whose ACTUAL tokenizer-measured length still exceeds NLLB's
     token limit, recursively bisecting until every resulting piece fits.
   - `NLLBTranslator._assert_within_token_limit()` is a hard, fail-closed
     backstop called immediately inside `translate_batch()`, before the
     retry loop (an oversized-token condition is a deterministic property
     of the text, not a transient failure, so retrying it would just waste
     time before failing the same way again): it measures every text in
     the batch with `count_tokens()` and raises `NLLBTranslationError`
     immediately if anything would exceed `GENERATION_PARAMS["max_length"]`
     tokens, rather than letting `truncation=True` silently discard part of
     an oversized input. In normal operation this should never actually
     fire -- reaching it means a chunk got here without having been
     measured against the real tokenizer, a chunking-safety-margin bug, not
     an expected runtime condition.
   Net effect: silent 512-token truncation is now structurally impossible
   on the production path (real `NLLBTranslator` + real tokenizer), not
   merely unlikely -- either a chunk is provably within the token limit
   before `generate()` ever sees it, or the run fails loudly and
   immediately naming the oversized chunk, never both silently truncating
   and reporting success.
2. **`STEP2_VALID` ignored translation sanity entirely.** The previous
   gate was `structural_validity AND translation_completeness_validity`
   only -- a corpus with, say, 7 language mismatches and 3 degenerate
   (empty/identical/repetitive) translation outputs could still report
   `STEP2_VALID=True`, with those failures visible only as a non-blocking
   `translation_sanity_status=REVIEW`. That conflated two very different
   kinds of "unusual": a slightly-off length ratio or a somewhat-low
   similarity score (genuinely just worth a human glance, common in
   Catalan/Spanish translation of natural interview speech) versus output
   that landed in the wrong language, or is empty/identical-to-source/
   repetitive -- output Step 2 did not actually produce usably. **Fixed**
   with a third, equally fatal validity gate:
   ```python
   translation_output_validity = (
       language_mismatch_count == 0
       and degenerate_output_count == 0
   )
   STEP2_VALID = (
       structural_validity
       and translation_completeness_validity
       and translation_output_validity
   )
   ```
   `translation_sanity_status` now distinguishes three states, not two:
   `"FAIL"` when `translation_output_validity` is False (a genuinely
   serious, now-fatal failure), `"REVIEW"` when only the non-gating
   diagnostics fired (a length-ratio outlier and/or a semantic-similarity
   score under the informational bound -- reported, never blocking), and
   `"PASS"` otherwise. Length-ratio outliers and low similarity remain
   exactly as non-gating as before -- this fix narrows what "sanity"
   silently let through, it does not add new ways for the run to fail over
   ordinary phrasing variation. See "The three-tier validity model" below.
3. **Historical downstream-analysis results remained active in
   `data/output/`.** Even after moving the three Step-2-bridge-input files
   in the previous revision, `data/output/` still held
   `topic_results_bertopic.json`, `topic_results_lda.json`,
   `topic_results_nmf.json`, `topic_results_svd.json`, and
   `sentiment_results_ca.json` from the pre-NLLB pipeline. "4. Visualize
   results" (`_visualize_results()`) does no freshness/Step-2-validity
   checking of its own -- it just looks for files at those exact names --
   so it would have happily rendered these old, pre-NLLB figures
   immediately after unzipping this project, before Step 2 or Stage 3/5
   had ever run under the new pipeline. **Fixed** by relocation: all five
   files moved to `data/legacy_analysis_outputs/` (kept, not deleted --
   see that folder's own README), leaving `data/output/` completely empty
   until a real Stage 3/5 run (itself gated on a `STEP2_VALID=True` Step 2
   run) populates it again. This is a data fix, not a code change --
   `_visualize_results()` itself is unmodified. A follow-on, not
   implemented here: `_visualize_results()` could additionally check the
   `step2_bridge_metadata.json` sidecar as defense in depth, the way
   `_analyze_topics()`/`_analyze_sentiment()` already do, rather than
   relying solely on `data/output/` staying clean.

Two further points the reviewer raised explicitly as non-blocking:
- `step2_bridge_metadata.json` not storing a hash of
  `data/input/interviews.json` was flagged as a reproducibility gap at the
  time -- **this is now implemented, see "Fixes from the third external
  review" below** (the third review round asked for exactly this, bundled
  with the one required fix).
- `fully_processed_ca.json`/`sentiment_input_ca.json` are named `_ca`
  despite now holding Spanish-standardized content, which is semantically
  misleading. Renaming to `_es` (with downstream code updated to match)
  would be a clearer end-state but is a pure renaming exercise, not a
  correctness issue, and remains **not implemented** -- still explicitly
  flagged as non-blocking as of the third review.

## Fixes from the third external review

A third review confirmed every fix from the second round works as
intended (tokenizer-verified chunk safety, the three-tier validity model,
historical outputs isolated, lazy BART loading) and found **one important
issue that would definitely need fixing before running Step 2**, plus two
smaller reproducibility/reporting points the reviewer asked to bundle in
at the same time. All three are implemented in this revision:

1. **Important -- short, valid translations could falsely make
   `STEP2_VALID=False`.** This corpus's real responses include **46 of 3
   words or fewer** ("Sí", "No.", "Sí.", "Totes.", "PC.", "2009", "-No,
   no.", "-No. -No."). Two things happen legitimately for text this short:
   `langdetect` is unreliable-to-meaningless on a single word or short
   exclamation (a genuinely correct Spanish translation of "Sí." could be
   misdetected as some other language), and a correct Catalan->Spanish
   translation of a short word/acronym/number/proper noun is frequently
   and CORRECTLY *identical* across both languages ("Sí." -> "Sí."). Before
   this fix, `check_target_language()` would unconditionally run
   `langdetect` on any output length and `check_degenerate_output()` would
   unconditionally flag ANY identical source/target as degenerate --
   either of which could mark a perfectly correct short translation as a
   fatal `language_mismatch` or `degenerate_output`, invalidating the
   entire corpus over correctly-translated one-word answers. **Fixed** by
   reusing this module's existing length gate (`_is_sufficiently_long` /
   `MIN_CHARS_FOR_DIRECT_DETECTION` / `MIN_WORDS_FOR_DIRECT_DETECTION` --
   the same threshold already used for response-level source-language
   detection, not a new concept):
   - `check_target_language()`: text below the length threshold is **not
     assessed at all** -- `langdetect` is never even called. Returns
     `{"passed": None, "status": "NOT_ASSESSED_SHORT_TEXT", ...}` rather
     than guessing. `passed=None` (never `False`) specifically so a
     careless `if not passed` can't silently treat "not assessed" as "the
     language is wrong" -- see the tri-state fix below.
   - `check_degenerate_output()`: `is_identical_to_source` is still
     computed and reported at every length (informational), but only
     counted as fatal (`is_identical_to_source_fatal`, and therefore
     `flagged`) when the text is long enough that identity is genuinely
     suspicious. `is_empty` and `repetition_flag` remain fatal at **every**
     length, unchanged -- an empty translation or an n-gram repetition
     loop is never a legitimate short-text outcome the way source==target
     can be.
   - **A real tri-state bug this fix also caught and closed**: with
     `passed` now able to be `None`, the aggregate filter that used to read
     `not resp["quality"]["target_language_check"]["passed"]` would have
     silently miscounted every short/unassessed response as a language
     mismatch (`not None` is `True` in Python) -- exactly backwards from
     the intent of this fix. Corrected to check `passed is False`
     explicitly.
   - The two new short-text signals (`short_text_language_not_assessed_
     count`/`_ids` and `short_text_identical_output_count`/`_ids`) are
     reported in `translation_quality_summary`, exactly like length-ratio
     outliers and low similarity: they push `translation_sanity_status` to
     `"REVIEW"` (worth a human glance) but **never** gate `STEP2_VALID` --
     consistent with the existing REVIEW-vs-FAIL distinction from the
     second review round.
2. **Recommended -- input hash in the bridge-freshness sidecar
   (reproducibility).** `_step2_bridge_is_fresh()` previously verified only
   `schema_version` and `step2_valid`, not that the sidecar corresponds to
   the *current* `data/input/interviews.json`. If the corpus were edited
   or replaced after a successful Step 2 run, the sidecar would have
   stayed valid-looking and topic modelling/sentiment analysis could run
   against Stage 3/5 input that no longer matches the actual corpus.
   **Fixed**: the sidecar now records `input_sha256` (SHA-256 of
   `data/input/interviews.json` at the time Step 2 wrote it),
   `nllb_model_name`, and `nllb_generation_version`.
   `_step2_bridge_is_fresh()` recomputes the hash of the CURRENT input file
   and compares it against the recorded one -- a changed corpus now
   correctly invalidates the bridge. `STEP2_BRIDGE_SCHEMA_VERSION` was
   bumped (`...v1` -> `...v2`) so an old sidecar written before this hash
   existed is correctly treated as not-fresh (there's nothing to check it
   against) rather than silently trusted -- the same deliberate
   version-break pattern `GENERATION_VERSION` already uses in
   `preprocess_v2.py`.
3. **Recommended -- explicit `topic_text_es_clean_empty` count
   reconciliation.** `topic_text_es_clean` can legitimately become empty
   after stopword removal/lemmatization (e.g. a response consisting only
   of stopwords), and `JsonHandler.create_topic_modeling_input()` silently
   drops these when building the actual BERTopic input. This was
   previously visible only buried in per-response `warnings`, with no
   aggregate count -- exactly the kind of `950 responses` vs. `documents
   actually entering BERTopic` count discrepancy the old pipeline had, and
   that reviewer feedback has repeatedly flagged as something not to
   recreate uncertainty about. **Fixed**: the report now includes
   `topic_text_es_clean_empty_count`, `topic_text_es_clean_empty_ids`, and
   `expected_topic_model_document_count` (`len(responses_out) -
   topic_text_es_clean_empty_count` -- exactly what
   `create_topic_modeling_input()` will actually keep), so this can be
   reconciled from the Step 2 report alone, before ever running topic
   modelling.

Also raised, as a workflow note rather than a Step 2 code issue: **after
Step 2 succeeds, do not proceed to "3. Analyze sentiment" for the final
confusion-analysis result until BART is replaced with the agreed
multilingual mDeBERTa setup** (Stage 5, still not implemented, and
correctly labeled as such in the menu and at runtime -- see "Sentiment /
confusion analysis stay out of Step 2" below). No code change accompanies
this point; it's a reminder about when to trust option 3's *output*, not
a Step 2 defect.

## Fixes from the fourth external review

The fourth review was not a code read-through -- it was a **real runtime
crash reported from an actual machine**, running the actual delivered ZIP:
both `python test_preprocess_v2_reliability.py` (aborting at "Test 21:
terminal.py stops before Stage 3...") and `python preprocess_v2.py
--preflight` (aborting immediately after NLLB's tokenizer/config files
finished downloading, right as model loading began) crashed with the
identical native abort:

```
TensorFlow library was compiled to use AVX instructions,
but these aren't available on your machine.
zsh: abort
```

**Root cause (diagnosed by the reviewer, confirmed independently here by
reading the actual pinned `transformers==4.33.2` source, not assumed):**
this project's entire ML stack (NLLB, the nlptown/BART sentiment models,
BERTopic's SentenceTransformer embeddings) runs on PyTorch -- there is no
TensorFlow requirement anywhere in this project. But Hugging Face
`transformers` still probes for and imports TensorFlow at its OWN import
time unless told not to, and on the reviewer's Anaconda base environment,
an old TensorFlow build compiled for AVX instructions the CPU doesn't have
aborts the entire Python process outright the moment that probe runs -- a
native-library abort, not a catchable Python exception, so no amount of
`try`/`except` inside this project's own code could have caught it.
Compounding this, `terminal.py` previously imported `topic_modeling.py`
(which imports `bertopic` -> `sentence_transformers` -> `transformers`)
and `sentiment_analysis.py` (which imports `transformers` directly) at
**module level** -- so merely `import terminal`, which both
`python terminal.py` and this project's own test suite do, triggered the
crash immediately, before a single line of Step 2 logic or a single test
assertion ever ran.

This is an environment/import-isolation problem, not an NLLB methodology
problem, and it is fixed as exactly that -- **the frozen NLLB methodology
itself (facebook/nllb-200-distilled-600M, the chunking/validity/diagnostic
design from the last three rounds) is completely unchanged in this
revision.** TensorFlow itself is deliberately NOT fixed, downgraded,
upgraded, or reinstalled anywhere in this delivery -- doing so would risk
damaging a reviewer's existing environment further and would still leave
every OTHER machine with the same latent problem. The fix instead is to
make sure `transformers` never reaches for TensorFlow in the first place,
and that importing this project's own modules never forces that reach to
happen before it's actually needed.

1. **New `ml_backend.py`** -- a small, dependency-free module whose only
   job is to set `USE_TF=0`, `USE_FLAX=0`, `USE_TORCH=1`, and
   `TRANSFORMERS_NO_TF=1` via `os.environ.setdefault(...)` (never a plain
   assignment, so an operator who has deliberately set one of these
   themselves, e.g. to test a different backend, is never silently
   overridden). **Every** module in this project that eventually reaches
   `transformers` -- `preprocess_v2.py`, `sentiment_analysis.py`,
   `topic_modeling.py`, `visualization.py` -- now does `import ml_backend`
   as its literal first import, before `torch`, before `transformers`,
   before anything else, so the guard is in effect no matter which of
   these four modules Python happens to import first.

   **Verified, not assumed**, against the project's actual pinned
   `transformers==4.33.2` (this sandbox's own installed `transformers` is
   a much newer 5.16.1 that has dropped TensorFlow support entirely, so it
   was useless as a test proxy -- the exact pinned wheel was downloaded
   separately and its `transformers/utils/import_utils.py` read directly):
   with `USE_TF` set to anything outside `{"1","ON","YES","TRUE","AUTO"}`
   while `USE_TORCH` is set to something in that set, `transformers` takes
   the branch that **never even calls
   `importlib.util.find_spec("tensorflow")`** -- so a broken TensorFlow
   install cannot be discovered, imported, or crashed on, regardless of
   what's sitting in the environment. `USE_TF`/`USE_TORCH`/`USE_FLAX` are
   the three variables that do this real work in the pinned version.
   `TRANSFORMERS_NO_TF` is, by contrast, **not a variable that version of
   `transformers` reads anywhere** (confirmed by searching the entire
   extracted wheel) -- it is set anyway, harmlessly, as a defensive no-op
   in case a different library or transformers version does honor it; it
   is not what makes the fix work, and `ml_backend.py`'s own docstring
   says so plainly rather than overclaiming it.

   `ml_backend.py` also exposes `get_ml_backend_info()`, used by both the
   preflight check and the Step 2 report (see below). It deliberately never
   does `import tensorflow` to check whether TensorFlow is present --
   doing that would risk triggering the exact abort this module exists to
   prevent, just to report on it. It uses
   `importlib.util.find_spec("tensorflow")` instead, which locates a
   module via the import machinery without executing any of its code, so
   it's safe to call even with a broken TensorFlow build sitting in the
   environment.

2. **`terminal.py`'s top-level `from topic_modeling import TopicModeler`,
   `from sentiment_analysis import SentimentAnalyzer`, and
   `from visualization import Visualizer` are all removed.** Each of these
   three modules pulls in a heavy ML dependency chain at ITS OWN import
   time (`topic_modeling.py` and `visualization.py` both import
   `bertopic` -> `sentence_transformers` -> `transformers`;
   `sentiment_analysis.py` imports `transformers` directly), so importing
   THEM used to be indistinguishable, crash-wise, from importing
   `terminal.py` itself. All three are now imported lazily -- **the import
   statement itself lives inside a getter method**, not just the
   construction:

   - `_get_topic_modeler()` -- imports `topic_modeling` and returns the
     `TopicModeler` class itself (not an instance: `extract_topics` is a
     `@staticmethod`, so `TopicModeler` is never instantiated anywhere in
     this project; this just caches the import), only on first use (menu
     option 2).
   - `_get_analyzer()` (already existed for lazy *construction*; now also
     lazily *imports*) -- imports `sentiment_analysis.SentimentAnalyzer`
     and constructs it only on first use (menu option 3).
   - `_get_visualizer()` (new) -- imports `visualization.Visualizer` and
     constructs it only on first use (menu option 4).

   `TerminalInterface.__init__` now sets `self.analyzer = None`,
   `self.visualizer = None`, `self._topic_modeler_cls = None`; the two call
   sites that used the old eagerly-imported names now go through
   `self._get_topic_modeler().extract_topics(...)` and
   `self._get_visualizer().generate_all_visualizations(...)`.
   `TYPE_CHECKING`-guarded imports keep the getters' return-type
   annotations readable without importing the real classes at runtime.

   Verified empirically, not just by inspection: after this change,
   `import terminal` no longer puts `bertopic`, `sentence_transformers`,
   `umap`, `hdbscan`, `topic_modeling`, `sentiment_analysis`, or
   `visualization` into `sys.modules` -- only `torch`/`transformers`
   remain (expected and correct, since `preprocess_v2.py`, Step 2's own
   always-needed core module, still imports them eagerly, now safely
   guarded by `ml_backend`). This exact check is now a permanent
   regression test (Test 21, see "Tests" below).

3. **Test 21 (`terminal.py stops before Stage 3 when STEP2_VALID is
   False`), Test 22 (`...continues to Stage 3/5 input files only when
   STEP2_VALID is True`), and Test 23 (the bridge-freshness gate) already
   avoided constructing a real `TerminalInterface()` -- they build small
   fake classes borrowing only the specific bound methods each test
   exercises -- so, once `terminal.py`'s own top-level imports stopped
   pulling in `topic_modeling`/`sentiment_analysis`/`visualization`,
   nothing further needed to change in their bodies. What DID need fixing
   is** Test 24 (`SentimentAnalyzer (BART/nlptown) is lazy, not loaded at
   construction`)**, whose stub-interception mechanism broke silently as a
   direct consequence of fix #2 above: it used to do
   `terminal.SentimentAnalyzer = StubSentimentAnalyzer`, monkeypatching a
   module-level name that no longer exists on `terminal` at all (the real
   import now happens as `from sentiment_analysis import SentimentAnalyzer`
   *inside* `_get_analyzer()`'s body, which resolves the name from the
   `sentiment_analysis` module's own namespace, not `terminal`'s). Left
   unfixed, this test would have silently started constructing the REAL
   BART/nlptown `SentimentAnalyzer` instead of the stub -- exactly the kind
   of hidden heavyweight-model dependency this whole round exists to
   remove from the test suite. **Fixed** by patching
   `sentiment_analysis.SentimentAnalyzer` instead (restored in a `finally`
   block so no later code ever observes the stub), plus a new explicit
   check that the real class was genuinely never instantiated. A new
   regression check was also added directly after `import terminal` in
   Test 21, asserting none of `topic_modeling` / `sentiment_analysis` /
   `visualization` / `bertopic` / `sentence_transformers` / `umap` /
   `hdbscan` appear in `sys.modules` -- the exact condition whose absence
   caused the reviewer's crash in the first place.

4. **An explicit, PyTorch-only preflight check.** `run_preflight()` now
   calls `ml_backend.get_ml_backend_info()` and prints it BEFORE attempting
   to load the tokenizer/model (i.e. before the exact step that previously
   aborted on the reviewer's machine), so a run that still somehow aborts
   there at least leaves this line in the terminal output, confirming the
   guard was in effect and narrowing the abort to the model-load step
   itself:

   ```
   ML backend:
     PyTorch available       PASS
     TensorFlow required     NO (USE_TF=0)
     Transformers backend    pytorch
   ```

   The same info is attached to `run_preflight()`'s own result dict
   (`result["ml_backend"]`). Separately, the main Step 2 report (returned
   by `process()`, written to `preprocessing_language_report.json`) now
   also carries this reproducibility metadata -- both as the flat
   `{"ml_backend": "pytorch", "tensorflow_used": false}` shape requested,
   and as the fuller `ml_backend_info` dict (which additionally records
   whether TensorFlow happens to be *installed*, even though it's never
   used, as useful provenance) -- and `print_validation_report()` prints
   the same two flat fields for consistency with the preflight output.

**What this round explicitly did NOT touch**, per the reviewer's own
framing of this as narrowly an import/backend-isolation fix: the NLLB
model, the chunking/tokenization logic, the three-tier validity model, the
translation-quality diagnostics, the bridge-freshness/input-hash
mechanism, and the topic-input count reconciliation are all completely
unchanged from the third-review revision.

## Fixes from the fifth external review

The fifth review was, again, a real runtime report rather than a code
read-through: `python preprocess_v2.py` (the full corpus run) had been
running for many hours and appeared "stuck." System-level diagnostics on
the machine showed the Python process running as an **x86_64 Intel binary
under Rosetta 2 translation** on Apple Silicon, not a native `arm64`
build, with a memory footprint reaching **~10.7-10.8GB** on an ~8GB Mac.

**Diagnosis:** the process was not actually hung -- it was thrashing.
Apple Silicon's MPS backend is not a discrete GPU with its own VRAM; it
shares unified memory with the rest of the OS. `DEFAULT_BATCH_SIZE=8` was
tuned assuming a discrete GPU or a CPU run, and running that batch size
through an x86_64-under-Rosetta process pushed memory usage high enough,
on an 8GB machine, to drive the whole system into memory
compression/swapping -- which is what actually made the run appear stuck
for hours. Compounding this, the run gave almost no visible progress
information, so there was no way to tell a genuinely stuck process from a
slow-but-progressing one, and the translation cache was only saved once,
at the very end of the run -- so if the process were killed (as it
eventually needed to be) after hours of real translation work, all of it
would have been lost and would need to be recomputed from scratch.

As with the fourth review, **the frozen NLLB methodology is completely
unchanged**: same model, same Catalan->Spanish translation, same Spanish
identity path, same deterministic beam-search decoding parameters, same
tokenizer-aware chunking, same topic/sentence preprocessing, same
semantic-similarity and round-trip diagnostics, same `STEP2_VALID` logic.
Every change in this round is about making a full corpus run safer,
lighter on memory, resumable after an interruption, and visibly
trackable while it's in progress -- not about what it computes.

1. **Incremental, atomic translation-cache saving.**
   `TranslationCache.save()` is now called after every batch that reaches
   the model in `translate_texts_batched()` (successful or not), instead
   of only once at the very end of `main()`'s multi-hour run. The write
   itself is now atomic: write to a `<cache_path>.tmp<pid>` temp file,
   `flush()` + `os.fsync()`, then `os.replace()` the real cache path with
   it -- `os.replace` is an atomic rename on POSIX and Windows, so a
   reader (or a process killed mid-write) can only ever see a complete
   previous version or a complete new version, never a half-written,
   corrupt cache file. A write failure (disk full, permissions, a
   transient I/O error) is caught, logged as a warning, and never raised
   -- an hours-long run must not crash over one failed cache write; the
   cache stays marked dirty so the next successful save retries
   persisting everything accumulated so far. On restart, `TranslationCache.
   __init__` already loads the existing on-disk cache (this part was
   already correct), so already-translated items are served as
   `CACHE_HIT` and never re-sent to NLLB -- what was missing before this
   fix was simply that there was rarely anything on disk yet to resume
   from.

2. **A lower default batch size on MPS/low-memory Macs, still fully
   configurable.** New `resolve_default_batch_size(device=None)`: resolves
   the device the same way `NLLBTranslator._auto_detect_device()` does
   (cheap, no model load) and returns `LOW_MEMORY_DEVICE_BATCH_SIZE = 2`
   for MPS, `DEFAULT_BATCH_SIZE = 8` otherwise. `--batch-size`'s CLI
   default changed from a fixed `8` to `None` ("not explicitly
   requested"); `main()` resolves it via `resolve_default_batch_size()`
   only when nothing was given, logging that it did so. An explicit
   `python preprocess_v2.py --batch-size N` always wins over the automatic
   choice. `terminal.py`'s `_get_nllb_translator()` was updated the same
   way, so the interactive `python terminal.py` entry point gets the same
   low-memory-aware default, not just the CLI. This is a pure
   runtime/memory knob -- it changes how many (smaller) model calls are
   made, never the model, decoding parameters, or translation output (see
   Test 40).

3. **A live NLLB progress bar.** `translate_responses_batched()` (the
   response-level translation function `process()` calls) now accepts
   `show_progress=True`, which renders a `tqdm` bar with one tick per
   fully-completed RESPONSE (not per chunk -- a multi-chunk response only
   advances the bar once every one of its chunks has resolved), captioned
   with running counts and the resolved device, e.g.:

   ```
   NLLB ca->es: 43%|########5           | 187/435 [01:46:21<02:19:08, fresh=153 cache_hits=34 failed=0 chunks=241 device=mps]
   ```

   `main()`'s full-run call to `process()`, and `terminal.py`'s
   `_create_processed_versions()`, both pass `show_progress=True`.

4. **Periodic "[NLLB checkpoint]" progress/checkpoint logs.** Every
   `PROGRESS_CHECKPOINT_INTERVAL` (25) fully-completed responses, a
   persistent log line is printed in addition to the live tqdm bar, so
   progress is visible even when stdout isn't a TTY (piped to a log file,
   redirected, etc., where tqdm's carriage-return updates don't show):

   ```
   [NLLB checkpoint] Processed: 200/435 | Fresh translations: 166 | Cache hits: 34 | Failed: 0 | Elapsed: 01:58:42
   ```

5. **Runtime information recorded in the Step 2 report.** The report
   (returned by `process()`, written to `preprocessing_language_report.
   json`) now includes, as flat top-level keys (matching the requested
   shape exactly) plus a fuller `runtime_info` block: `device`,
   `batch_size`, `runtime_architecture`, `elapsed_seconds`, `cache_hits`,
   `fresh_translations`, `failed_translations`, `translation_chunks` --
   all diagnostic/reproducibility metadata, same spirit as `ml_backend`/
   `tensorflow_used` added in the fourth review. `print_validation_report()`
   prints the same fields for a run's own terminal output.

6. **A runtime-architecture warning.** New `_detect_rosetta_translation()`
   (best-effort, macOS-only, using `sysctl -in sysctl.proc_translated`) and
   `get_runtime_architecture_info()`: detects `platform.machine() ==
   "x86_64"` while running under Rosetta 2 on Apple Silicon, and returns a
   warning string when it is. `process()` logs and prints this warning up
   front (before any translation work starts) whenever it applies; `run_
   preflight()` prints it too, for the same "leave a visible trail even if
   something later goes wrong" reason the ML-backend banner was added for
   in the fourth review. **This is a warning only** -- it is recorded in
   the report (`runtime_architecture`, and `running_under_rosetta`/
   `rosetta_warning` inside `runtime_info`) but never gates `STEP2_VALID`
   or any of the three validity tiers (see Test 43).

7. **New reliability tests** (Tests 38-43, all offline): the on-disk cache
   is created and updated incrementally during a run, not just once at the
   end (Test 38); an interrupted run resumes from the cache, with cached
   translations never re-sent to NLLB (Test 39); `batch_size=2` produces
   byte-identical `text_es` output and identical chunk counts to
   `batch_size=8` -- proving the memory-safety change doesn't touch
   translation output (Test 40); live progress counts are correct and
   reach exactly 100% of the response total, verified via a
   `progress_hook` that doesn't depend on tqdm's own rendering (Test 41);
   a cache-write failure (simulated with a filename exceeding the
   filesystem's `NAME_MAX`, which fails regardless of user privilege --
   unlike a permission-based simulation, which this suite may run past as
   root) is caught, logged, and never raised, and never falsely marks the
   cache as saved (Test 42); and runtime-architecture detection is
   diagnostic-only and provably does not affect `STEP2_VALID` or any
   validity tier even when a Rosetta environment is simulated (Test 43).

## Fixes from the sixth external review

The sixth review read the actual delivered ZIP against the fifth review's
own fix list and confirmed every major item was present and correctly
isolated from the scientific pipeline (MPS auto batch size, incremental
atomic cache saves, resumability, the tqdm bar, checkpoint logging, the
Rosetta warning, runtime metadata, Tests 38-43, a clean `py_compile` pass).
It found **one required regression** and **one small reporting issue**,
both fixed in this revision:

1. **Required -- `huggingface-hub==0.16.4` had gone missing from
   `requirements.txt`.** This project already hit and fixed this exact
   failure once before: `sentence-transformers==2.2.2` imports
   `cached_download` from `huggingface_hub`, a function later
   `huggingface_hub` releases removed entirely, so installing this
   project's dependencies without pinning `huggingface_hub` reproduces
   `ImportError: cannot import name 'cached_download' from
   'huggingface_hub'` -- breaking `SemanticSimilarityScorer` and
   `topic_modeling.py`'s BERTopic embeddings on a clean install. The pin
   was present in an earlier revision and was lost when `requirements.txt`
   was touched for an unrelated reason. **Fixed**: `huggingface-hub==
   0.16.4` is restored, directly next to `sentence-transformers==2.2.2`,
   with a comment explaining why it must not be removed.

2. **Small -- the live progress postfix's fresh/cache_hits/failed counts
   were CHUNK-level, not RESPONSE-level.** The tqdm bar's own tick (one
   per fully-completed response) and its "n/total" reading were always
   correct, but `_on_chunk_done()` incremented `fresh`/`cache_hits`/
   `failed` once per completed CHUNK, inside the same callback that also
   drives the once-per-response bar tick. A response spanning several
   chunks (this corpus has real responses up to ~740 words, well past the
   150-word-per-chunk budget) would get counted multiple times in those
   three numbers -- computationally harmless (nothing downstream reads
   them) but misleading for exactly the "clear progress monitoring"
   purpose this feature exists for. **Fixed** by computing true
   response-level counts: each response's chunk statuses are tracked as
   they arrive, and once every chunk for that response is in, the whole
   response is classified ONCE using the same rule this function's own
   final per-text status computation already uses (any failed chunk ->
   failed; else all-cache-hit -> cache hit; else -> fresh) -- so
   `fresh + cache_hits + failed` now always equals `responses_done`, and
   both reach exactly the true response total. `chunks=<total>` in the
   postfix is deliberately unchanged -- it was always meant as a
   chunk-level total, and it is genuinely useful as-is (context on how
   much chunking this run required), so it did not need the response-level
   fix the other three labels did. New Test 44 exercises a mixed batch of
   short (1-chunk) and long (multi-chunk) responses specifically to prove
   `fresh` no longer over-counts the multi-chunk response, and that no
   intermediate snapshot ever shows the three counts summing past the
   response total.

Nothing else changed: the NLLB model, chunking, validity logic, and every
other fifth-review fix (incremental atomic cache saves, MPS batch-size
detection, the progress bar's own tick/denominator, checkpoint logging,
the Rosetta warning, runtime-metadata reporting) are unchanged from that
revision.

## Why

1. **Reliability.** Live testing showed `deep_translator.GoogleTranslator`
   failing intermittently and inconsistently -- which direction failed
   changed between runs, independent of `source='auto'` vs. an explicit
   source language, and independent of text length. That is a sign of
   backend instability in the free scraping endpoint itself, not a bug in
   the retry/pacing/cooldown logic built around it (rev. 1-3 already
   tuned that logic twice). No amount of client-side retry tuning fixes an
   endpoint that is unreliable at the source.
2. **Reproducibility.** Comparing topic-modelling output (built from a
   translation) against sentiment/confusion output (built from a
   *different* translation call, of the same text, against the same live
   endpoint) is methodologically weak -- two independently-timed calls to
   an unversioned external service are not guaranteed to produce the same
   translation, so any downstream disagreement between the two branches
   could be an artifact of translation drift rather than a genuine
   analytical finding. Reviewer feedback flagged exactly this.

## Frozen methodology

```
Original Spanish/Catalan
          |
          | Language detection + stable IDs
          v
NLLB-200-distilled-600M
Catalan -> Spanish only
(Spanish stays unchanged -- copied, never translated Spanish->Spanish)
          |
          v
Spanish-standardized response (text_es)
          |
          +-- response-level  -> topic_text_es_raw / topic_text_es_clean
          |                      (consumed by Stage 3: BERTopic/NMF/LDA/SVD)
          |
          +-- sentence splitter (applied to the SPANISH text, once)
                +-- sentence_id ..._s000 -> text_es (Stage 3/5: sentiment + NLI)
                +-- sentence_id ..._s001 -> text_es
                +-- ...
```

Compared to the previous revision, this is a deliberate simplification:

- **English is gone from Step 2.** There is no Spanish->English
  translation and no BART-English branch here. A separate, later Stage 5
  change is expected to replace `facebook/bart-large-mnli` with
  `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (multilingual NLI, consuming
  Spanish directly) -- that is `sentiment_analysis.py`'s concern, tracked
  separately, and deliberately **not** part of this delivery. Construct
  validation for that model (label wording, hypothesis template,
  threshold, sensitivity) is deferred to that later confusion-analysis
  stage, not decided here.
- **Only one translation direction is required:** `cat_Latn -> spa_Latn`.
  A Spanish-original response is **copied** into `text_es` unchanged
  (`translation_required=False`, `text_es_status="IDENTITY_COPY"`) --
  never "translated" Spanish->Spanish.
- **Translation happens once, at the response level, before sentence
  splitting** -- not once per sentence, not once per downstream branch.
  Sentence segmentation then runs on the Spanish-standardized response, so
  the topic-modelling branch and the sentiment/NLI branch see the exact
  same translated text and the exact same sentence boundaries. Under the
  old per-sentence-routed design, two branches could in principle see text
  translated via two independent live calls; that source of divergence is
  now structurally impossible, not just unlikely.
- **Per-sentence language detection/routing is removed.** The old
  `determine_sentence_language()` and the code-switch diagnostic built on
  top of it existed specifically to decide, per sentence, which language
  to tell the translator a given sentence was in, because several
  sentence-level translation calls were in play. With exactly one
  response-level `cat_Latn -> spa_Latn` call per response, there is
  nothing left for that machinery to route -- keeping it would have been
  dead code pretending to still matter.
- **The translation engine is a local model**, not a scraped web
  endpoint. This removes the network-reliability problem entirely -- a
  local model either loads and runs on this machine, or it doesn't; it
  cannot be intermittently throttled by a remote service. It introduces a
  different concern instead: a local model can produce fluent-looking but
  wrong output without ever raising an exception. That is exactly what
  `translation_quality_summary` (below) exists to catch.

## On determinism -- an intentionally careful claim

Decoding is configured with `do_sample=False` and a fixed beam count
(`num_beams=4`), which makes translation output far more reproducible than
calls to a live, unversioned web service. This is **not** the same claim
as "byte-identical output on any hardware" -- PyTorch/device/library
version differences can still affect floating-point execution in ways
that could, in principle, change output. The claim actually made, and used
throughout this project's documentation, is:

> A fixed local NLLB checkpoint and a deterministic decoding configuration
> were used to improve computational reproducibility and remove
> dependence on a changing external translation service.

To make that claim checkable rather than asserted, every report includes
a `model_info` block (model name, requested/actual device, whether a
device fallback occurred and why, the exact generation parameters, and the
generation-version string) and a `software_versions` block (Python, torch,
transformers, sentence-transformers, langdetect, spaCy, and the loaded
`ca_core_news_sm`/`es_core_news_sm` pipeline versions).

## Device handling (M1 / MPS)

Device resolution is `cuda > mps > cpu`, decided once at
`NLLBTranslator` construction (not re-decided per call). If a generation
call on a non-CPU device fails, the translator falls back to CPU **for
the remainder of that run** -- it does not bounce between devices
sentence by sentence. The fallback is recorded once, not silently:
`requested_device`, `actual_device`, `device_fallback`, and
`fallback_reason` are all in `model_info()` and therefore in every report.

## Batching

`translate_texts_batched()` batches same-direction cache-miss texts
through `NLLBTranslator.translate_batch()`. Default `batch_size=8`
(conservative, appropriate for a laptop CPU/MPS run), overridable via
`--batch-size`. A batch that fails outright (after its own local retry) is
recorded via `ConsecutiveBatchFailureTracker`; 3 consecutive fully-failed
batches abort the run early with a full partial report, rather than
grinding through the rest of the corpus against what looks like a
systemic local problem (e.g. persistent out-of-memory).

## Boundary-aware chunking (fixes silent truncation of long responses)

NLLB generates with `max_length=512` **subword tokens**
(`GENERATION_PARAMS`). Translation happens at the response level, and this
corpus's real responses run up to 740 words (24 over 400 words, 9 over
500) -- comfortably enough to exceed 512 subword tokens for Catalan or
Spanish. Handing an over-length response straight to a single
`translate_batch()` call would not raise an error: the tokenizer silently
truncates at `max_length`, so the model would translate only the
beginning of the response, with nothing anywhere in the output signaling
that the rest was dropped.

This is fixed without changing the frozen "one Spanish-standardized
response" methodology -- chunking is purely an internal detail of how
that one response gets translated:

```
Catalan response
        |
        v
split at sentence boundaries into <=150-word pieces
(a single sentence longer than that is hard-split by words, last resort)
        |
        v
NLLB ca->es chunk 1, chunk 2, chunk 3, ...  (batched across ALL responses,
                                              not just within one response)
        |
        v
rejoined, in original order, into ONE Spanish-standardized response
        |
        v
Spanish sentence segmentation (unchanged -- still runs once, on this text)
```

`_split_into_translation_chunks(text, preprocessor, lang)` does the
splitting, using the same `preprocessor.split_sentences_strict()` already
used elsewhere, so no new sentence-boundary logic was introduced.
`translate_responses_batched()` flattens chunks from every response
needing translation into one batch-translation pass (so batching still
spans across responses' chunks, not just within a single response's
chunks), then reassembles each response from its chunks' results. A
response is translated **in full or not at all** -- if any one chunk
fails, the whole response's status is `WARNING_TRANSLATION_FAILED`, never
a silently partial translation.

150 words per chunk is a deliberately conservative, portable **first-pass**
heuristic: it would take an average subword-per-word expansion ratio above
~3.4x to reach the 512-token limit, which real Catalan/Spanish interview
text does not approach. This is a word-count heuristic rather than a real
tokenizer measurement specifically so the chunking logic is
tokenizer-independent and testable offline (`test_preprocess_v2_reliability
.py` exercises it directly, and end-to-end through `process()`, without
needing a real NLLB checkpoint) -- but a heuristic is still a heuristic,
and the second external review correctly pushed back on treating "very
unlikely" as "impossible."

**Second pass -- tokenizer-verified, closing the gap.** When the real
`NLLBTranslator` is in use, `translate_responses_batched()` wires its
`count_tokens()` method in as a second, tokenizer-verified pass inside
`_split_into_translation_chunks()` (via `hasattr(translator,
"count_tokens")`, so offline test doubles without a real tokenizer keep
exercising only the word-heuristic path, unchanged): `_enforce_real_token_
budget()` re-splits, at word boundaries, any chunk whose ACTUAL tokenizer-
measured length still exceeds `GENERATION_PARAMS["max_length"]`, and
`NLLBTranslator._assert_within_token_limit()` is a fail-closed backstop
inside `translate_batch()` itself, before the retry loop, that refuses to
translate anything that would still exceed the limit rather than letting
`truncation=True` silently discard part of it. Together, this makes silent
512-token truncation structurally impossible on the production path -- not
merely unlikely -- while keeping the word-count heuristic as the
lightweight, always-available first pass and offline-testable default. See
"Fixes from the second external review" above for the full detail.

**Cache impact, and why `GENERATION_VERSION` was bumped.** Translation is
now cached per CHUNK, not per whole response. For a long response this
means the OLD cache key (the whole raw response text) is simply never
looked up again -- chunk keys are different strings entirely, so there is
no risk of a previously-cached, possibly-truncated whole-response
translation ever being served under the new code. To make that
cache-safety property explicit and auditable rather than an incidental
side effect of key-string differences, `GENERATION_VERSION` was bumped
(`...-v1` -> `...-chunked-v2`) -- per `translation_cache.py`'s own design,
a version bump forces a clean break so nothing translated before this fix
can ever be served as if it came from after it. Practical consequence:
the first full run after upgrading to this revision will not get cache
hits from a `nllb_translation_cache_v1.json` produced by the previous
revision (a fresh `nllb_translation_cache_v1.json` cache file is expected,
and every response gets a genuine, complete translation this pass).

Verified directly against this corpus's actual longest response (740
words, read from `data/input/interviews.json`): produces 6 chunks (sizes
144/130/141/138/144/43 words), zero content lost when reassembled.
`test_preprocess_v2_reliability.py` additionally verifies, with a synthetic
70-sentence/~700-word response and a translator stub that echoes its
input back verbatim, that every one of the 70 sentence markers survives
translation and rejoining.

The report now includes a `translation_chunking` block
(`total_chunks`, `responses_requiring_multiple_chunks`,
`hard_split_chunk_count`, `max_chunk_words`), and the printed validation
report shows these counts.

## Original text preservation

`original_text` is never overwritten, for both Catalan and Spanish
responses -- required for the original-vs-translated sensitivity analysis
reviewers asked about. Every response record carries, among other fields:

```json
{
  "response_id": "interview_003::q7",
  "source_language": "ca",
  "original_text": "<verbatim original Catalan text>",
  "text_es": "<Spanish-standardized text -- translated if source was Catalan, copied unchanged if source was Spanish>",
  "translation_required": true
}
```

## What context Step 2 actually preserves

Stated precisely, because it's easy to overclaim this:

- **Response context preservation: YES.** `original_text`, `text_es`,
  `source_language`, `translation_required`, and the full
  `interview_id`/`question_id`/`response_id` hierarchy are all preserved
  per response.
- **Question-ID preservation: YES.** `question_id` (e.g. `"6.11b"`) is
  preserved on every response and sentence record.
- **Actual question-text wording: NOT YET.** `data/input/interviews.json`
  keys responses by question ID and stores response text -- it does not
  carry the literal question wording the interviewer asked, and the raw
  transcript files (`data/input/n*_timestamps_corrected_PP.txt`) mark
  question boundaries like `-.Pregunta 6.11b.-` rather than supplying a
  reusable questionnaire prompt. So a later "sentence + question text"
  contextual-sensitivity test is not yet possible from this output alone
  -- only "sentence alone" vs. "full response" context is. Enabling the
  question-text version would require locating or reconstructing the
  original questionnaire wording per question ID first; that has not been
  done here and should not be described as done until it has.

## Topic-field renaming (`*_ca` -> `*_es`)

The old `topic_text_ca_raw` / `topic_text_ca_clean` fields are renamed to
`topic_text_es_raw` / `topic_text_es_clean` -- they now genuinely hold
Spanish text (the previous names would have been actively misleading under
this design, since **every** response's topic text is Spanish now,
Catalan-sourced or not). The old `*_ca` names are historical only, from the
retired per-sentence-routed design, and appear nowhere in this revision's
active output.

## Sentiment / confusion analysis stay out of Step 2

`preprocess_v2.py` prepares `text_es` and nothing else -- it does not run
sentiment analysis, BERTopic, or the confusion/NLI model. Those are
Stage 3/5 concerns (`sentiment_analysis.py`, `topic_modeling.py`) and
consume Step 2's output through `terminal.py`'s existing
`_analyze_topics()` / `_analyze_sentiment()` methods, which are otherwise
**unmodified** by this revision.

## `terminal.py` integration

`python terminal.py` remains the normal, everyday entry point (this was
actually broken before this revision -- the file had no
`if __name__ == "__main__":` guard, so running it directly did nothing;
that guard has been added, calling `TerminalInterface().run()`, matching
what `python main.py --mode terminal` already did).

`TerminalInterface._create_processed_versions()` is now the Step 2
integration point. It:

1. Accepts `source_data=None` (used by `_analyze_topics()` /
   `_analyze_sentiment()`'s existing "input file missing -> re-run
   `_create_processed_versions()`" fallback, which used to call this
   method with **no** argument against a signature that *required* one --
   a latent bug in the original code that would have raised `TypeError`
   the first time that fallback path was actually exercised). When
   `source_data` is `None`, the full corpus is loaded from
   `data/input/interviews.json`, the same file `_select_questions()`
   reads.
2. Calls `preprocess_v2.process()` with a lazily-constructed, per-session
   `NLLBTranslator` / `TranslationCache` / `SemanticSimilarityScorer` (the
   ~600M-parameter model is loaded once and reused for the rest of the
   terminal session, not reloaded on every menu selection).
3. Writes Step 2's own audit-trail outputs
   (`preprocessed_responses_v2.json`, `preprocessed_sentences_v2.json`,
   `preprocessing_language_report.json`) **regardless** of validity, so an
   invalid run leaves a full diagnostic record, never silence.
4. **Checks `STEP2_VALID` before doing anything else.** If it is `False`,
   the method logs a clear failure message (naming which of
   `structural_validity` / `translation_completeness_validity` failed) and
   returns `False` **without** writing `fully_processed_ca.json`,
   `sentiment_input_ca.json`, or `topic_modeling_input.json`.
5. If it is `True`, it bridges `responses_out`/`sentences_out` into those
   same three legacy filenames -- kept under their historical names for
   backward compatibility with `_analyze_topics()` / `_analyze_sentiment()`,
   even though their *content* is now the Spanish-standardized text
   (`topic_text_es_clean` / `text_es`), not Catalan -- and, last, writes
   `step2_bridge_metadata.json` (schema version + `step2_valid: true`,
   plus -- added after the third external review -- `input_sha256` of
   `data/input/interviews.json`, `nllb_model_name`, and
   `nllb_generation_version`) alongside them.

`_analyze_topics()` and `_analyze_sentiment()` check for those three
bridge files before running anything, and call back into
`_create_processed_versions()` when a file is missing -- **but existence
alone is not trusted.** Both now also call `_step2_bridge_is_fresh()`,
which requires the `step2_bridge_metadata.json` sidecar to be present,
carry the current schema version, say `step2_valid: true`, AND (added
after the third external review) carry an `input_sha256` that matches a
fresh SHA-256 of the CURRENT `data/input/interviews.json`. This closes two
gaps: a bare `os.path.exists()` check cannot tell fresh, validated Step 2
output apart from bridge files from before the NLLB redesign
(Catalan-sourced) that this project's `data/output/` has, at various
points, held -- it would have let `2. Analyze topics` or `3. Analyze
sentiment`, run as the very first action, silently use historical content
without ever running Step 2 or checking `STEP2_VALID`; and even a genuinely
valid Step 2 run's sidecar would otherwise still read as fresh after the
corpus itself was edited or replaced underneath it, silently feeding
Stage 3/5 content that no longer matches the current corpus. (As of this
delivery, those three specific historical files have also been moved out
of `data/output/` entirely, into `data/legacy_pre_nllb_step2_outputs/`, as
a further, independent layer of defense -- see that folder's README.)
There is still no separate "is Step 2 valid" flag to maintain anywhere
else in `terminal.py`: the combination of "bridge file present" AND
"metadata says it came from a valid, current-schema Step 2 run against
the current corpus" *is* the gate, and it is structurally impossible for
stage 3 (topic modelling) or stage 3/5 (sentiment analysis, and later,
confusion analysis) to run against an invalid, stale, or corpus-mismatched
Step 2 output.

`self.analyzer` (the `SentimentAnalyzer` wrapping nlptown sentiment +
`facebook/bart-large-mnli`) is now constructed lazily, via
`_get_analyzer()`, on the first actual call to `_analyze_sentiment()` --
previously `TerminalInterface.__init__()` constructed it unconditionally,
so `python terminal.py` started loading/downloading BART even for a
session that only ever runs Step 2. `_analyze_sentiment()` also now prints
an explicit note that its confusion signal is still English BART run
against Spanish text, pending the Stage 5 mDeBERTa replacement, so it is
not mistaken for part of the corrected pipeline; the menu itself labels
option 3 the same way.

## Translation-quality diagnostics (reference-free, reported as distributions)

Two of these (target-language, degenerate-output) can gate `STEP2_VALID`
for **sufficiently long** text -- see the validity model below. Length
diagnostics and the two similarity-based diagnostics never gate on their
own. They are:

- **Target-language check** (`check_target_language`): langdetect on the
  *output* of translation. Catches the most catastrophic local-model
  failure mode -- echoing the source language back, or drifting into a
  third language. **Length-gated** (added after the third external
  review): text below `_is_sufficiently_long` (the same
  `MIN_CHARS_FOR_DIRECT_DETECTION` / `MIN_WORDS_FOR_DIRECT_DETECTION`
  threshold this module already used for response-level source-language
  detection) is **not assessed at all** -- `langdetect` on a single word or
  short exclamation ("Sí.", "No.", "PC.") is unreliable-to-meaningless, and
  this corpus has 46 real responses of 3 words or fewer. Returns
  `passed: None`, `status: "NOT_ASSESSED_SHORT_TEXT"` rather than guessing;
  `passed` is only ever `True`/`False` for text long enough to judge.
- **Degenerate-output check** (`check_degenerate_output`): flags empty
  output (any length, always fatal), n-gram repetition loops (any length,
  always fatal; a 3-gram repeated 4+ times is flagged), and output
  identical to a (different-language) source. **The identical-to-source
  check is length-gated** (added after the third external review): still
  computed and reported (`is_identical_to_source`) at every length, but
  only counted as fatal (`is_identical_to_source_fatal`, and therefore
  `flagged`) when the text is long enough that identity is genuinely
  suspicious -- short Catalan/Spanish text (a single word, acronym,
  number, or proper noun) is frequently and CORRECTLY identical across
  both languages ("Sí." -> "Sí."), and treating every such case as
  degenerate would invalidate correct translations.
- **Length diagnostics** (`compute_length_diagnostics`): char/token ratio
  between source and translation; ratios outside `[0.4, 3.0]` are flagged
  as outliers (informational, never gates).
- **Semantic-preservation similarity**: multilingual sentence embeddings
  (`paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`),
  cosine similarity between the original Catalan and its Spanish
  translation (informational, never gates).
- **Round-trip diagnostic**: a sample (default 100) of successfully
  translated responses are translated back `es -> ca` and compared, via
  the same embedding model, to the original Catalan (informational, never
  gates).

The two similarity-based diagnostics are reported as **distributions**
(`mean`, `median`, `stdev`, `p05`, `p25`, `min`, `count`, and a
`count_below_informational_bound` against a stated, non-gating bound) --
deliberately, per the explicit instruction not to impose an arbitrary
"similarity < 0.80 = bad" threshold before ever having seen this corpus's
actual distribution. Read the printed validation report (or
`preprocessing_language_report.json`) after the first real run and decide,
from the real numbers, whether any bound is worth acting on.

Two additional, purely informational counts (added after the third
external review) track the length-gating above without hiding it:
`short_text_language_not_assessed_count`/`_ids` (responses too short for
the target-language check to run at all) and
`short_text_identical_output_count`/`_ids` (responses whose short output
is identical to source but was NOT counted as fatal). Both push
`translation_sanity_status` to `"REVIEW"` -- worth a human glance -- but,
like length-ratio outliers and low similarity, never gate `STEP2_VALID`.

## The three-tier validity model

```
STEP2_VALID = (
    structural_validity
    and translation_completeness_validity
    and translation_output_validity
)
```

- `structural_validity`: no response lost, no duplicate response/sentence
  IDs, no sentence orphaned from its response. A **fatal** structural
  problem -- this must never be silently true.
- `translation_completeness_validity`: no early abort, zero failed
  required translations, `successful_translations == required_translations`,
  no missing `text_es` / `topic_text_es_raw` for a non-excluded response.
  Also fatal -- a translation that never happened is not usable input to
  Stage 3/5, whatever the reason.
- `translation_output_validity` (added in the second review round):
  `language_mismatch_count == 0 and degenerate_output_count == 0`. Also
  fatal -- a translation that "completed" but landed in the wrong language,
  or is empty/identical-to-source/repetitive, did not actually produce a
  usable Spanish-standardized response, even though nothing raised an
  exception. **Length-gated as of the third review round**: both counts
  now exclude the short-text cases explained in "Translation-quality
  diagnostics" above (a not-assessed language check, or a short identical
  output) -- those surface as informational, REVIEW-only counts instead,
  never as `language_mismatch_count` / `degenerate_output_count` itself.

`translation_sanity_status` has **three** states, matching which tier (if
any) actually failed:

- `"FAIL"` -- `translation_output_validity` is False: a genuinely serious,
  now-fatal failure (wrong target language, or a degenerate/repetitive/
  pathological output, on text long enough to judge). This blocks
  `STEP2_VALID`.
- `"REVIEW"` -- `translation_output_validity` is True, but a **purely
  diagnostic, non-gating** signal fired: a length-ratio outlier, a
  semantic-similarity score under the informational bound, a short output
  not assessed for target language, and/or a short output identical to
  its source. Interview responses genuinely vary in how much
  Catalan/Spanish phrasing compresses or expands, similarity scores from a
  general-purpose embedding model are a noisy proxy not ground truth, and
  this corpus's 46 real responses of 3 words or fewer are exactly the case
  where language detection and identity checks are least reliable --
  gating on any of these would produce false negatives that block an
  otherwise-good run over unusually short or phrased (but correctly
  translated) responses. Worth a human glance; never a reason to
  invalidate the corpus on its own.
- `"PASS"` -- nothing flagged.

Three real, distinct outcomes this design produces:

```
Structural validity                  True
Translation completeness             True
Translation output validity          True
Translation sanity                   REVIEW
STEP2_VALID                          True
```

```
Structural validity                  True
Translation completeness             True
Translation output validity          False
Translation sanity                   FAIL
STEP2_VALID                          False
```

```
Structural validity                  True
Translation completeness             False
Translation output validity          True
Translation sanity                   PASS
STEP2_VALID                          False
```

## Cache

New cache file: `data/cache/nllb_translation_cache_v1.json` -- **not** a
continuation of the old Google-Translate-backed
`data/output/translation_cache_v2.json`, which is never read by this
revision. The cache key includes the model name and a generation-settings
version string (`GENERATION_VERSION`) in addition to
`(source_text, source_language, target_language)`, so:

- A decoding-parameter change (e.g. `num_beams` 4 -> 1) automatically
  misses the cache instead of silently returning a translation generated
  under the old settings.
- Nothing from the old Google-Translate cache can ever be served as an
  NLLB result, or vice versa, even if the two files were merged by hand.

## What was removed

All Google-Translate-specific runtime machinery is gone from the active
Step 2 path: `GoogleTranslator` calls, network retry/backoff, request
pacing, the 15-second failure cooldown, endpoint preflight-by-live-call,
and the consecutive-*network*-failure tracker. `preprocessing.py` still
imports `deep_translator.GoogleTranslator` at module level (its legacy
`translate_to_catalan()` / `translate_to_english()` / `translate_to_spanish()`
methods are no longer called by anything in this revision) -- `preprocessing.py`
itself needed no changes, since `full_preprocess_v2()` and
`split_sentences_strict()` already accepted a `lang` parameter and already
had Spanish spaCy resources loaded.

Per-sentence language detection/routing (`determine_sentence_language()`
and its code-switch diagnostic) is also removed -- see "Frozen
methodology" above for why.

## Tests

`test_preprocess_v2_reliability.py` is a full rewrite for this
architecture (**232 checks, all offline, no network/model download** --
run it yourself; don't take this count on faith). It stubs
`NLLBTranslator` and `SemanticSimilarityScorer` at the module level for
integration-style tests of `process()`, but also exercises real,
unmodified logic directly against the real classes/module functions where
a loaded model isn't needed: `NLLBTranslator._forced_bos_token_id`'s
version-compatibility shim, `NLLBTranslator._auto_detect_device`,
`NLLBTranslator.count_tokens`/`_assert_within_token_limit` (via a
duck-typed fake tokenizer, no real SentencePiece model needed),
`SemanticSimilarityScorer.batch_cosine_similarity` against a fake
embedding model, `_split_into_translation_chunks`,
`_enforce_real_token_budget`, and `_evenly_spaced_sample`. Coverage: NLLB
language-code mapping; the identity Spanish->Spanish path; the
Catalan->Spanish path; batching; cache read/write (including the
generation-version cache-key safety); a failed batch is never cached;
per-item empty-generation handling; language-mismatch and degenerate-
output diagnostics; length diagnostics; the distribution-summary helper;
semantic-similarity computation (including that it runs as one batched
call across every translated response, not one call per response); the
round-trip diagnostic and its evenly-spaced sampling; stable IDs and a
crafted duplicate-response-ID collision; full 950-response structural
preservation via a stub NLLB; a `STEP2_VALID=True` case and a
`STEP2_VALID=False` (completeness-failure) case; a diagnostic-only,
non-gating sanity-`REVIEW` case (a length-ratio outlier with no language
mismatch or degenerate output -- `STEP2_VALID` stays True); **two
genuinely-serious-failure cases that now correctly force
`STEP2_VALID=False`** (a language-mismatch-only case, and a
degenerate/repetitive-output-only case, each isolated from the other so
both fatal conditions are proven independently sufficient); **boundary-
aware chunking** (short text stays one chunk, a long multi-sentence text
splits into several chunks with no word lost, an oversized single sentence
triggers the last-resort hard split); **tokenizer-verified chunk safety**
(`count_tokens` reads the real tokenizer correctly and includes special
tokens; `_assert_within_token_limit` raises, naming the oversized count,
only when a text genuinely exceeds the limit; `_enforce_real_token_budget`
recursively re-splits an over-budget chunk at word boundaries with zero
content lost, and leaves an already-safe chunk untouched;
`_split_into_translation_chunks` actually engages this second pass when a
`token_counter` is supplied; `translate_responses_batched` actually wires
a real translator's `count_tokens` into that second pass via `hasattr`,
while a plain stub translator without `count_tokens` is provably
unaffected); **a synthetic ~700-word response with 70 uniquely-markable
sentences translated end-to-end through `process()`, confirming every
marker survives** (the direct regression test for the truncation bug --
this test also confirms fix #2 correctly flags the stub's non-Spanish echo
output as a language mismatch, rather than asserting `STEP2_VALID=True`
against a stub that was never meant to produce realistic Spanish output);
`terminal.py` stopping before Stage 3 when Step 2 is invalid vs.
continuing (and writing the correct bridge files, plus the freshness
metadata sidecar) only when it is valid; **the bridge-freshness gate
rejecting a stale file with no metadata, a metadata sidecar saying
`step2_valid: false`, a metadata sidecar with a mismatched schema version,
a current-schema sidecar with NO `input_sha256`, and a current-schema
sidecar whose `input_sha256` does not match the current corpus, and
accepting only a sidecar whose recorded hash matches** (the direct
regression tests for the stale-bridge-file bypass and, new in this round,
the corpus-drift reproducibility gap); `terminal._hash_file` directly
(deterministic, changes when file content changes, returns `None` rather
than raising for a missing file); that a real Step 2 run's sidecar
actually records a genuine `input_sha256`/`nllb_model_name`/
`nllb_generation_version`; **the short-text false-positive fix** (a short
output is not assessed for target language at all -- `passed` is `None`,
never `False`; a long identical source/target IS fatal but a short one is
NOT; an end-to-end `process()` run on a corpus containing only a "Sí." ->
"Sí." style short response now correctly reports `STEP2_VALID=True` with
`translation_sanity_status="REVIEW"`, where it would previously have
incorrectly failed); and **`topic_text_es_clean_empty` count
reconciliation** (`topic_text_es_clean_empty_count`/`_ids` and
`expected_topic_model_document_count` correctly identify and exclude
exactly the responses whose cleaned topic text became empty); that
`SentimentAnalyzer` is constructed lazily, once, on first actual use, via a
stub that patches `sentiment_analysis.SentimentAnalyzer` directly (not
`terminal.SentimentAnalyzer`, which no longer exists as a module-level
name -- see the fourth-review section above) and confirms the real
BART/nlptown class was never instantiated; and **that merely `import
terminal` never pulls `topic_modeling`, `sentiment_analysis`,
`visualization`, `bertopic`, `sentence_transformers`, `umap`, or `hdbscan`
into `sys.modules`** -- the direct regression test for the fourth review's
TensorFlow/AVX crash, which was triggered by exactly that import. The
fifth review added its own six tests (38-43, see "Fixes from the fifth
external review" above): incremental/atomic cache saving, an interrupted
run resuming from the cache without re-sending cached items to NLLB,
`batch_size=2` vs. `8` producing byte-identical translation output, live
progress counts reaching exactly 100%, a cache-write failure being
reported safely, and runtime-architecture detection never affecting
`STEP2_VALID`. The sixth review added Test 44 (see "Fixes from the sixth
external review" above): a mixed batch of short (1-chunk) and long
(multi-chunk) responses proving the live fresh/cache_hits/failed counts
are response-level, not chunk-level -- the multi-chunk response is counted
exactly once, and no intermediate snapshot ever shows the three counts
summing past the true response total. Run it with:

```bash
python test_preprocess_v2_reliability.py
```

This suite does **not** verify that the real `facebook/nllb-200-distilled-
600M` model loads or produces good translations on real hardware -- it
can't, without network access to download the checkpoint. That real-model
verification is exactly what `--preflight` and `--smoke-test` are for (see
below), and both must be run, once, on a machine with network access,
before trusting a full corpus run.

## CLI

```bash
python preprocess_v2.py --preflight     # local model/device readiness check, exit
python preprocess_v2.py --smoke-test    # small real end-to-end run, writes to data/output/smoke_test/, exit
python preprocess_v2.py                 # full corpus run (preflight runs first automatically)
python preprocess_v2.py --batch-size 4  # override the auto-selected default batch size
```

`--batch-size` (fifth external review): when omitted, the batch size is no
longer a flat constant -- it's resolved automatically from the detected
device via `resolve_default_batch_size()`: `8` normally, or `2` on MPS
(Apple Silicon's GPU backend, which shares unified memory with the OS
rather than having its own VRAM, so a large batch there can drive a
low-memory Mac into swapping on a full corpus run -- see "Fixes from the
fifth external review"). An explicit `--batch-size N` always overrides the
automatic choice. This changes only how many (smaller) model calls are
made -- never the model, decoding parameters, or translation output.

`--preflight` checks **local environment/model readiness** -- can the
tokenizer/model load, is the resolved device usable, does a real
`cat_Latn -> spa_Latn` call on a test sentence produce non-empty,
Spanish-detected output. Precisely stated: after the NLLB checkpoint has
been downloaded and cached locally, translation inference does not depend
on an external translation service, and there is no *runtime*
network-availability check left to perform. The FIRST time this runs on a
given machine, `AutoTokenizer`/`AutoModelForSeq2SeqLM.from_pretrained()`
still needs network access to Hugging Face to download that checkpoint --
that is a one-time setup cost, not a runtime dependency, and every run
after the first is purely local.

`--smoke-test` runs one Catalan-primary and one Spanish-primary synthetic
response through the entire pipeline (real translation, real
cleaning/splitting) and checks that `original_text`, `text_es`, and
`topic_text_es_raw` all populate, plus that sentence IDs are produced.
Output goes to `data/output/smoke_test/` only -- it never touches
`data/output/preprocessed_responses_v2.json`,
`data/cache/nllb_translation_cache_v1.json`, or any other production file.

Exit code is 0 only when the requested mode passed (or, for a full run,
when `STEP2_VALID` is `True`); non-zero otherwise.

## Execution sequence for the next real run

Run these, in order, on the M1 machine (network access required once, to
download the ~2.4GB `facebook/nllb-200-distilled-600M` checkpoint -- it is
then cached locally by `transformers` and no further network access is
needed):

1. `python test_preprocess_v2_reliability.py` -- confirms the logic itself
   (offline, ~seconds, no model download).
2. `python preprocess_v2.py --preflight` -- confirms the real model loads
   and translates on this machine, and which device it resolved to
   (`cuda` / `mps` / `cpu`).
3. `python preprocess_v2.py --smoke-test` -- confirms the full pipeline
   end-to-end on two small real examples.
4. Review both outputs. Only once both pass, run `python terminal.py`
   (menu option 1) for the real 950-response corpus.
5. Read the printed validation report and/or
   `data/output/preprocessing_language_report.json`. Confirm
   `STEP2_VALID = True` before treating the run as usable input to Stage
   3 (topic modelling) or Stage 3/5 (sentiment analysis). If
   `translation_sanity_status` is `REVIEW`, read
   `translation_quality_summary` in the same report before deciding
   whether anything needs attention -- it does not, by itself, mean the
   run is unusable. If it is `FAIL`, `STEP2_VALID` will already be `False`
   for the same reason (see "The three-tier validity model" above) --
   `translation_quality_summary`'s `language_mismatch_ids` /
   `degenerate_output_ids` name exactly which responses need attention
   before re-running.
6. Do not commit or push until that real run has been reviewed.

## Files in this delivery

This is a full project archive, not a patch -- every file needed to run
the project is included:

```
terminal.py
preprocess_v2.py
preprocessing.py
ml_backend.py
translation_cache.py
test_preprocess_v2_reliability.py
requirements.txt
json_handler.py
topic_modeling.py
sentiment_analysis.py
evaluation.py
visualization.py
config.py
main.py
flask_web.py
data_audit.py
data/
  legacy_pre_nllb_step2_outputs/   (moved out of data/output/ -- see its own README.md)
  legacy_analysis_outputs/         (moved out of data/output/ in this revision -- see its own README.md)
STEP2_NLLB_CHANGES.md
STEP2_RELIABILITY_CHANGES.md   (kept for history -- the retired Google-Translate-era design)
```

`config.py`, `main.py`, `flask_web.py`, and `data_audit.py` are unchanged,
pre-existing project files, included because this is meant to be the
entire project, not a subset of it. `ml_backend.py` is new in this
revision (see "Fixes from the fourth external review" above) -- a small,
dependency-free module imported first by every module that eventually
reaches `transformers`, so it has no dependencies of its own to add to
`requirements.txt`.

Note also: `RUN_EVALUATION_ONLY.txt` (from an earlier, separate
manuscript-evaluation task) says "Do NOT run preprocessing" -- that
predates this Step 2 redesign and no longer fully applies, since this
delivery's own execution sequence explicitly calls for running Step 2. It
was left unmodified since it documents a different, earlier task, but
don't follow its "do not run preprocessing" instruction when validating
this delivery.

## Requirements

`requirements.txt` was updated to add `sentence-transformers` (needed by
`preprocess_v2.py`'s quality diagnostics and already, silently, needed by
`topic_modeling.py`'s BERTopic embeddings -- it was missing from this file
entirely before this revision). While verifying `import terminal` end to
end, `bertopic`, `umap-learn`, `hdbscan`, `seaborn`, and `tqdm` were also
found missing despite being genuinely required by `topic_modeling.py` /
`visualization.py` / `terminal.py`; they've been added too. `torch==2.0.1`
/ `transformers==4.33.2` are unchanged -- `transformers` has supported
NLLB tokenizers since ~4.28, so this pin did not need to move.
`deep-translator` stays, only because `preprocessing.py` still imports
`GoogleTranslator` at module level for its now-unused legacy methods;
it can be removed once those methods are deleted. `sentencepiece==0.1.99`
was added explicitly in this round -- NLLB's tokenizer is
SentencePiece-based, and the whole active Step 2 path now depends on it
directly rather than relying on it arriving as a transitive dependency.
