# Step 2 — frozen NLLB redesign (rev. 12 -- see "Round 7" through "Round 9" below for the source-side sentence redesign, incremental progress reporting, and this round's frozen-file/quality-gating/memory-safety hardening)

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
   `nllb_generation_version`) alongside them. **`nllb_generation_version`
   here is, and remains as of Round 12, the single corpus-wide PRIMARY
   `GENERATION_VERSION` only** (`translator.model_info()` is a fixed
   property of the translator, not computed per sentence) -- it does
   **not** flip to `RETRY_GENERATION_VERSION` for a corpus that contains
   one or more successfully-repaired sentences. This sidecar field exists
   to gate corpus freshness (see `_step2_bridge_is_fresh()` below), not to
   carry exhaustive generation provenance, so this is not a bug; the
   ambiguity that would otherwise create -- "was this whole corpus really
   generated under exactly the version this sidecar names?" -- is what
   Round 12's per-sentence `sentence_quality.repetition_fallback_retry.
   selected_generation_version` field (in `preprocessed_sentences_v2.json`,
   not in this sidecar) actually resolves, sentence by sentence.

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

**As of Round 11**, only ONE of these (degenerate-output) can gate
`STEP2_VALID`, and only for **sufficiently long** text -- see the validity
model below. The target-language check, length diagnostics, and the two
similarity-based diagnostics never gate on their own; target-language
mismatch gated `STEP2_VALID` (for sufficiently long text) from the second
external review through Round 10, but was demoted to diagnostic/REVIEW-
only at every length in Round 11, once real-corpus evidence showed it
never caught a genuine failure that degenerate-output detection didn't
already independently catch. They are:

- **Target-language check** (`check_target_language`): langdetect on the
  *output* of translation. Was intended to catch the most catastrophic
  local-model failure mode -- echoing the source language back, or
  drifting into a third language -- but on this corpus's real informal,
  often-short transcribed speech, `langdetect` itself turned out to be
  the dominant source of false positives (Round 10/11), so its result is
  now diagnostic/REVIEW-only, never fatal, at every length. **Length-
  gated** (added after the third external review, unchanged by the
  fatality change): text below `_is_sufficiently_long` (the same
  `MIN_CHARS_FOR_DIRECT_DETECTION` / `MIN_WORDS_FOR_DIRECT_DETECTION`
  threshold this module already used for response-level source-language
  detection) is **not assessed at all** -- `langdetect` on a single word or
  short exclamation ("Sí.", "No.", "PC.") is unreliable-to-meaningless, and
  this corpus has 46 real responses of 3 words or fewer. Returns
  `passed: None`, `status: "NOT_ASSESSED_SHORT_TEXT"` rather than guessing;
  `passed` is only ever `True`/`False` for text long enough to judge.
  Reported at both response grain (`language_mismatch_count`/`_ids`) and,
  since Round 11, sentence grain (`sentence_language_mismatch_count`/
  `_ids`).
- **Degenerate-output check** (`check_degenerate_output`): flags empty
  output (any length, always fatal), n-gram repetition loops (any length,
  always fatal; a 3-gram repeated 6+ times is flagged -- raised from 4 in
  Round 11), and output identical to a (different-language) source. **As
  of Round 12**, a sentence whose primary translation this check flags for
  repetition gets one automatic fallback retry under different decoding
  settings (see Round 12's own section below); if the retry's own result
  from this same function comes back clean, that clean result -- not the
  original flagged one -- is what's recorded here and what `flagged`
  reflects for that sentence. **The
  identical-to-source check is length-gated** (added after the third
  external review): still computed and reported (`is_identical_to_
  source`) at every length, but only counted as fatal (`is_identical_to_
  source_fatal`, and therefore `flagged`) when the text is long enough
  that identity is genuinely suspicious -- short Catalan/Spanish text (a
  single word, acronym, number, or proper noun) is frequently and
  CORRECTLY identical across both languages ("Sí." -> "Sí."), and treating
  every such case as degenerate would invalidate correct translations.
  **As of Round 11, this is the ONLY sentence-level FATAL quality
  signal** -- see the validity model below.
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
- `translation_output_validity` (added in the second review round; the
  definition below is the **current, Round 11** one -- see Round 9 and
  Round 11's own sections for how it got here): `sentence_flagged_count
  == 0`, where a sentence is `FLAGGED` exactly when its own
  `check_degenerate_output()` result fires (empty output, a long output
  identical to its source, or n-gram repetition) -- computed once per
  sentence, at the same grain as the frozen sentence IDs themselves (Round
  9). `check_target_language()`'s result does **not** contribute to this
  condition, at any length, as of Round 11 -- see "Translation-quality
  diagnostics" above for why. Also fatal in the sense that a corpus with
  even one such sentence is not usable input to Stage 3/5, whatever the
  reason.

`translation_sanity_status` has **three** states, matching which tier (if
any) actually failed:

- `"FAIL"` -- `translation_output_validity` is False: at least one
  sentence's own output is genuinely degenerate (empty, identical to a
  long source, or a repetition/pathological-collapse loop). This blocks
  `STEP2_VALID`.
- `"REVIEW"` -- `translation_output_validity` is True, but a **purely
  diagnostic, non-gating** signal fired: a length-ratio outlier (response-
  or sentence-grain), a semantic-similarity score under the informational
  bound, a short output not assessed for target language, a short output
  identical to its source, or (any length, as of Round 11) a target-
  language mismatch (response- or sentence-grain). Interview responses
  genuinely vary in how much Catalan/Spanish phrasing compresses or
  expands, similarity scores from a general-purpose embedding model are a
  noisy proxy not ground truth, and a full manual review of the first real
  corpus run's language-mismatch flags found `langdetect` itself -- not
  the translations -- was the dominant source of false positives at every
  length tried, not just short text. Gating `STEP2_VALID` on any of these
  would produce false negatives that block an otherwise-good run over
  unusually short, phrased, or (per `langdetect`) misjudged but correctly
  translated responses. Worth a human glance; never a reason to invalidate
  the corpus on its own.
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

**Note on the filename below, caught during Round 12's external review:**
this section predates the Round 7 redesign and was never updated when
Round 7 split translation caching onto its own new path. The file the
real production run (`main()` / `python preprocess_v2.py`) actually reads
and writes today is `data/cache/nllb_sentence_translation_cache_v1.json`
(`SENTENCE_CACHE_PATH`) -- see Round 7's own section, point 7 ("Separate
cache, old evidence untouched"), for exactly why that split happened. The
plain `data/cache/nllb_translation_cache_v1.json` (`CACHE_PATH`) named
throughout the rest of this section is the PRE-Round-7 whole-response/
chunk-level cache -- it still exists as a constant in the code and is
never read or written by the current sentence-level pipeline; it is kept
only so the six-hour evidence run that surfaced the target-side-splitting
problem (`evidence/round7_full_corpus_run_2026-09-07/`) stays
byte-for-byte auditable. Everything else below this note -- the cache-key
design, the Round 12 two-namespace addition -- applies identically to
`SENTENCE_CACHE_PATH`; only the filename in the surrounding prose is
stale. Restoring a real corpus's existing warm cache (see "Execution
sequence for the next real run" below) means restoring THIS file,
`nllb_sentence_translation_cache_v1.json`, not the older one.

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

**As of Round 12, this one file holds two independent, coexisting
generation-version namespaces, not one.** Every primary translation is
still keyed under `GENERATION_VERSION`, exactly as above. A sentence whose
primary translation is confirmed as a repetition loop additionally gets
one fallback-retry entry keyed under `RETRY_GENERATION_VERSION` (a
different string, so it can never collide with or be served as a primary
result) -- see Round 12's own section above for the mechanism
(`attempt_repetition_fallback_retry()`). Both namespaces are read and
written through the exact same `TranslationCache` instance, `get()`/
`set()` calls, and `save()` (atomic write, unchanged) -- there is no
second cache file and no special-cased persistence path for retries. One
consequence worth knowing when reading a run's printed cache statistics:
`TranslationCache.get()` increments the same `hits_this_run`/
`misses_this_run` counters regardless of which namespace it was called
for, so the corpus-wide `cache_stats` in a validation report is a
**blended total across primary lookups and any retry lookups**, not
primary-only. This is intentional (both are genuinely cache activity) but
means `cache_stats` alone cannot tell you how many of those hits/misses
were retries -- for that, read the `repetition_fallback_retry_*` fields
in `sentence_translation_summary` (Round 12), which report retry activity
on its own, separately from the blended total.

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

## Round 7 -- source-side sentence segmentation + one-sentence-per-translation

The sixth-review architecture above translated at the whole-response
(chunk) level and then re-split the *Spanish output* into sentences for
downstream analysis (`preprocessed_sentences_v2.json`). A real full-corpus
run against this architecture (kept, unmodified, in
`evidence/round7_full_corpus_run_2026-09-07/` for exactly this reason)
came back `STEP2_VALID = False` -- 8 language mismatches, 25
degenerate-output flags -- and manual inspection of the flagged cases
showed the deeper problem wasn't just those 33 flags: translating at the
chunk/response level and then guessing sentence boundaries afterward, in
the *translated* text, let NLLB merge or omit content across what should
have been a sentence boundary, with no way to attribute a downstream
sentence back to a single, specific source sentence. Re-splitting Spanish
output can never fix that -- the information about where one source
sentence ended and the next began was already lost by the time NLLB saw
the text.

The fix moves sentence segmentation to the *source* side, before any
translation happens, and makes translation operate one frozen source
sentence at a time:

1. **`segment_source_sentences(text, lang, preprocessor)`** --
   `split_sentences_strict()` (spaCy sentence boundaries) followed by
   `merge_ellipsis_continuations()`, a narrow, iterative repair for one
   confirmed artifact: a speaker trailing off mid-clause ("...") that
   spaCy treats as a sentence boundary, with the lowercase continuation
   split off as its own fragment. Audited against the real corpus
   (`segmentation_audit.py` / `segmentation_audit_v2.py` in the evidence
   folder): 6,434 raw source-side sentences, 38 genuine ellipsis-
   continuation cases (all read in full, including the two multi-fragment
   chains; zero false merges across an actual speaker-turn change), 6,396
   sentences and zero remaining ellipsis-continuation flags after the
   merge.
2. **Frozen sentence IDs.** `freeze_sentence_segmentation.py` runs
   `segment_source_sentences()` once across all 950 real responses and
   writes `data/frozen_source_sentence_segmentation_v1.json`: 950
   responses, 6,396 sentences, `response_id::sNNN` IDs, 0 duplicate IDs,
   38 merges (exactly matching the audit), and 0 reconstruction
   mismatches (every response's frozen sentences, rejoined, reproduce the
   same lexical content as the original raw text -- checked
   programmatically). `response_id::sNNN` is now the permanent analytical
   sentence ID; re-running the freeze against a changed segmentation
   function produces a new file, it never silently overwrites this one's
   meaning.
3. **One frozen sentence = one translation unit.**
   `translate_frozen_sentences_batched()` replaces `translate_responses_
   batched()` as the primary ca->es path (that function and
   `_split_into_translation_chunks()` are kept, unchanged, only for the
   secondary es->ca round-trip diagnostic sample). Spanish source
   sentences are copied unchanged (`SOURCE_ES`, never sent to NLLB).
   Catalan source sentences are batched across the *entire corpus* (not
   per-response) through `translate_texts_batched()` -- the same
   generation-parameters-in-cache-key, consecutive-batch-failure-abort
   machinery as before, just at sentence instead of chunk granularity.
   Failed sentences get one automatic retry pass in a fresh batch (a
   retry-pass failure is logged and left `FAILED`, never re-raised as an
   abort). Status per sentence: `SOURCE_ES / CACHE_HIT / FRESH_OK /
   RETRY_OK / FLAGGED / FAILED` -- `FLAGGED` overlays a successful
   translation whose sentence-level `check_target_language` /
   `check_degenerate_output` diagnostics fired, without discarding the
   translation itself; the mechanical outcome (did NLLB produce usable
   text, and from where) is tracked separately as `translation_
   provenance` so a flagged-but-successful sentence is never miscounted
   as a failure.
4. **Per-sentence atomicity.** A failed sentence no longer erases its
   whole response. `text_es` is rebuilt only from the ordered, successful
   sentence translations (`" ".join(...)`, in `sentence_index` order);
   `text_es_status` becomes an aggregate of the response's own sentences:
   `IDENTITY_COPY` (all-Spanish), `CACHE_HIT` (100% cache), `TRANSLATED`
   (a normal mix of fresh/retried/cached), `PARTIAL_TRANSLATION_FAILURE`
   (some but not all sentences failed -- the response still carries real,
   usable partial text_es and still gets response-level quality
   diagnostics run against it), or `WARNING_TRANSLATION_FAILED` (every
   sentence failed, or there were no sentences at all).
5. **No target-side re-splitting.** The old `split_sentences_strict(text_es, "es")`
   pass over translated output is gone entirely.
   `sentences_out` is now built directly and only from the frozen
   sentence structure -- one record per frozen `sentence_id`, carrying
   that ID, its `text_source`, its `text_es`, `sentence_status`,
   `translation_provenance`, and the frozen `response_id`/`interview_id`/
   `question_id`/`sentence_index` it belongs to. Downstream sentence-level
   analyses read these frozen IDs directly instead of re-deriving
   sentence boundaries from Spanish text.
6. **New validation invariant, strictly stronger than a sentence count.**
   `report["sentence_count_invariant"] = {"expected", "actual", "match"}`
   compares the number of frozen sentences fed into a run against the
   number of sentence records `process()` actually produced, folded into
   `structural_validity`/`silent_loss`. This is deliberately not "did we
   get 6,396 Spanish sentences by splitting Spanish" (a count that could
   coincidentally match while individual sentences drifted from their
   source) -- every downstream record also carries its own frozen
   `sentence_id`, so the real check is exact ID-set equality between the
   frozen file and `sentences_out`, not just equal cardinality. Verified
   end-to-end (real 950-response corpus, real frozen file, deterministic
   stub translator so it runs in seconds): `sentence_count_invariant ==
   {"expected": 6396, "actual": 6396, "match": True}`, frozen ID set ==
   downstream ID set exactly, 0 fatal errors, `structural_validity ==
   True`, 950/950 responses produced.
7. **Separate cache, old evidence untouched.** Translation now caches
   under a new path, `data/cache/nllb_sentence_translation_cache_v1.json`
   (`SENTENCE_CACHE_PATH`) -- distinct from the pre-Round-7
   `data/cache/nllb_translation_cache_v1.json` (`CACHE_PATH`, kept, never
   read or written by this path). Nothing under `evidence/round7_full_
   corpus_run_2026-09-07/` was touched: that six-hour run is what
   surfaced the target-side-splitting problem in the first place, and
   stays on disk, byte-for-byte as it ran, as the auditable record of why
   this redesign happened.
8. **`main()` requires the frozen file.** The `python preprocess_v2.py`
   CLI entry point now hard-fails (before doing any work) if
   `data/frozen_source_sentence_segmentation_v1.json` is missing, loads
   it, and passes it into `process(frozen_segmentation=...)` as the
   authoritative sentence structure for the full-corpus run -- a
   response missing from the frozen file, or whose frozen
   `source_language` no longer matches what's freshly detected for it, is
   a FATAL error, not a silent fallback. `terminal.py`'s interactive
   `_create_processed_versions()` deliberately does **not** auto-load the
   frozen file: it can legitimately run against a filtered subset or
   synthetic data (`_select_questions()`, or a caller's own test
   fixtures) that would never match a frozen file keyed to the full
   950-response corpus, so it computes segmentation live via the exact
   same `segment_source_sentences()` function the frozen file itself was
   built from -- byte-identical whenever the input actually is the full
   corpus (0 reconstruction mismatches, confirmed above).

`test_preprocess_v2_reliability.py` was re-run in full against this
redesign (**265/265 checks passing** -- superseding the "232 checks"
count in the Tests section below, which predates Round 7). A handful of
existing tests encoded the retired chunk-level architecture directly
(`response["translation_chunk_count"]`, an exact-string identity-copy
comparison, and two stub corpora that relied on the old naive test
splitter's period-eating quirk producing exactly one sentence) and were
updated to assert the equivalent invariant under the new architecture
rather than weakened; none of the updates touch `preprocess_v2.py`'s
actual translation, validation, or status logic.

## Round 8 -- incremental NLLB progress reporting

A real full-corpus run under the Round 7 architecture reported `0/5792`
in its progress bar for a long time while real translation work was
demonstrably happening underneath it (cache growing, CPU/GPU active),
then jumped straight to 100% once the whole run finished. On an hours-long
run this makes it impossible to tell a genuinely stuck process from a
slow-but-progressing one -- exactly the ambiguity the fifth review's
progress bar was built to remove in the first place, just reintroduced one
level down by the Round 7 redesign.

**Root cause.** `translate_frozen_sentences_batched()`'s first pass called
`translate_texts_batched()` once for the *entire* corpus-wide list of
cache-miss sentences, and only iterated the per-sentence progress-tick
loop *after* that whole call returned. `translate_texts_batched()` itself
does accept a `progress_callback`, but Round 7's caller never wired
anything into it -- so every sentence's tick was deferred to one bulk loop
at the very end, correct in its final count but with nothing visible while
it ran.

**Fix**, keeping the sentence-level translation architecture and the
frozen 6,396-sentence structure completely unchanged -- this was a
reporting-plumbing fix, not a translation-logic change:

- Progress signaling and result-data assembly are split into two separate
  concerns (`_tick_progress` vs. `_record`), so a tick can fire the moment
  a sentence's outcome is known, independent of when its full result
  record gets built.
- Per-sentence ticking is idempotent -- each sentence_id is guaranteed to
  advance the progress count exactly once, however it resolves (cache hit,
  first-pass success, retry-pass success, or permanent failure).
- `_on_first_pass_piece_done` / `_on_retry_piece_done` callbacks are wired
  into `translate_texts_batched`'s existing `progress_callback` parameter
  (already there, just previously unused by this caller), so both the
  first pass and the retry pass now tick live, per completed batch, as
  translation actually happens.
- An end-of-function reconciliation loop guarantees every sentence gets
  ticked exactly once even if the retry pass itself aborts internally
  (e.g. via the existing consecutive-batch-failure guard raising
  `NLLBTranslationError` mid-retry) -- a retried sentence that never got a
  chance to succeed or fail explicitly still ends up counted as `FAILED`,
  never silently dropped from the progress total.
- Cache hits now advance progress immediately (previously they were also
  deferred to the end-of-call bulk loop); fresh translations advance
  per-batch, not per-whole-corpus-call; retries are counted and reported
  separately (`retried_ok`) from first-pass successes (`fresh`).

**New tests (44-50 -- Round 7's own regression suite already reached
Test 46; Tests 47-50 are new in this round):**
Test 47 proves progress ticks are genuinely incremental during a live run
(one `progress_hook` call per sentence, not one bulk call at the end),
that the first tick fires with a partial count, that a cache hit produces
the very first tick, and that final tallies (`cache_hits`/`fresh`/
`retried_ok`/`failed`) are correct for a mixed cache-hit/fresh/retry-
recovered/Spanish-passthrough scenario. Test 48 proves a retry pass that
fails permanently still reaches the correct final count, with every
affected sentence correctly `FAILED`. Test 49 proves a retry pass that
itself trips the consecutive-batch-failure abort recovers cleanly --
the run completes without raising, every sentence (including one never
actually attempted before the internal abort) is still accounted for
exactly once, and no sentence is ever ticked twice. Test 50 re-runs
`translate_frozen_sentences_batched()` directly against the real, on-disk
6,396-sentence frozen file with a fast deterministic stand-in translator,
confirming the fix holds at real corpus scale: exactly one record per
frozen sentence ID, progress reaching exactly the true Catalan-sentence
total, and thousands of separate incremental ticks (not one).

`test_preprocess_v2_reliability.py`: **295/295 checks passing** after this
round -- superseding the "265/265" count above. Nothing about the
translation architecture, the frozen sentence structure, cache resumption,
or `STEP2_VALID` gating changed; this round is entirely about making
in-progress work visible while it happens.

## Round 9 -- exact frozen-file/ID-set validation, sentence-level quality gating, and memory-safety fixes

A further review of the Round 8 delivery -- reading `preprocess_v2.py` in
full against the frozen-file/validity/memory-load machinery specifically
-- found four issues considered necessary before the next real corpus run,
plus two further hardening fixes prompted by this project's own prior
Mac memory-crash history (see "Fixes from the fifth external review"
above). All six are implemented in this round, plus three points that
were confirmed already accurately documented and needed no code change.

1. **Exact frozen-file <-> input validation (new hard pre-run gate).**
   `main()`'s per-response frozen lookup (`missing_from_frozen_
   segmentation` / `frozen_segmentation_source_language_mismatch`) only
   ever iterates the *current* corpus's responses, so it can never notice
   a response that quietly disappeared from the frozen file entirely, or
   one whose recorded text silently drifted while its ID and detected
   language happened to stay the same -- the frozen file could go stale
   for the corpus as a whole without any single per-response check ever
   firing. **Fixed** with a new hard gate, checked before any model load:
   - `_hash_file(path)` (mirrors `terminal.py`'s existing `_hash_file` --
     SHA-256, 1MB chunks, raw bytes -- duplicated into `preprocess_v2.py`
     since `terminal.py` imports `preprocess_v2`, not the reverse).
   - `FROZEN_SEGMENTATION_SCHEMA_VERSION = 2`. `freeze_sentence_
     segmentation.py` now writes `schema_version` and `input_sha256`
     (the current corpus file's hash at freeze time) as new top-level
     fields; the `responses` content itself, and every existing top-level
     field, is completely unchanged -- confirmed by regenerating the real
     6,396-sentence frozen file and diffing it against the pre-Round-9
     version: `responses`, `ellipsis_continuation_merges`,
     `response_count`, `total_sentence_count`, `generated_from`, and
     `segmentation_function` are all byte-for-byte identical; only
     `schema_version` (1 -> 2) and the new `input_sha256` field differ.
   - `validate_frozen_segmentation_against_input(frozen_file, raw_data,
     input_file_path)`: five checks, in order -- schema currency (an
     older file predates `input_sha256` and can't be verified at all, so
     this check short-circuits everything else), the frozen file's own
     declared `response_count`/`total_sentence_count` against what's
     actually inside it (self-consistency), `input_sha256` recomputed
     from the current on-disk input file against what the frozen file
     recorded, response-ID-set equality in *both* directions (a current
     response missing from the frozen file, and a frozen response no
     longer in the current corpus -- the latter being exactly the case
     the per-response lookup can never catch by itself), and per-response
     `original_text` equality for every ID present in both. Returns a
     list of human-readable problems (never raises); `main()` now loads
     the frozen file and calls this **before** `run_preflight()` (i.e.
     before any model load), logs every problem, and refuses to run if
     the list is non-empty.
2. **Exact sentence-ID-set validation in `sentence_count_invariant`, not
   just cardinality.** The existing invariant only ever compared
   `expected_sentence_count == actual` -- two numbers matching by
   coincidence would still pass even if, say, one frozen sentence ID were
   silently dropped from `sentences_out` while a duplicate of another one
   were produced instead. **Fixed**: `sentence_count_invariant` now also
   carries `sentence_id_set_match`, `missing_sentence_id_count`/
   `missing_sentence_ids` (frozen, never produced), and
   `unexpected_sentence_id_count`/`unexpected_sentence_ids` (produced,
   not frozen) -- `match` now requires *both* the count and the exact set
   to agree, and is folded into the existing `silent_loss`/
   `structural_validity` computation exactly as the count-only version
   was. `print_validation_report()` prints the new fields alongside the
   existing count line. Test 52 proves the two checks are genuinely
   complementary, not redundant: a corrupted frozen file with a duplicate
   sentence ID within one response produces a case where the *set* of IDs
   still matches (a duplicate collapses to an equal set) but the
   *cardinality* does not -- proving `sentence_id_set_match` alone would
   have missed a real problem, and that the combined `match` check
   catches it correctly (`structural_validity=False`,
   `STEP2_VALID=False`).
3. **Sentence-level quality gating, replacing the response-level-only
   gate.** `translation_output_validity` was still computed from
   *response-level* diagnostics run on the reconstructed (joined)
   `text_es` -- even though Round 7 moved translation itself to
   sentence-level granularity, the validity gate never followed. This
   reopens exactly the false-positive failure mode the third and sixth
   reviews already fixed once, at a different grain: several
   individually-fine sentences, each too short to trip its own
   per-sentence repetition threshold, can still concatenate into a
   RECONSTRUCTED response whose full text happens to repeat a short
   phrase often enough to trip the *response-level* repetition check --
   even though nothing was actually wrong with any single sentence.
   **Fixed**:
   ```python
   translation_output_validity = sentence_flagged_count == 0
   ```
   `sentence_flagged_count` is `sentence_translation_summary`'s corpus-wide
   count of sentences whose *own* `sentence_status == "FLAGGED"` (set in
   Pass 2, from that sentence's own `check_target_language`/
   `check_degenerate_output` results -- unchanged from Round 7). The
   response-level `language_mismatch_count`/`degenerate_output_count`
   (still computed, in Pass 3, on the reconstructed text -- unchanged)
   are demoted to the same REVIEW-only tier as length-ratio outliers and
   low similarity: they push `translation_sanity_status` to `"REVIEW"`
   but never gate `STEP2_VALID` on their own anymore.
   `print_validation_report()` now prints "FLAGGED sentences (fatal --
   gates STEP2_VALID)" as its own line, with the response-level counts
   relabeled "Response-level diagnostics (review-only)".
   Test 53 is the direct regression test: four distinct, individually
   unflagged sentences that each translate to the same short phrase
   trip the response-level repetition check on the reconstruction
   (`degenerate_output_count == 1`) but leave `translation_output_
   validity=True`, `translation_sanity_status="REVIEW"`, and
   `STEP2_VALID=True` -- proving this specific false-positive shape no
   longer fails the run. Test 54 is the mirror case, proving the fix
   didn't overcorrect into a majority-vote or an average: one genuinely
   bad sentence (wrong-language output), sitting between two good ones in
   the very same response, still correctly fails `translation_output_
   validity` and `STEP2_VALID` on its own -- not diluted or averaged away
   by its good neighbors.
4. **Per-sentence length diagnostics (diagnostic-only, new in the
   sentence-level report).** A response-level length ratio can
   statistically absorb one sentence collapsing to a fraction of its
   source length while the rest of the response compensates -- an 8-word
   source sentence translated to one word would barely move a whole
   response's aggregate ratio. **Fixed**: `compute_length_diagnostics()`
   (already existed, response-level) is now also run per-sentence in Pass
   2, stored at `sentences_out[...]["quality"]["length_diagnostics"]`, and
   rolled up corpus-wide as `sentence_translation_summary["sentence_
   length_ratio_outlier_count"]`/`"_ids"`. Deliberately **not** part of
   the `FLAGGED`/fatal-quality condition -- source<->target length
   legitimately varies a lot sentence-to-sentence (a short
   acknowledgement, an elided clause), so an arbitrary per-sentence
   threshold would reproduce the same kind of false positive the second
   and third reviews already found and fixed for length ratios and short
   text. Test 55 confirms a drastically-shortened sentence is correctly
   identified as an outlier by ID, that a normally-proportioned neighbor
   sentence in the same response is not, and that the outlier sentence's
   own `sentence_status` stays `FRESH_OK` (not `FLAGGED`) -- diagnostic
   only, surfaced via `REVIEW`, never gating.
5. **A single NLLB model load, not two.** `run_preflight()` already
   constructs a real `NLLBTranslator` (a full ~600M-parameter model load,
   confirmed by an actual translation call) purely to verify the
   environment; `main()` used to throw that one away and construct a
   SECOND `NLLBTranslator` immediately afterward for the actual corpus
   run -- loading the model twice, back-to-back, before doing any real
   work, on the exact low-memory Mac that had already crashed once under
   memory pressure (see "Fixes from the fifth external review"). **Fixed**:
   `run_preflight()` now stores the constructed translator at
   `result["translator"]`, but only on the success path (an exception
   during construction never sets it, so a caller can never mistakenly
   reuse a translator from a failed or partial preflight); `main()` now
   does `translator = preflight_result["translator"]` instead of
   constructing a new one. `run_smoke_test()` (a separate, independent
   `--smoke-test` mode with its own tiny synthetic corpus, never invoked
   together with the real run in the same process) still constructs its
   own translator -- that is a different mode entirely, not part of this
   fix. Test 56 confirms `run_preflight()` returns the translator on
   success and omits it on failure, and statically confirms (via
   `inspect.getsource`) that `main()`'s own source never constructs an
   `NLLBTranslator` directly and does reuse `preflight_result["translator"]`
   -- a regression guard against a future refactor silently reintroducing
   the double load.
6. **`SemanticSimilarityScorer` made genuinely lazy.** Its own docstring
   already claimed to load "lazily and only if actually needed," but
   `__init__` unconditionally constructed the real `SentenceTransformer`
   regardless -- so every run paid the embedding-model load cost even
   when the corpus had nothing for it to score (e.g. an all-Spanish-
   original corpus needs no Catalan<->Spanish similarity at all), again a
   real concern given this project's prior Mac memory-crash history.
   **Fixed**: `__init__` now does nothing but set attributes.
   `_ensure_loaded()` performs the (at-most-once) real construction
   attempt, called from the `available` property -- which remains the
   correct trigger at its existing real-use call sites (`similarity_pairs
   and similarity_scorer.available`, short-circuited so an empty-pairs run
   never reaches it) -- and a new `load_attempted` property lets other
   code inspect whether a load was ever tried *without* forcing one.
   This mattered concretely at one existing call site: `process()`'s
   end-of-run `translation_quality_summary` construction used to read
   `similarity_scorer.available` unconditionally to fill in a summary
   field, which would have silently forced a load on every single run
   just to populate a report field, defeating the laziness fix for
   exactly the runs it matters most for. **Also fixed**: that call site
   now reads `load_attempted` first, and reports a
   `"not_needed_for_this_run_no_similarity_pairs_computed"` reason
   (distinct from `"similarity_scorer_not_provided"` and from a real load
   failure's actual exception message) when no load was ever attempted.
   Test 57 proves construction alone attempts nothing (no network/model
   access even attempted), that the first real use triggers exactly one
   load attempt (verified via a fake `sentence_transformers` module
   injected into `sys.modules`, so this is deterministic and needs no
   network either way), that later calls do not load again, that a load
   failure degrades gracefully without raising, and -- the integration
   case that matters most -- that a full `process()` run against an
   all-Spanish corpus never forces the load at all and reports the
   correct non-misleading reason.

**Three points confirmed already accurately documented, no code change
needed:**
- **Code-switching is still response-level routing, not sentence-level.**
  Round 7 moved *translation* to per-sentence granularity, but *language
  detection* is still done once per response
  (`determine_response_language(raw_text, primary_lang)`), and every
  sentence in that response inherits the single resulting
  `source_language` -- there is no per-sentence language re-detection.
  A response that code-switches between Catalan and Spanish sentence-by-
  sentence is therefore still translated (or identity-copied) as if it
  were entirely one language. This is an accurate, pre-existing
  limitation, not newly introduced this round, and is recorded here
  explicitly so it is never mistaken for something Round 7's per-sentence
  redesign already fixed.
- **Question-text wording is not in the frozen records** -- already
  stated precisely in "What context Step 2 actually preserves" above;
  unchanged this round.
- **The sentiment/confusion signal remains legacy BART, pending the Stage
  5 mDeBERTa replacement** -- already stated in "Sentiment / confusion
  analysis stay out of Step 2" and the third-review section above;
  unchanged this round, and out of scope for Step 2 either way.

**New tests (51-57):** see each fix above for what its own test proves.
`test_preprocess_v2_reliability.py`: **360/360 checks passing** after this
round -- superseding the "295/295" count above. The frozen 6,396-sentence
structure, the one-sentence-per-translation architecture, and every prior
round's fixes are all unchanged and re-verified; this round adds a
pre-run freshness gate, a stronger structural invariant, a corrected
quality gate, a new diagnostic, and two memory-safety fixes, none of
which touch what gets translated or how.

## Round 10 -- diagnosis of the first real 950-response corpus run, anti-repetition generation parameters, and a raised short-text language-detection threshold

The Round 9 delivery was run for real, for the first time, on the full
950-response corpus (`python preprocess_v2.py --batch-size 1`, ~10.5
hours on the user's M1 Mac). It completed without crashing or losing any
data -- `structural_validity=True`, `translation_completeness_
validity=True` -- but `translation_output_validity=False`
(`translation_sanity_status="FAIL"`), so `STEP2_VALID=False`:
`sentence_translation_summary` named 146 flagged sentence IDs. The user
supplied the real `preprocessed_responses_v2.json`,
`preprocessed_sentences_v2.json`, and `preprocessing_language_report.json`
from that run; every one of the 146 flagged sentences was individually
diagnosed against those files (not sampled -- every flagged ID was pulled
up and read) and fell into exactly three buckets:

1. **36 sentences (25%): genuine NLLB decoder repetition-loop failures.**
   Short, low-information, often-repetitive Catalan source utterances
   ("No, no.", "- Vint, vint.", "Força.") caused beam search to loop,
   generating the same short phrase 70-93+ times instead of stopping
   (confirmed via each sentence's `degenerate_output.repetition_detail`:
   dominant repeated n-grams like "no, no, no," at counts 74-92, "veinte,
   veinte, veinte," at count 15, and one outright hallucination,
   "tortilla de tortilla", at count 87). All 36 also tripped the
   language-mismatch check, because the resulting garbage doesn't read as
   Spanish to `langdetect` either -- a symptom of the same underlying
   failure, not a second, independent bug. Only 122 distinct source texts
   sit behind the 146 flagged IDs: identical source sentences across
   different responses share one cached translation
   (`translation_cache.py`), so a single bad cached output can surface as
   several flagged sentence IDs at once -- "No, no." alone accounts for
   14 of the 146.
2. **105 sentences (72%): `langdetect` false positives on correct
   Spanish.** Every one of these 105 was manually read in full and
   confirmed to be a fluent, grammatically correct Spanish translation of
   short, informal transcribed speech (contractions, dialogue dashes,
   interjections like "eh"/"Ostia", occasional code-switched loanwords).
   `langdetect`'s statistical char-n-gram model misjudges genuinely
   correct Spanish at these lengths -- misdetected mostly as Portuguese
   (107 of 141 total mismatch flags), but also Catalan, Italian, French,
   English, Somali, Tagalog, and German, often at reported confidence
   above 0.99. `char_ratio` (translated length vs. source length) for
   these 105 ranges 0.71-1.58 with zero outliers, confirming they are not
   distorted or truncated translations -- just wrongly flagged as the
   wrong language.
3. **~4-5 sentences (3%, not separately remediated this round): borderline
   repetition-threshold false positives.** A handful of long, otherwise
   correctly-translated responses contain a short, natural phrase ("el
   plan de", "el tema de", "que hay que", "se gasta el") that legitimately
   repeats exactly `REPETITION_MIN_REPEATS` (4) times in normal rambling
   speech, coincidentally meeting `check_degenerate_output()`'s repetition
   threshold. This is a distinct, much smaller issue from buckets 1-2 --
   the user was not asked to choose a remediation for it this round, and
   neither generation-parameter fix below touches
   `REPETITION_NGRAM_SIZE`/`REPETITION_MIN_REPEATS` or the detection logic
   at all. **It is possible a small number of sentences in this bucket
   remain flagged after the next real run even with both fixes below
   applied** -- if so, that is expected, not a sign either fix failed, and
   is a natural follow-up decision once real post-fix numbers exist.

Presented to the user with full evidence and examples; the user chose the
remediation for buckets 1 and 2 explicitly (via two separate decisions,
not a single bundled one, since each touches different "frozen
methodology" -- decoding parameters vs. language-detection logic -- that
this project has been deliberately careful to keep stable and auditable
across every prior round):

**Fix for bucket 1 -- anti-repetition generation parameters (`GENERATION_PARAMS`, `preprocess_v2.py`).**
Two parameters added, passed straight through to `model.generate()`:
- `repetition_penalty = 1.3` -- a soft, proportional penalty applied at
  every decoding step to the score of any already-generated token. It
  discourages repetition without ever forbidding it, so it does not
  distort short, legitimately-repetitive translations ("No, no, no." ->
  "No, no, no.").
- `no_repeat_ngram_size = 4` -- a hard constraint: once any 4-token
  sequence has been generated, that exact sequence can never recur later
  in the same output. This makes the observed failure mode -- the same
  short phrase repeating dozens of times -- structurally impossible by
  construction, while a 4-gram floor (one token wider than the 3-gram
  `check_degenerate_output()` uses for *detection*) still leaves room for
  genuine short 3-word repeats that occur naturally in this corpus's
  rambling interview speech.
Chosen as a standard, moderate combination for this specific NMT
beam-search pathology, rather than a more aggressive setting (e.g.
`no_repeat_ngram_size=2` or `3`) that would force paraphrasing of ordinary
short emphatic repeats and risk degrading otherwise-correct translations.
**`GENERATION_VERSION` bumped** to
`"nllb600M-beams4-maxlen512-chunked-v2-antirep1"` -- per this project's
established cache-invalidation discipline (see the "Cache impact, and why
GENERATION_VERSION was bumped" note from the Round 7 write-up), any
`GENERATION_PARAMS` change requires this, so that no translation generated
under the old, non-anti-repetition settings can ever be silently served
after this fix. Concretely, this means **the entire ~5.8k-sentence
translation cache is invalidated, not just the 36 known-bad entries** --
every sentence's cached translation was generated without these
parameters, so its correctness under the new settings has never actually
been verified, and reusing it selectively would defeat the point of a
reproducible, auditable generation pipeline. **The next real run will
therefore re-translate the full corpus from scratch (~5 hours, based on
the prior run's timing), not resume from the existing cache.** This has
not been validated against the real corpus yet -- confirming it actually
resolves the 36 known repetition-loop cases (and doesn't introduce new
ones) is the point of that next run, not something this delivery can
verify itself, consistent with this project never running the real NLLB
corpus on its own.

**Fix for bucket 2 -- raised short-text language-detection threshold
(`MIN_CHARS_FOR_DIRECT_DETECTION` / `MIN_WORDS_FOR_DIRECT_DETECTION`,
`preprocess_v2.py`).** Raised from 30 chars / 5 words to **50 chars / 8
words**, chosen from a data-driven tradeoff analysis run directly against
the real corpus's 4,298 already-assessed sentences (4,157 correctly
passed, 36 confirmed true-positive repetition-loop cases from bucket 1,
105 false positives from bucket 2):

| MIN_CHARS / MIN_WORDS | false positives resolved | correctly-passed sentences newly exempted | true positives still caught |
|---|---|---|---|
| 40 / 7  | 51/105 (49%) | 536/4157 (12.9%)  | 36/36 |
| 45 / 7  | 66/105 (63%) | 731/4157 (17.6%)  | 36/36 |
| **50 / 8 (chosen)** | **78/105 (74%)** | **974/4157 (23.4%)** | **36/36** |
| 55 / 9  | 86/105 (82%) | 1171/4157 (28.2%) | 36/36 |
| 60 / 10 | 89/105 (85%) | 1365/4157 (32.8%) | 36/36 |

All five candidates caught 100% of the 36 confirmed true-positive
repetition-loop sentences -- those all have `target_chars` well over 300,
far above any threshold considered, so raising this threshold never risks
hiding a real repetition-loop failure. 50/8 was picked as the point past
which each further step buys progressively less false-positive resolution
for progressively more corpus-wide assessment coverage given up (e.g. the
55/9 -> 60/10 step buys only +3 points of resolution for +4.6 points of
newly-exempted coverage); it is a reasoned middle-ground choice from this
data, not the only defensible one -- if the next real run's post-fix
numbers suggest otherwise, this is easy to revisit with the same method.
Both bars must still be cleared together (`_is_sufficiently_long` is an
AND, unchanged) -- this only changes where the bars sit, not the
short-circuit logic, `NOT_ASSESSED_SHORT_TEXT` status, or the fact that a
short output's `passed` stays `None`, never a false `False`.

**New tests (58-59):** Test 58 proves the raised threshold using a real,
grammatically correct short Spanish phrase that clears the *old* 30/5
bars (so would have been assessed, and was exactly the shape of the real
false positives) but not the *new* 50/8 bars (so is now correctly
exempted, `NOT_ASSESSED_SHORT_TEXT`, `passed=None`) -- and separately
confirms a genuinely long wrong-language text is still caught and a
genuinely long correct translation still passes, so the raise provably
only removes false positives rather than blinding the check generally.
Test 59 proves `GENERATION_PARAMS` carries both new anti-repetition keys
with sane values (`repetition_penalty > 1.0`; `no_repeat_ngram_size`
strictly larger than `REPETITION_NGRAM_SIZE`, so generation-time blocking
can never be tighter than what detection itself tolerates), that
`GENERATION_VERSION` was actually bumped to reflect the change, and that
the pre-existing deterministic decoding settings (`num_beams=4`,
`do_sample=False`, `max_length=512`) were preserved, not replaced.
`test_preprocess_v2_reliability.py`: **377/377 checks passing** after this
round -- superseding the "360/360" count above.

**Not run this round, and not run by this delivery at all:** the real
NLLB corpus. Per this project's unbroken discipline across every prior
round, these fixes are delivered for the user to review and run
themselves, on their own machine, with network/model access this
environment doesn't have. The full re-run this round's `GENERATION_
VERSION` bump requires (~5 hours, no cache reuse) is the user's next step,
not something performed here.

## Round 11 -- demoting language-detection to non-fatal, a raised repetition threshold, and a documented terminology-audit gap

The user independently re-analyzed the same three real Round 10 output
files (`preprocessed_responses_v2.json`, `preprocessed_sentences_v2.json`,
`preprocessing_language_report.json`) in detail and produced their own
breakdown of the 146 flagged sentences, arriving at numbers consistent
with the Round 10 diagnosis (141 language mismatches, 41 repetition
flags, 36 overlapping both, 105 language-only, 5 repetition-only) but
going further in two ways that changed this round's design: a concrete
example of `langdetect`'s unreliability (`"- Bueno, tenemos dos
empresas."` detected as Portuguese at 0.999995 confidence), a precise
count of the 4 legitimate-repeated-phrase false positives inside the 41
repetition flags (`"se gasta el ..."`, `"el plan de ..."`, `"el tema de
..."`, `"que hay que ..."`, each repeating naturally exactly 4 times),
and a new finding outside the 146 flags entirely: domain-terminology
mistranslations inside sentences the automatic gate marks `FRESH_OK`
(Catalan `truges` -> `"truegos"`/`"trozos"` instead of *cerdas*, `pagesos`
-> `"paganos"` instead of *agricultores*/*ganaderos*, `engreix` ->
`"carne de cerdo"`/`"engrejos"`/`"greso"`, `granges de mare` -> `"granjas
de mamá"`). The user proposed a three-way remediation (language detector
never fatal; severe repetition still fatal but the 4-repeat false
positives fixed; terminology as a separate concern) and asked for
agreement before any further code change, given this touches the same
"frozen methodology" territory as Round 10 and partially reverses a
choice made there.

Four separate decisions were confirmed with the user (`AskUserQuestion`,
each independent) before implementing:

1. **Target-language mismatch is no longer part of any FATAL gate, at
   any length** (`check_target_language()` in `preprocess_v2.py`,
   `flagged` computation in `process()`'s sentence-quality Pass 2). Round
   10 had raised the short-text exemption threshold (`MIN_CHARS_FOR_
   DIRECT_DETECTION`/`MIN_WORDS_FOR_DIRECT_DETECTION`, still 50/8,
   unchanged this round) to resolve 74% of the known false positives
   while leaving mismatches above that bar fatal. This round goes
   further: a language mismatch is now **never** fatal, at any length.
   Justification is the data itself, not just precedent -- of the 141
   real language-mismatch flags from the Round 10 run, manual review
   found **zero** that were a real translation failure uniquely caught
   by language detection and not already independently caught by `check_
   degenerate_output()`; the 36 that were real failures were real
   because of repetition, and `check_degenerate_output()` flagged them on
   that basis regardless of what `check_target_language()` said. In other
   words, on this corpus, on this run, language-mismatch-alone had 0%
   precision as a fatal-error signal. `check_target_language()` is still
   fully computed and reported -- response-level (`language_mismatch_
   count`/`_ids` in `translation_quality_summary`, unchanged from prior
   rounds) and, new this round, sentence-level (`sentence_language_
   mismatch_count`/`_ids` in `sentence_translation_summary`, the direct
   per-sentence mirror of `sentence_length_ratio_outlier_count`/`_ids`) --
   and both now feed `translation_sanity_status="REVIEW"` the same way
   length-ratio outliers and low-similarity responses already did.
   `check_degenerate_output()` (empty output, a long output identical to
   its source, and n-gram repetition) is now the **sole** sentence-level
   FATAL signal.
2. **`REPETITION_MIN_REPEATS` raised from 4 to 6** (`preprocess_v2.py`).
   Directly fixes the 4 natural-repeated-phrase false positives the user
   found inside the 41 repetition flags -- a short, ordinary phrase
   recurring exactly 4 times in long, otherwise correctly-translated,
   rambling interview speech no longer trips the detector. 6 was chosen
   with real margin on both sides: every genuine NLLB repetition-loop
   failure in the Round 10 run repeated its worst 3-gram at least 15
   times (most 70-93+), so raising the bar to 6 cannot miss any of the 37
   confirmed real failures, while comfortably clearing the observed
   false-positive count of exactly 4. Note `check_degenerate_output()`'s
   outer length guard (`len(tokens) >= REPETITION_NGRAM_SIZE *
   REPETITION_MIN_REPEATS`) also moved with this change (3*6=18 words
   minimum before repetition is even checked, up from 3*4=12) -- Test 61
   proves the exact boundary (5 repeats: not flagged; 6 repeats: flagged)
   directly.
3. **The anti-repetition `GENERATION_PARAMS` fix and its full-corpus
   re-run stand as delivered in Round 10, unchanged.** The user's initial
   message this round proposed retranslating only the ~37-41 confirmed-
   bad sentence IDs under the new `repetition_penalty`/`no_repeat_ngram_
   size` settings and reusing the existing cache for everything else,
   to avoid the ~5-hour full re-run. Given the choice explicitly again
   this round (full re-run vs. a new targeted-retranslation mode with
   explicit per-sentence generation-provenance tracking so a mixed-
   version cache is never silently ambiguous), the user chose to keep
   the full re-run. This preserves the reproducibility property this
   project has protected since Round 7's original `GENERATION_VERSION`
   design ("Cache impact, and why GENERATION_VERSION was bumped"): every
   sentence in a delivered corpus was generated under one documented,
   fully-verified decoding configuration, never a silent mix of two.
4. **Domain-terminology mistranslations are documented as a known
   limitation, not addressed with new tooling this round.** The `truges`/
   `pagesos`/`engreix`/`granges de mare` examples above are real, and
   real automatic detection: high `translation_quality_summary.semantic_
   preservation` similarity does not catch a specific wrong-but-related-
   sounding noun substituted for a correct domain term (a general-purpose
   multilingual sentence embedding model has no notion of this corpus's
   Catalan pig-farming dialect vocabulary). This is **out of scope for
   Step 2's automatic `STEP2_VALID` gate** -- there is no principled,
   general way to auto-detect "this specific noun is domain-wrong" without
   either a curated glossary (which itself risks false alarms on
   legitimate variation) or a domain expert's read, and building one
   was explicitly declined this round in favor of documenting the gap
   here for manual/domain-expert review. If this becomes a priority, the
   `AskUserQuestion` alternative already scoped -- a separate, non-gating
   terminology-audit diagnostic checked against a small user-supplied
   glossary, never affecting `STEP2_VALID` -- remains available as a
   future addition; nothing in this round's code forecloses it.

**New tests (60-61):** Test 60 is the direct mirror of Test 54 (a bad
sentence among two good neighbors in the same response) with the opposite
expected outcome now that language mismatch is non-fatal: none of the
three sentences is `FLAGGED`, the mismatched one is still correctly named
in the new `sentence_language_mismatch_ids`, and `translation_output_
validity`/`STEP2_VALID` are both `True` with `translation_sanity_
status="REVIEW"`. Test 61 pins `REPETITION_MIN_REPEATS == 6` and proves
the exact boundary directly: a 3-gram repeating 5 times is not flagged,
the same 3-gram repeating 6 times is. **Existing tests updated in place**
for the new behavior rather than left to silently assert the old, now-
incorrect behavior: Test 33 (a language-mismatch-only failure) now
asserts `STEP2_VALID=True`/`REVIEW`, the mirror of what it asserted
through Round 10; Test 54 now uses a degenerate (repetitive) bad sentence
instead of a wrong-language one, since a wrong-language sentence can no
longer serve as "the one bad sentence" a sentence-level-fatality test
needs; Tests 11, 34, and 53 (which all depend on tripping the repetition
detector) were adjusted to repeat their test phrases enough times to
clear the new `REPETITION_MIN_REPEATS=6` bar with margin, since their
prior repeat counts (built around the old bar of 4) no longer trip
detection at all -- `STUB_REPETITIVE_OUTPUT` now repeats its phrase 9
times (was 7) and Test 53's response-level false-positive case now joins
6 sentences (was 4). `test_preprocess_v2_reliability.py`: **390/390
checks passing** after this round -- superseding the "377/377" count
above.

**Not run this round, and not run by this delivery at all:** the real
NLLB corpus, for the same reason as every prior round. None of this
round's changes require re-translation on their own -- `MIN_CHARS_FOR_
DIRECT_DETECTION`/`MIN_WORDS_FOR_DIRECT_DETECTION` are unchanged from
Round 10, and `REPETITION_MIN_REPEATS` and the language-mismatch-fatality
change are both pure post-hoc scoring/gating logic over already-generated
translations, not generation-time settings -- so a fresh run against an
UNCHANGED `GENERATION_VERSION` would hit the existing cache for the
entire already-translated corpus and only spend time on diagnostics/
report assembly, not on NLLB inference. The ~5-hour cost this round's
delivery still carries comes entirely from Round 10's `GENERATION_
VERSION` bump (the anti-repetition parameters), confirmed to stand as-is
in decision 3 above, not from anything new in Round 11.

## Round 12 -- a factual correction on Round 11's re-run decision, and a targeted always-on fallback-retry mechanism replacing the global anti-repetition parameter change

After Round 11 was delivered, the user reviewed it against a 15-point
checklist and reported that it contradicted "the plan we just agreed on"
-- specifically, that a targeted, provenance-aware repair mode had
already been agreed, and Round 11's ZIP (which kept Round 10's global
`GENERATION_PARAMS` change and its ~5-hour full re-run) broke that
agreement. This needed a factual check before any further code change:
the actual Round 11 `AskUserQuestion` record shows the retranslation-scope
question was answered **"Full corpus re-run (as already delivered)"**,
not a targeted-repair path -- see decision 3 in the Round 11 section
above. That correction was given to the user directly, framed as a
factual record check rather than a rebuttal, and the user did not dispute
it; the conversation moved on to jointly designing the targeted-repair
architecture as a **new** decision this round, not as compliance with a
prior one that was never actually made. Two stale "living" reference
sections elsewhere in this file (`## Translation-quality diagnostics` and
`## The three-tier validity model`, both still describing pre-Round-11
fatal-language-mismatch behavior) were also identified by the user as
contradicting Round 11's actual code and have been corrected in place --
see those sections below for their current, accurate text.

Two decisions were confirmed with the user (`AskUserQuestion`, each
independent) before implementing this round:

1. **The targeted-repair mechanism is built as an always-on part of the
   normal `process()` pipeline, not as a separate `--repair-existing`
   CLI flag.** The user's original 15-point plan specified an explicit
   repair-mode flag that would load a prior run's output JSON as trusted
   input and repair only the sentences it named as bad. Offered the
   choice between that and a simpler always-on design where the
   translation cache itself does the targeting automatically (a sentence
   whose primary translation is confirmed repetition-flagged gets one
   extra retry call, unconditionally, on every run -- first-ever or a
   rerun against a warm cache), the user chose the always-on design.
   This is simpler (no new CLI surface, no "trust this prior JSON as
   input" pathway to keep in sync with the pipeline's own output schema)
   and strictly more general (it also self-heals a sentence that becomes
   repetition-flagged for the first time on some future rerun, e.g. after
   an unrelated code change, with no manual `--repair-existing` step
   required).
2. **The terminology-mistranslation audit (`truges`/`pagesos`/`engreix`/
   `granges de mare`, documented as a known limitation in Round 11) will
   use a term list the user supplies**, not one built without pig-farming
   /Catalan-dialect domain expertise. This audit is **not built in this
   round** -- it is blocked on the user actually supplying that list, which
   has been requested but not yet received. Nothing else in this round
   depends on it.

**The architectural change, in detail.** Round 10's fix bumped the single,
global `GENERATION_PARAMS`/`GENERATION_VERSION` used for every sentence's
primary translation, which is why it required a full corpus re-run: the
cache key includes `generation_version` (see `translation_cache.py`), so
changing that one string for all callers invalidated the entire existing
cache, including the ~5,755 sentences that were never wrong in the first
place. This round reverts that: `GENERATION_PARAMS` and `GENERATION_
VERSION` are back to their exact pre-Round-10 values (`{"num_beams": 4,
"max_length": 512, "do_sample": False}` / `"nllb600M-beams4-maxlen512-
chunked-v2"`, no anti-repetition keys, no "antirep" marker) -- Test 59,
rewritten this round, pins this directly. **This means the real corpus's
existing cache, built entirely under this exact pre-Round-10
configuration, is fully valid again and will be hit for every sentence
that was never repetition-flagged**, with zero re-translation cost.

In place of the global change, a new, separate cache namespace and a
narrowly-scoped retry function do the actual repair. `RETRY_GENERATION_
PARAMS` (primary params plus `repetition_penalty=1.3`, `no_repeat_ngram_
size=4`) and `RETRY_GENERATION_VERSION` (a distinct version string,
containing "antirep-retry1", never equal to `GENERATION_VERSION`) are new
module-level constants. `NLLBTranslator.translate_batch()` gained a new
`generation_params: Optional[dict] = None` parameter -- when omitted it
behaves exactly as before (falls back to the module-level `GENERATION_
PARAMS`), and when supplied it overrides the decoding settings for that
one call only, leaving the module-level default and every other caller
untouched (Test 65 proves both the default-fallback and the per-call-only
override directly against the real, unmodified class, using the
established duck-typed fake-`self` pattern -- no model load). A new
function, `attempt_repetition_fallback_retry(source_text, translator,
cache, source_lang="ca", target_lang="es")`, checks the retry cache
namespace first (`cache.get(..., RETRY_GENERATION_VERSION)`); on a miss it
calls `translator.translate_batch([source_text], ..., generation_
params=RETRY_GENERATION_PARAMS)`, caches the result under `RETRY_
GENERATION_VERSION` on success, and returns a `cache_status` of
`"CACHE_HIT"`, `"FRESH"`, or `"FAILED"` alongside the translated text (or
`None` on failure -- a retry failure is logged and handled, never raised,
consistent with this project's standing rule that no single sentence's
translation failure may crash a multi-hour run).

`process()`'s Pass 2 wires this in: for every sentence whose primary
translation is independently confirmed as a repetition loop by `check_
degenerate_output()` (the same function and the same `REPETITION_MIN_
REPEATS=6` threshold from Round 11 -- nothing about what counts as
"pathological repetition" changed this round, only what happens once one
is found), exactly one fallback retry is attempted. If the retry's own
`check_degenerate_output()` result comes back clean, the retry's text
*replaces* `text_es` for that sentence and its clean diagnostic becomes
the one recorded as `sentence_quality["degenerate_output"]` (so a
resolved sentence is `FLAGGED=False` and contributes zero to `STEP2_
VALID`'s gate); if the retry does not resolve it, the *original* primary
text is kept unchanged (never replaced by a second bad translation) and
the sentence stays `FLAGGED=True`, still fatal, exactly as before this
round. Either way, the retry is attempted **once and only once** per
unique source text needing it: because the retry cache is keyed by source
text (like every other cache lookup in this project), two sentences that
happen to share the same pathological source text -- confirmed to occur in
the real corpus during the original diagnosis, e.g. "No, no." behind 14
flagged sentence IDs -- only ever trigger one real `translate_batch` call
between them; the second sentence's lookup is served as `CACHE_HIT` from
the first (Test 64 proves this directly, including that `CALL_LOG` shows
exactly one retry-mode call across two sentences).

Full per-sentence provenance is recorded regardless of outcome, in a new
`sentence_quality["repetition_fallback_retry"]` field (`None` when no
retry was needed): `attempted` (bool), `reason` (currently always
`"pathological_repetition"`), `cache_status`, `primary_generation_
version` and `retry_generation_version` (so it's always explicit which
exact decoding configuration produced each candidate), `primary_
degenerate_output` (the *original*, pre-retry diagnostic that triggered
the retry, preserved for audit even when the retry resolved it and the
sentence's live `degenerate_output` field now shows the clean result
instead), `resolved` (bool), and `selected_generation_version` (naming
whichever version's output actually became the sentence's final `text_
es`). This directly answers the audit-ability requirement behind the
user's original repair-mode design: a mixed-provenance corpus is never
silently ambiguous about which decoding configuration produced any given
sentence's final text, without needing a `--repair-existing` mode or a
"trust this prior JSON" pathway to get there.

`sentence_translation_summary` gained four corpus-wide rollup fields
mirroring the pattern already used for `sentence_language_mismatch_
count`/`_ids`: `repetition_fallback_retry_attempted_count`, `_resolved_
count`/`_resolved_ids`, and `_unresolved_count`/`_unresolved_ids`.
`print_validation_report()` gained matching print lines. An unresolved
retry does not change `STEP2_VALID`'s behavior at all -- it was fatal
before this round (as an unrepaired repetition-flagged sentence) and
remains fatal now (Test 63 proves this end to end: `text_es` is
unchanged, `resolved=False`, `translation_output_validity=False`,
`STEP2_VALID=False`); a resolved retry converts what would have been a
fatal sentence into a clean one (Test 62 proves this end to end,
including the exact `CALL_LOG` shape: one primary call, one retry call,
no wasted extra calls, and `STEP2_VALID=True`).

**New tests (62-65):** Test 62 is the full resolved-path integration test
described above. Test 63 is its unresolved-path mirror, using a new
`StubNLLBTranslator.REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS` opt-out (a
frozenset of source texts that stay bad even on a retry call) -- without
this opt-out, the stub's new default retry behavior (a retry call returns
clean output by default, matching the common case) would have silently
"fixed" Tests 34 and 54's intentionally-persistent bad text and broken
what those two existing tests were built to prove (that the fatal gate
still fires when nothing resolves the problem); both were updated to mark
their bad text with this opt-out so their original intent survives
unchanged. Test 64 proves retry-cache reuse across two sentences sharing
one bad source text. Test 65 exercises the real, unmodified `NLLBTranslator.
translate_batch()`'s new `generation_params` override directly (new fake
classes `_FakeTensorForGeneration`/`_FakeTokenizerForGeneration`/
`_FakeModelForGeneration`/`_FakeSelfForGeneration`, following the
established duck-typed fake-`self` pattern from Tests 2/3/14) -- no model
load, confirms both the default-fallback and per-call-only-override
behavior, and that `forced_bos_token_id` still resolves correctly in both
cases. `StubNLLBTranslator.translate_batch()`'s `CALL_LOG` entries are now
4-tuples (`source_lang, target_lang, list(texts), generation_params`) so
tests can distinguish primary calls (`generation_params is None`) from
retry calls (`generation_params` carrying `repetition_penalty`) directly,
as Tests 62 and 64 do. `test_preprocess_v2_reliability.py`: **426/426
checks passing** after this round -- superseding the "390/390" count
above.

**Re-run impact of this round, stated precisely:** none, for the corpus
that already exists. Because `GENERATION_VERSION` is back to its exact
pre-Round-10 value, a run against the real corpus's existing cache hits
every one of the ~5,755 sentences that were never repetition-flagged at
zero NLLB cost -- no full re-run, unlike Round 10's delivery. Only the
sentences genuinely confirmed as repetition loops (the ~36-37 real
failures identified across the Round 10/11 diagnoses) will incur new NLLB
calls, and only one call per unique bad source text, not per flagged
sentence ID (multiple sentence IDs sharing one bad source text, as
observed in the real data, share one retry). This is the direct
architectural fix for the concern the user raised about Round 11 still
carrying Round 10's full-re-run cost forward unnecessarily.

**Not addressed this round, and explicitly out of scope until the user
supplies a term list (per decision 2 above):** the domain-terminology
mistranslation audit. The `truges`/`pagesos`/`engreix`/`granges de mare`
examples from Round 11 remain a real, undetected gap -- `STEP2_VALID`
still says nothing about whether a specific noun was mistranslated to a
wrong-but-plausible-sounding word, only about structural/generation-level
failure. No code for this audit exists yet in this delivery.

### Round 12 fixes from external review

The user independently re-verified the Round 12 ZIP against the frozen
artifact (950 responses, 6,396 sentence IDs, 5,792 Catalan + 604 Spanish
sentence units, `input_sha256` matching) and the full architecture above
("Round 12 architecture: approved"), and found four concrete problems
with how it was packaged/implemented, all fixed in this delivery without
changing the approved architecture:

1. **A pre-run primary-cache coverage safety check, gating on request.**
   The Round 12 ZIP is, correctly, code-only -- it never bundles the real
   corpus's actual `data/cache/` file (see "Files in this delivery"
   below). The user identified the real operational risk this creates: if
   an operator replaces their project folder with a delivery and forgets
   to restore their existing warm cache first, `python preprocess_v2.py`
   would silently re-translate the entire corpus from scratch -- turning
   a cheap targeted repair into another multi-hour run, with no warning
   until it's too late. New function `compute_primary_cache_coverage()`
   (in `preprocess_v2.py`) checks, cheaply and read-only (see point about
   `TranslationCache.contains()` below), whether every one of this run's
   Catalan sentence units already has a primary-cache entry under the
   current `GENERATION_VERSION` -- counted by attempting the lookup for
   **every sentence unit individually** (not by counting distinct cache
   entries or distinct source texts first), per the user's own
   specification, so sentences sharing one source text (documented cache
   reuse throughout this project) are each counted correctly rather than
   silently undercounted. `process()` now always computes and prints this
   report immediately, before any NLLB call, in exactly the requested
   shape:
   ```
   Primary cache coverage before run:
     expected Catalan sentence units: 5792
     covered: 5792
     missing: 0
     SAFE TO REUSE PRIMARY CACHE = True
   ```
   A new `require_warm_primary_cache` parameter (default `False` -- zero
   behavior change for every existing caller, including this whole test
   suite) turns this into a hard gate: when `True` and even one sentence
   is missing, `process()` refuses to call the translation pass at all
   (zero NLLB calls made) and fails the run cleanly through the exact same
   machinery an aborted translation already uses (`translation_aborted_
   early=True`, `abort_reason` naming the shortfall, every pending
   sentence recorded `FAILED`, `STEP2_VALID=False`) -- no new exception
   type, no new failure path to keep in sync with the rest of the report.
   `main()`'s new `--require-warm-cache` CLI flag sets this for the real
   corpus run (see "CLI" and "Execution sequence" below); the default
   stays `False` specifically because a genuinely first-ever run has an
   empty cache by definition and must not be blocked by its own safety
   net. **Deliberately NOT wired into `terminal.py`'s menu-driven path**
   in this round -- `_create_processed_versions()` is explicitly not tied
   to the frozen production corpus (see its own comment, unchanged since
   Round 7) and has no CLI flag surface to carry this through; a caller
   who wants the same protection there can pass
   `require_warm_primary_cache=True` directly to `preprocess_v2.process()`.
   New `TranslationCache.contains()` method (`translation_cache.py`): a
   pure existence check that, unlike `get()`, does **not** increment
   `hits_this_run`/`misses_this_run` -- using `get()` for this coverage
   check would have inflated the run's real cache-hit statistics by
   however many sentences the check walks, ahead of the pipeline's own
   genuine lookups for those same sentences moments later.
2. **Successful-repair provenance now correctly updates the TOP-LEVEL
   `translation_provenance` field, not just the nested diagnostic.** This
   was a real bug: `provenance` (the primary attempt's `CACHE_HIT`/
   `FRESH_OK`/etc.) was computed once before the repetition-retry check
   and never reassigned even when the retry resolved the sentence and
   `text_es_sent` WAS replaced -- so a repaired sentence could be written
   to `sentences_out[...]["translation_provenance"]` (and rolled up into
   `sentence_translation_summary.by_translation_provenance` and
   `retried_ok_translations`) as if nothing had changed, silently
   understating how much repair work actually happened. Fixed in `process
   ()`'s Pass 2: the original primary provenance is preserved for audit
   under the new `repetition_fallback_retry.primary_translation_
   provenance` field, and `provenance` itself (and `sentence_status`, which
   mirrors it) is reassigned to a new value, `"REPETITION_REPAIRED"`, the
   moment a retry resolves a sentence -- also recorded in the nested record
   as `repetition_fallback_retry.selected_translation_provenance`, so the
   top-level field and the nested audit trail always agree. **Deliberately
   NOT named `"RETRY_OK"`**, despite that being the label used in the
   user's own suggested provenance table -- `"RETRY_OK"` already means
   something different and pre-existing in this codebase (a PRIMARY-pass
   piece that needed a transient-failure retry inside `translate_texts_
   batched`, still entirely under `GENERATION_PARAMS`, unrelated to
   repetition -- see `_on_retry_piece_done`'s long-standing logic). Reusing
   it for the Round 12 repair outcome would have silently merged two
   unrelated kinds of retry under one label, exactly the ambiguity this
   project has spent eleven rounds eliminating. The corpus-wide rollups
   are correct now as a direct consequence (no separate fix needed there):
   `sentence_translation_summary.by_translation_provenance` and the report's
   new flat `repetition_repaired_translations` field (plus its
   `runtime_info` mirror) both read straight off the now-correct per-
   sentence `translation_provenance` values.
3. **A resolved fallback retry is now persisted to disk immediately.**
   `attempt_repetition_fallback_retry()` previously called `cache.set()`
   but not `cache.save()`, leaving a successful repair sitting only in
   memory until some later batch's save happened to flush it. The PRIMARY
   translation path has treated every successful translation this way
   since the fifth external review specifically so a crash later in the
   same run (e.g. during the embeddings/round-trip diagnostic pass, which
   runs after all translation work) cannot lose already-completed work --
   the same principle now applies here: `attempt_repetition_fallback_
   retry()` calls `cache.save()` (atomic, warning-only on failure, same as
   every other call site) immediately after a successful `cache.set()`, so
   an expensive, hard-won repair (one real extra NLLB call, made at most
   once per unique bad source text) is never silently redone on a future
   run just because the process died before some unrelated later save.
4. **A blank/empty fallback retry result is rejected, not cached as a
   success.** `translate_batch()`'s underlying `tokenizer.batch_decode`
   can legitimately return an empty string for a genuinely empty
   generation -- the PRIMARY translation path already guards against this
   exact case (`translate_texts_batched`'s `if translated is None or not
   str(translated).strip()`), but `attempt_repetition_fallback_retry()`
   did not have the matching guard, so a blank retry result would have
   been cached under `RETRY_GENERATION_VERSION` as if it were a genuine
   repair and served back as a false `CACHE_HIT` on every future lookup
   for that source text. Now checked identically to the primary path: a
   blank/`None` retry result is never cached, is reported as `"FAILED"`
   (not a success), and the caller correctly keeps the sentence's original
   (still-degenerate) primary translation, exactly like any other retry
   failure.

**New tests (66-70):** Test 66 is the direct regression test for fix 2,
using the user's own reported scenario (a primary translation served as
`CACHE_HIT` that turns out to need repair) and asserting the top-level
`translation_provenance`, `sentence_status`, the nested `repetition_
fallback_retry.primary_translation_provenance`/`selected_translation_
provenance` pair, and every corpus-wide rollup (`by_translation_
provenance`, `repetition_repaired_translations`, its `runtime_info`
mirror) all agree and are no longer left stale under `CACHE_HIT`. Test 67
proves fix 4 both at the unit level (`attempt_repetition_fallback_retry()`
called directly against a translator stub that returns `""`) and end to
end through `process()` (a new `StubNLLBTranslator.REPETITIVE_RETRY_
RETURNS_BLANK_FOR_TEXTS` opt-in), confirming a blank retry is reported
`"FAILED"`, is never cached, and the sentence correctly stays `FLAGGED`
with its original text untouched. Test 68 proves fix 3 by calling
`attempt_repetition_fallback_retry()` against a real on-disk cache file,
then opening a **second, independent** `TranslationCache` instance
pointed at the same path without ever calling `save()` again on the
first -- the repaired translation is visible there, proof it was actually
flushed to disk inside the function itself, not left for some later call
to persist. Test 69 proves fix 1 with three scenarios: a cold cache with
`require_warm_primary_cache=True` correctly aborts with zero NLLB calls
made; a genuinely warm cache with the same flag set is correctly **not**
blocked (and still needs no primary-direction NLLB calls -- any `CALL_LOG`
entries there are the unrelated round-trip diagnostic, which legitimately
still runs `es->ca` against the now-cached translation); and a cold cache
with the flag left at its default (`False`) translates normally,
confirming this round's change is fully opt-in and does not alter
behavior for this test suite or any other existing caller. Test 70 is a
direct unit test of `compute_primary_cache_coverage()`: two sentence IDs
sharing one cached source text both count as covered independently (not
"1 distinct entry, therefore covered"), a response with `translation_
required=False` is correctly excluded from the expected count entirely,
and `cache.stats()`'s hit/miss counters are provably untouched by the
coverage check (using `contains()`, not `get()`).
`test_preprocess_v2_reliability.py`: **466/466 checks passing** after
these fixes -- superseding the "426/426" count above.

**Not run this round either:** the real NLLB corpus. None of these four
fixes touch generation parameters, `GENERATION_VERSION`, or the retry
mechanism's decision logic (when a retry is attempted, what makes it
"resolved") -- they fix how the *outcome* of that unchanged logic is
persisted, cached, and labeled. A run against the real corpus's existing
cache (restored into place first, per the execution sequence below) still
needs no full re-translation.

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
`data/cache/nllb_sentence_translation_cache_v1.json` (the real production
cache -- see "Cache" above for why this section previously named the
wrong, pre-Round-7 file here), or any other production file. `--smoke-
test` uses its own fully isolated cache
(`data/output/smoke_test/_smoke_test_cache.json`), deleted and rebuilt
fresh on every run.

`--require-warm-cache` (Round 12, added from external review): before
translating anything, verifies every Catalan sentence in the frozen
corpus is already present in the primary cache under the current
`GENERATION_VERSION`, and refuses to proceed -- no NLLB calls made -- if
any are missing, rather than silently re-translating them. See "Round 12
fixes from external review" above for why this exists and exactly what it
checks. Pass this on every run **except** a genuinely first-ever run
against an empty cache (see "Execution sequence" below). Ignored by
`--preflight`/`--smoke-test`, neither of which calls `process()` against
the real corpus.

Exit code is 0 only when the requested mode passed (or, for a full run,
when `STEP2_VALID` is `True`); non-zero otherwise.

## Execution sequence for the next real run

**Updated for Round 12.** This delivery is a code-only ZIP by design (see
"Files in this delivery" below) -- it does NOT contain the real corpus's
existing translation cache. Restoring that cache to its correct path is
now step 0, before anything else, and is exactly what `--require-warm-
cache` (step 5 below) exists to verify was actually done correctly rather
than silently trusting it.

0. **Copy the existing cache back into place first.** From wherever the
   prior real run's cache was preserved, restore it to
   `data/cache/nllb_sentence_translation_cache_v1.json` in THIS project
   directory (the exact path `SENTENCE_CACHE_PATH` points to -- see
   "Cache" above for why this is the file that matters, not the
   similarly-named `nllb_translation_cache_v1.json`). Skip this step ONLY
   for a genuinely first-ever run against an empty cache.
1. `python test_preprocess_v2_reliability.py` -- confirms the logic itself
   (offline, ~seconds, no model download). **466/466 checks** as of Round
   12's external-review fixes.
2. `python preprocess_v2.py --preflight` -- confirms the real model loads
   and translates on this machine, and which device it resolved to
   (`cuda` / `mps` / `cpu`).
3. `python preprocess_v2.py --smoke-test` -- confirms the full pipeline
   end-to-end on two small real examples. Uses its own isolated cache
   file, never the real one from step 0.
4. Review both outputs.
5. Only once all of the above pass, run the real 950-response corpus via
   `python preprocess_v2.py --require-warm-cache` (the CLI entry point,
   not `terminal.py`'s menu -- `main()` is what ties a run to the actual
   frozen corpus artifact and, as of Round 12, is the only entry point
   this gate is wired into; see "Round 12 fixes from external review"
   above). This prints the primary-cache coverage report FIRST, before
   any NLLB call:
   ```
   Primary cache coverage before run:
     expected Catalan sentence units: 5792
     covered: 5792
     missing: 0
     SAFE TO REUSE PRIMARY CACHE = True
   ```
   If step 0 was done correctly, `missing` is `0` and the run proceeds,
   translating only the sentences step 0's cache never covered in the
   first place plus, per sentence, at most one new targeted anti-
   repetition retry call for whatever is still genuinely repetition-
   flagged. If step 0 was skipped or restored the wrong file, the run
   stops immediately here (`STEP2_VALID=False`, zero NLLB calls made)
   instead of silently re-translating the whole corpus -- fix the cache
   path and re-run, rather than proceeding.
6. Read the printed validation report and/or
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
7. Do not commit or push until that real run has been reviewed.

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
