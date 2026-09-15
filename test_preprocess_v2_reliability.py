"""
test_preprocess_v2_reliability.py -- offline unit/integration tests for the
frozen NLLB redesign of Step 2 (rev. 5): local facebook/nllb-200-distilled-
600M translation (Catalan -> Spanish only), tokenizer-verified chunk-safety
(count_tokens / _assert_within_token_limit / _enforce_real_token_budget),
the three-tier STEP2_VALID validity model (structural_validity AND
translation_completeness_validity AND translation_output_validity) with its
PASS/REVIEW/FAIL translation_sanity_status, the quality-diagnostic
functions, the NLLB-aware cache, batching, device-fallback bookkeeping, and
the terminal.py integration (stopping cleanly when Step 2 is invalid,
continuing only when it is valid).

This suite needs NO network access and does NOT download the real NLLB
checkpoint or the real sentence-transformers embedding model -- both are
monkeypatched with deterministic, offline stand-ins (StubNLLBTranslator,
StubSimilarityScorer) that satisfy the same interface preprocess_v2.py's
real classes expose. It DOES exercise the real preprocess_v2.py logic
(process(), translate_texts_batched(), the quality-diagnostic functions,
TranslationCache, two NLLBTranslator methods that don't need a loaded
model -- _forced_bos_token_id and _auto_detect_device, called on the REAL
class captured before it is monkeypatched -- the real
SemanticSimilarityScorer.batch_cosine_similarity method against a fake
embedding model, and terminal.py's _create_processed_versions()). None of
this is a reimplementation of that logic under a different name.

It does NOT verify that the REAL facebook/nllb-200-distilled-600M model
loads or translates correctly on real hardware -- that is exactly what
`python preprocess_v2.py --preflight` and `--smoke-test` are for, and both
need to be run separately, on a machine with network access to download
the checkpoint once, before trusting a full corpus run.

Run it from the project root:

    python test_preprocess_v2_reliability.py

Prints PASS/FAIL per check and a final summary line; exits 0 only if every
check passed.
"""
import os
import sys
import json
import shutil
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import preprocess_v2  # noqa: E402

# Captured BEFORE any monkeypatching below replaces these module
# attributes with offline stubs -- these keep a live reference to the
# REAL classes so a few tests (2, 3, 14) can exercise their real methods
# directly (via a duck-typed fake `self`, so no real model/embedding
# download is triggered) rather than testing a stand-in under the same
# name.
_RealNLLBTranslator = preprocess_v2.NLLBTranslator
_RealSemanticSimilarityScorer = preprocess_v2.SemanticSimilarityScorer


# ---------------------------------------------------------------------------
# Offline stand-ins
# ---------------------------------------------------------------------------

# Real, distinct sentences per target language so the REAL check_target_
# language() (real langdetect) actually passes/fails the way it would
# against genuine model output -- these are not placeholder strings.
STUB_ES_OUTPUT = "Esta es una respuesta traducida al español para la prueba de traducción del corpus."
STUB_CA_OUTPUT = "Aquesta és una resposta traduïda al català per a la prova de traducció del corpus."
STUB_EN_OUTPUT = "This is a completely different sentence written only in English for testing purposes."
# Confirmed (via real langdetect) to be detected as Spanish with very high
# confidence, unlike an earlier draft of this constant ("el gato el gato
# ...") which real langdetect actually classified as Italian -- that would
# have made every repetition_one test case ALSO trip the language-mismatch
# check, conflating two diagnostics this suite needs to exercise
# independently (see Test 34).
# Round 11: REPETITION_MIN_REPEATS was raised 4 -> 6, and check_degenerate_
# output's outer length guard requires len(tokens) >= REPETITION_NGRAM_SIZE
# * REPETITION_MIN_REPEATS (3*6=18) before it even looks for a repeated
# n-gram -- so this needs enough repeats to clear BOTH bars with margin,
# not just the old 4x.
STUB_REPETITIVE_OUTPUT = "el perro el perro el perro el perro el perro el perro el perro el perro el perro corre"


class NLLBTranslationError(preprocess_v2.NLLBTranslationError):
    pass


class StubNLLBTranslator:
    """Duck-types preprocess_v2.NLLBTranslator's public interface (the
    surface translate_texts_batched(), run_preflight(), run_smoke_test(),
    main(), and terminal.py's _create_processed_versions() actually call)
    without loading any real model. Class-level switches let different
    tests exercise success, always-fail, empty-output, identical-output,
    and repetition-output behavior.
    """

    FAIL_MODE = "none"  # "none" | "always" | "n_times" | "empty_one" | "identical_one" | "repetition_one" | "wrong_language_one"
    FAILS_REMAINING = 0
    EMPTY_FOR_TEXTS = frozenset()
    IDENTICAL_FOR_TEXTS = frozenset()
    REPETITIVE_FOR_TEXTS = frozenset()
    WRONG_LANGUAGE_FOR_TEXTS = frozenset()
    # Round 12: texts in REPETITIVE_FOR_TEXTS get STUB_REPETITIVE_OUTPUT on
    # their PRIMARY translate_batch call, same as always. By default, a
    # subsequent RETRY call (generation_params carrying "repetition_penalty"
    # -- see attempt_repetition_fallback_retry()) for that same text "fixes"
    # it, returning STUB_ES_OUTPUT instead -- this is what most tests using
    # repetition_one want, so process()'s new fallback-retry path has
    # something realistic to exercise without every existing repetition
    # test needing to opt in explicitly. A text listed here is the
    # exception: it STAYS bad even on retry (still STUB_REPETITIVE_OUTPUT),
    # for tests that specifically need an unfixable case.
    REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS = frozenset()
    # Round 12 (external review): a text in this set, on its RETRY call
    # only (never its primary call), gets a blank ("") result instead of
    # either STUB_ES_OUTPUT or STUB_REPETITIVE_OUTPUT -- for testing that
    # attempt_repetition_fallback_retry() correctly rejects a blank retry
    # result rather than caching or "succeeding" with it.
    REPETITIVE_RETRY_RETURNS_BLANK_FOR_TEXTS = frozenset()
    CALL_LOG = []  # list of (source_lang, target_lang, [texts], generation_params) per translate_batch call

    def __init__(self, model_name=preprocess_v2.NLLB_MODEL_NAME, requested_device=None, batch_size=preprocess_v2.DEFAULT_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.requested_device = requested_device or "cpu"
        self.device = self.requested_device
        self.device_fallback = False
        self.fallback_reason = None

    def translate_batch(self, texts, source_lang, target_lang, generation_params=None):
        StubNLLBTranslator.CALL_LOG.append((source_lang, target_lang, list(texts), generation_params))
        is_retry_call = generation_params is not None and "repetition_penalty" in generation_params

        if StubNLLBTranslator.FAIL_MODE == "always":
            raise NLLBTranslationError("stubbed: always fails")
        if StubNLLBTranslator.FAIL_MODE == "n_times" and StubNLLBTranslator.FAILS_REMAINING > 0:
            StubNLLBTranslator.FAILS_REMAINING -= 1
            raise NLLBTranslationError("stubbed: transient failure")

        default_output = STUB_ES_OUTPUT if target_lang == "es" else STUB_CA_OUTPUT
        outputs = []
        for text in texts:
            if StubNLLBTranslator.FAIL_MODE == "empty_one" and text in StubNLLBTranslator.EMPTY_FOR_TEXTS:
                outputs.append("")
            elif StubNLLBTranslator.FAIL_MODE == "identical_one" and text in StubNLLBTranslator.IDENTICAL_FOR_TEXTS:
                outputs.append(text)
            elif StubNLLBTranslator.FAIL_MODE == "repetition_one" and text in StubNLLBTranslator.REPETITIVE_FOR_TEXTS:
                if is_retry_call and text in StubNLLBTranslator.REPETITIVE_RETRY_RETURNS_BLANK_FOR_TEXTS:
                    outputs.append("")
                elif is_retry_call and text not in StubNLLBTranslator.REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS:
                    outputs.append(STUB_ES_OUTPUT)
                else:
                    outputs.append(STUB_REPETITIVE_OUTPUT)
            elif StubNLLBTranslator.FAIL_MODE == "wrong_language_one" and text in StubNLLBTranslator.WRONG_LANGUAGE_FOR_TEXTS:
                outputs.append(STUB_EN_OUTPUT)
            else:
                outputs.append(default_output)
        return outputs

    def model_info(self):
        return {
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "actual_device": self.device,
            "device_fallback": self.device_fallback,
            "fallback_reason": self.fallback_reason,
            "generation_params": dict(preprocess_v2.GENERATION_PARAMS),
            "generation_version": preprocess_v2.GENERATION_VERSION,
            "batch_size": self.batch_size,
        }

    @classmethod
    def reset(cls):
        cls.FAIL_MODE = "none"
        cls.FAILS_REMAINING = 0
        cls.EMPTY_FOR_TEXTS = frozenset()
        cls.IDENTICAL_FOR_TEXTS = frozenset()
        cls.REPETITIVE_FOR_TEXTS = frozenset()
        cls.WRONG_LANGUAGE_FOR_TEXTS = frozenset()
        cls.REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS = frozenset()
        cls.REPETITIVE_RETRY_RETURNS_BLANK_FOR_TEXTS = frozenset()
        cls.CALL_LOG = []


class StubSimilarityScorer:
    """Deterministic stand-in for SemanticSimilarityScorer -- avoids
    downloading the real multilingual sentence-embedding model. Returns a
    bounded [0, 1] proxy similarity, which is enough to exercise
    process()'s distribution reporting and roundtrip-diagnostic wiring
    without asserting anything about real embedding quality (that is
    covered separately, against the REAL class, in Test 14 below).
    """

    def __init__(self, *args, **kwargs):
        self.unavailable_reason = None

    @property
    def available(self):
        return True

    @property
    def load_attempted(self):
        # Stub is always "ready" -- matches the real SemanticSimilarityScorer's
        # post-lazy-load state, so process()'s report-assembly code (which
        # reads `load_attempted` rather than `available` to avoid forcing a
        # real load just to fill in a summary field) works unchanged against
        # this stub too.
        return True

    def batch_cosine_similarity(self, texts_a, texts_b):
        sims = []
        for a, b in zip(texts_a, texts_b):
            if not a or not b:
                sims.append(0.0)
            else:
                set_a, set_b = set(a.lower()), set(b.lower())
                overlap = len(set_a & set_b) / max(len(set_a | set_b), 1)
                sims.append(overlap)
        return sims


class StubPreprocessor:
    """Duck-types the two Preprocessor methods process() actually calls,
    without needing spaCy language models.
    """

    def full_preprocess_v2(self, text, lang):
        return text.lower().strip()

    def split_sentences_strict(self, text, lang):
        parts = [p.strip() for p in text.split(".") if p.strip()]
        return parts if parts else [text]


preprocess_v2.NLLBTranslator = StubNLLBTranslator
preprocess_v2.SemanticSimilarityScorer = StubSimilarityScorer
preprocess_v2.Preprocessor = StubPreprocessor
preprocess_v2.LOCAL_RETRY_DELAY_SECONDS = 0.0

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def fresh_cache():
    tmp_dir = tempfile.mkdtemp(prefix="nllb_test_cache_")
    return preprocess_v2.TranslationCache(os.path.join(tmp_dir, "cache.json")), tmp_dir


preprocessor = StubPreprocessor()
similarity_scorer = StubSimilarityScorer()


# ===========================================================================
print("=== Test 1: NLLB language-code mapping ===")
check("ca -> cat_Latn", preprocess_v2.LANG_TO_NLLB["ca"] == "cat_Latn")
check("es -> spa_Latn", preprocess_v2.LANG_TO_NLLB["es"] == "spa_Latn")
check("only two languages mapped (no English)", set(preprocess_v2.LANG_TO_NLLB.keys()) == {"ca", "es"})
check("required direction is ca->es only", preprocess_v2.REQUIRED_TRANSLATION_DIRECTION == "ca->es")


# ===========================================================================
print("\n=== Test 2: _forced_bos_token_id compatibility shim (real class, no model load) ===")


class _FakeTokenizerWithDict:
    lang_code_to_id = {"spa_Latn": 42, "cat_Latn": 7}

    def convert_tokens_to_ids(self, token):
        raise AssertionError("should not be called when lang_code_to_id has the key")


class _FakeTokenizerWithoutDict:
    lang_code_to_id = None

    def convert_tokens_to_ids(self, token):
        return {"spa_Latn": 99, "cat_Latn": 13}[token]


class _FakeTokenizerWithIncompleteDict:
    lang_code_to_id = {"cat_Latn": 7}  # missing spa_Latn

    def convert_tokens_to_ids(self, token):
        return {"spa_Latn": 55}[token]


class _FakeSelf:
    """A plain namespace, NOT a real NLLBTranslator instance -- avoids
    __init__ (which would try to download/load the real model). This
    works because _forced_bos_token_id only ever reads self.tokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


check(
    "uses lang_code_to_id when present and complete",
    _RealNLLBTranslator._forced_bos_token_id(_FakeSelf(_FakeTokenizerWithDict()), "spa_Latn") == 42,
)
check(
    "falls back to convert_tokens_to_ids when lang_code_to_id is absent",
    _RealNLLBTranslator._forced_bos_token_id(_FakeSelf(_FakeTokenizerWithoutDict()), "spa_Latn") == 99,
)
check(
    "falls back to convert_tokens_to_ids when lang_code_to_id exists but lacks the key",
    _RealNLLBTranslator._forced_bos_token_id(_FakeSelf(_FakeTokenizerWithIncompleteDict()), "spa_Latn") == 55,
)


# ===========================================================================
print("\n=== Test 3: device auto-detection (real class, no model load) ===")
import torch as _torch  # noqa: E402


class _FakeMps:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


_orig_cuda_is_available = _torch.cuda.is_available
_orig_backends_mps = getattr(_torch.backends, "mps", None)

try:
    _torch.cuda.is_available = lambda: True
    check("cuda preferred when available", _RealNLLBTranslator._auto_detect_device() == "cuda")

    _torch.cuda.is_available = lambda: False
    _torch.backends.mps = _FakeMps(True)
    check("mps used when cuda unavailable but mps available", _RealNLLBTranslator._auto_detect_device() == "mps")

    _torch.backends.mps = _FakeMps(False)
    check("cpu used when neither cuda nor mps available", _RealNLLBTranslator._auto_detect_device() == "cpu")
finally:
    _torch.cuda.is_available = _orig_cuda_is_available
    if _orig_backends_mps is not None:
        _torch.backends.mps = _orig_backends_mps


# ===========================================================================
print("\n=== Test 4: identity Spanish->Spanish path (no translation call) ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()

raw_data = {"interview_es_1": {"q1": "Esta es una respuesta suficientemente larga en español para la prueba."}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)

response = responses_out["interview_es_1::q1"]
check("source_language detected as es", response["source_language"] == "es")
check("translation_required is False", response["translation_required"] is False)
check("text_es_status is IDENTITY_COPY", response["text_es_status"] == "IDENTITY_COPY")
# Not a byte-exact comparison: under the frozen-sentence architecture,
# text_es is rebuilt as " ".join(sentence texts), and StubPreprocessor's
# naive split-on-"." (unlike the real spaCy-backed splitter) drops the
# separating period when it segments the single source sentence. That's a
# known stub artifact (see the identical _normalize_words rationale in
# Test 25), not a translation bug -- the actual invariant this response
# path guarantees is lexical content preservation, exactly like the
# frozen-file reconstruction check in freeze_sentence_segmentation.py.
check(
    "text_es equals original_text (copied, not translated)",
    [w.strip(".,;:!?") for w in response["text_es"].split()]
    == [w.strip(".,;:!?") for w in response["original_text"].split()],
)
check("no translate_batch call happened for a same-language response", len(StubNLLBTranslator.CALL_LOG) == 0)
check("STEP2_VALID True for an all-Spanish corpus", report["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 5: Catalan->Spanish translation path ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()

raw_data = {"interview_ca_1": {"q1": "Aquesta és una resposta prou llarga en català per a la prova de traducció."}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)

response = responses_out["interview_ca_1::q1"]
check("source_language detected as ca", response["source_language"] == "ca")
check("translation_required is True", response["translation_required"] is True)
check("text_es_status is TRANSLATED (fresh cache)", response["text_es_status"] == "TRANSLATED")
check("text_es is the stub Spanish output", response["text_es"] == STUB_ES_OUTPUT)
check("topic_text_es_raw set", response["topic_text_es_raw"] == STUB_ES_OUTPUT)
check("topic_text_es_clean set (from StubPreprocessor.full_preprocess_v2)", response["topic_text_es_clean"] == STUB_ES_OUTPUT.lower().strip())
check("target_language_check passed (real langdetect on real Spanish text)", response["quality"]["target_language_check"]["passed"] is True)
check("sentence_ids populated", len(response["sentence_ids"]) > 0)
check("STEP2_VALID True for a clean Catalan->Spanish run", report["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 6: batching respects translator.batch_size ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=3)
texts = [f"Text numero {i} unic per evitar la memoria cau." for i in range(7)]
batch_tracker = preprocess_v2.ConsecutiveBatchFailureTracker()
results = preprocess_v2.translate_texts_batched(texts, "ca", "es", translator, cache, batch_tracker)

batch_sizes = [len(call[2]) for call in StubNLLBTranslator.CALL_LOG]
check("7 texts split into batches of size <= 3", all(n <= 3 for n in batch_sizes))
check("batch sizes are exactly [3, 3, 1]", batch_sizes == [3, 3, 1])
check("all 7 results are TRANSLATED", all(status == "TRANSLATED" for _, status in results))
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 7: cache read/write (hit avoids a second model call) ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
text = "Un text catala unic per a la prova de la memoria cau de traduccio."

first = preprocess_v2.translate_texts_batched([text], "ca", "es", translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker())
check("first call is a real TRANSLATED", first[0][1] == "TRANSLATED")
check("cache has 1 entry after first call", cache.stats()["entries"] == 1)
check("cache counted a write", cache.stats()["writes_this_run"] == 1)

calls_before = len(StubNLLBTranslator.CALL_LOG)
second = preprocess_v2.translate_texts_batched([text], "ca", "es", translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker())
check("second call is CACHE_HIT", second[0][1] == "CACHE_HIT")
check("no new translate_batch call on a cache hit", len(StubNLLBTranslator.CALL_LOG) == calls_before)
check("cache hit returns the same translated text", second[0][0] == first[0][0])

miss_record = cache.get(text, "ca", "es", preprocess_v2.NLLB_MODEL_NAME, "a-different-generation-version")
check("a different generation_version is a cache miss (settings-change safety)", miss_record is None)
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 8: a failed batch is never cached ===")
StubNLLBTranslator.reset()
StubNLLBTranslator.FAIL_MODE = "always"
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
tracker = preprocess_v2.ConsecutiveBatchFailureTracker(max_consecutive=5)
texts = ["Text que sempre fallara la traduccio number one.", "Text que sempre fallara number two."]

results = preprocess_v2.translate_texts_batched(texts, "ca", "es", translator, cache, tracker)
check("both results are WARNING_TRANSLATION_FAILED", all(status == "WARNING_TRANSLATION_FAILED" for _, status in results))
check("both translated texts are None", all(text is None for text, _ in results))
check("nothing was written to the cache", cache.stats()["entries"] == 0)
check("failure tracker recorded exactly one consecutive failure (one batch)", tracker.consecutive_failures == 1)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 9: consecutive-batch-failure abort threshold ===")
StubNLLBTranslator.FAIL_MODE = "always"
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=1)
tracker = preprocess_v2.ConsecutiveBatchFailureTracker(max_consecutive=3)
texts = [f"Text numero {i} que fallara sempre." for i in range(5)]

raised = False
try:
    preprocess_v2.translate_texts_batched(texts, "ca", "es", translator, cache, tracker)
except preprocess_v2.NLLBTranslationError:
    raised = True
check("aborts with NLLBTranslationError once 3 consecutive batches fail", raised)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 10: empty-generation handling (per-item, not per-batch) ===")
StubNLLBTranslator.FAIL_MODE = "empty_one"
good_text = "Un text catala que es tradueix perfectament be per a la prova."
empty_text = "Un text catala que produeix una sortida buida del model."
StubNLLBTranslator.EMPTY_FOR_TEXTS = frozenset([empty_text])
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=8)

results = preprocess_v2.translate_texts_batched([good_text, empty_text], "ca", "es", translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker())
by_text = dict(zip([good_text, empty_text], results))
check("the good text is TRANSLATED", by_text[good_text][1] == "TRANSLATED")
check("the empty-output text is WARNING_TRANSLATION_FAILED", by_text[empty_text][1] == "WARNING_TRANSLATION_FAILED")
check("only the successful item was cached", cache.stats()["entries"] == 1)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 11: language-mismatch and degenerate-output diagnostics ===")
check("real Spanish output passes an es target-language check", preprocess_v2.check_target_language(STUB_ES_OUTPUT, "es")["passed"] is True)
check("a sufficiently-long check is reported as ASSESSED", preprocess_v2.check_target_language(STUB_ES_OUTPUT, "es")["status"] == "ASSESSED")
check("real English output fails an es target-language check", preprocess_v2.check_target_language(STUB_EN_OUTPUT, "es")["passed"] is False)

# Short-text language check (fix for the third external review's false-
# positive concern): text too short for _is_sufficiently_long is NOT
# assessed at all -- passed is None (never True/False), status says why.
# This corpus's real responses include 46 of 3 words or fewer ("Sí.",
# "No.", "PC.", "2009") where langdetect is unreliable-to-meaningless.
short_lang_check = preprocess_v2.check_target_language("Sí.", "es")
check("a short output is NOT assessed for target language", short_lang_check["status"] == "NOT_ASSESSED_SHORT_TEXT")
check("a short, unassessed output's passed is None, not False (never counts as a mismatch)", short_lang_check["passed"] is None)

degenerate_empty = preprocess_v2.check_degenerate_output("source text", "")
check("empty output flagged as degenerate (any length)", degenerate_empty["is_empty"] is True and degenerate_empty["flagged"] is True)

# Long-enough identical source/target IS fatal -- a real translation
# shouldn't come back byte-identical to a substantial Catalan source.
long_identical_text = "aquest es un text prou llarg per considerar sospitos que surti identic despres de traduir-lo"
degenerate_identical_long = preprocess_v2.check_degenerate_output(long_identical_text, long_identical_text)
check(
    "a LONG output identical to source is fatal (is_identical_to_source_fatal)",
    degenerate_identical_long["is_identical_to_source"] is True and degenerate_identical_long["is_identical_to_source_fatal"] is True,
)
check("...and therefore flagged as degenerate", degenerate_identical_long["flagged"] is True)

# Short identical source/target (the "Sí." -> "Sí." case) is frequently
# CORRECT, not degenerate -- must NOT be fatal, per the third external
# review's explicit ask. Still reported as is_identical_to_source=True
# (informational -- see short_text_identical_output_count in
# translation_quality_summary) but not fatal, and not in `flagged`.
degenerate_identical_short = preprocess_v2.check_degenerate_output("Sí.", "Sí.")
check("a SHORT output identical to source is NOT fatal", degenerate_identical_short["is_identical_to_source"] is True and degenerate_identical_short["is_identical_to_source_fatal"] is False)
check("...and therefore NOT flagged as degenerate", degenerate_identical_short["flagged"] is False)

degenerate_repetition = preprocess_v2.check_degenerate_output("el gat original", STUB_REPETITIVE_OUTPUT)
check("repeated 3-gram flagged as degenerate", degenerate_repetition["repetition_flag"] is True and degenerate_repetition["flagged"] is True)

degenerate_clean = preprocess_v2.check_degenerate_output("un text original catala", STUB_ES_OUTPUT)
check("a normal, varied translation is NOT flagged as degenerate", degenerate_clean["flagged"] is False)


# ===========================================================================
print("\n=== Test 12: length diagnostics ===")
short_target = preprocess_v2.compute_length_diagnostics("a" * 100, "a" * 10)  # ratio 0.1 -> outlier (< 0.4)
check("very short output vs. source flagged as a length-ratio outlier", short_target["length_ratio_outlier"] is True)

long_target = preprocess_v2.compute_length_diagnostics("a" * 10, "a" * 100)  # ratio 10.0 -> outlier (> 3.0)
check("very long output vs. source flagged as a length-ratio outlier", long_target["length_ratio_outlier"] is True)

normal_target = preprocess_v2.compute_length_diagnostics("a" * 100, "a" * 120)  # ratio 1.2 -> within bounds
check("a normal-length output is NOT flagged as a length-ratio outlier", normal_target["length_ratio_outlier"] is False)


# ===========================================================================
print("\n=== Test 13: distribution summary (reported, never thresholded) ===")
empty_summary = preprocess_v2._distribution_summary([])
check("empty distribution has count 0 and None stats (never raises)", empty_summary["count"] == 0 and empty_summary["mean"] is None)

values = [0.9, 0.8, 0.95, 0.4, 0.85]
summary = preprocess_v2._distribution_summary(values)
check("distribution count matches input length", summary["count"] == 5)
check("distribution mean matches manual calculation", abs(summary["mean"] - (sum(values) / len(values))) < 1e-9)
check("distribution min is the actual minimum", summary["min"] == 0.4)
check("count_below_informational_bound counts values under the bound, doesn't gate anything", summary["count_below_informational_bound"] == 1)


# ===========================================================================
print("\n=== Test 14: SemanticSimilarityScorer.batch_cosine_similarity (real method, fake embedding model) ===")
import numpy as _np  # noqa: E402


class _FakeEmbeddingModel:
    VECTORS = {
        "a": _np.array([1.0, 0.0]),
        "b": _np.array([1.0, 0.0]),   # identical direction to "a" -> similarity 1.0
        "c": _np.array([0.0, 1.0]),   # orthogonal to "a"/"b" -> similarity 0.0
    }

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return _np.stack([self.VECTORS[t] for t in texts])


# object.__new__ bypasses __init__ (which would try to download the real
# sentence-transformers model) while still testing the REAL, unmodified
# batch_cosine_similarity method on the REAL class captured above.
real_scorer = object.__new__(_RealSemanticSimilarityScorer)
real_scorer.model = _FakeEmbeddingModel()
real_scorer.model_name = "fake-embedding-model"
real_scorer.unavailable_reason = None
# Pretend a load already happened (this is what __init__ + a real first
# use would leave behind) -- object.__new__ bypasses __init__ entirely, so
# without this the lazy `available`/`_ensure_loaded()` machinery added in
# Round 7 (see Test 51) would try to read an attribute that was never set.
real_scorer._load_attempted = True

sims = real_scorer.batch_cosine_similarity(["a", "a"], ["b", "c"])
check("identical-direction embeddings -> similarity ~1.0", abs(sims[0] - 1.0) < 1e-6)
check("orthogonal embeddings -> similarity ~0.0", abs(sims[1] - 0.0) < 1e-6)
check("real SemanticSimilarityScorer.available is a property, not a constructor call", _RealSemanticSimilarityScorer.available.fget(real_scorer) is True)


# ===========================================================================
print("\n=== Test 15: round-trip diagnostic runs es->ca and reports a distribution ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {
    "interview_ca_rt": {
        "q1": "Aquesta és una resposta en català prou llarga per activar la traducció i el diagnòstic de anada i tornada.",
    },
}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, StubSimilarityScorer())
check("roundtrip sample_size > 0 when a similarity scorer is available", report["translation_quality_summary"]["roundtrip"]["sample_size"] > 0)
directions_called = {(c[0], c[1]) for c in StubNLLBTranslator.CALL_LOG}
check("both ca->es and es->ca directions were exercised", ("ca", "es") in directions_called and ("es", "ca") in directions_called)
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 16: stable IDs + duplicate-ID detection ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv1": {"q1": "Esta es una respuesta normal en español para probar el formato del identificador."}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
check("response_id format is interview_id::question_id", "iv1::q1" in responses_out)
if responses_out["iv1::q1"]["sentence_ids"]:
    check("sentence_id format is response_id::sNNN", responses_out["iv1::q1"]["sentence_ids"][0] == "iv1::q1::s000")
shutil.rmtree(cache_dir, ignore_errors=True)

# Construct a genuine response_id collision via IDs that themselves contain
# "::" -- interview_id="a::b" + question_id="c" collides with
# interview_id="a" + question_id="b::c". This exercises the FATAL
# duplicate_response_id guard and confirms it degrades to
# structural_validity=False / STEP2_VALID=False rather than crashing or
# silently dropping one of the two responses.
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
colliding_data = {
    "a::b": {"c": "Esta es una respuesta en español suficientemente larga para la prueba de colision."},
    "a": {"b::c": "Esta es otra respuesta en español suficientemente larga para la prueba de colision."},
}
responses_out, sentences_out, report = preprocess_v2.process(colliding_data, preprocessor, cache, translator, similarity_scorer)
check("a crafted response_id collision is caught as a FATAL error, not silently dropped", report["duplicate_response_id_count"] == 1)
check("structural_validity is False when a duplicate response_id is detected", report["structural_validity"] is False)
check("STEP2_VALID is False when structural_validity is False", report["STEP2_VALID"] is False)
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 17: full-corpus structural preservation (950 responses, stub NLLB) ===")
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=50)

big_corpus = {}
for i in range(190):  # 190 interviews x 5 questions = 950 responses
    interview_id = f"interview_{i:04d}"
    big_corpus[interview_id] = {}
    for j in range(5):
        question_id = f"q{j + 1}"
        if (i + j) % 2 == 0:
            text = f"Aquesta és una resposta en català número {i}-{j} prou llarga per a la detecció de l'idioma."
        else:
            text = f"Esta es una respuesta en español numero {i}-{j} suficientemente larga para la deteccion del idioma."
        big_corpus[interview_id][question_id] = text

input_count = sum(len(qs) for qs in big_corpus.values())
check("synthetic corpus has exactly 950 responses", input_count == 950)

responses_out, sentences_out, report = preprocess_v2.process(big_corpus, preprocessor, cache, translator, similarity_scorer)
check("output_response_count == input_response_count (no structural loss)", report["output_response_count"] == 950)
check("no fatal errors across the full synthetic corpus", report["fatal_error_count"] == 0)
check("no unmapped sentences across the full synthetic corpus", len(report["unmapped_sentences"]) == 0)
check("structural_validity True for the full synthetic corpus", report["structural_validity"] is True)
check("every response has a non-null text_es (stub never fails)", report["missing_text_es_count"] == 0)
big_corpus_report = report  # reused by Test 18
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 18: STEP2_VALID True case (clean run, no quality issues) ===")
check("full-corpus run above already has STEP2_VALID True", big_corpus_report["STEP2_VALID"] is True)
check("translation_sanity_status is PASS when nothing was flagged", big_corpus_report["translation_sanity_status"] == "PASS")


# ===========================================================================
print("\n=== Test 19: STEP2_VALID False case (translation failures -> completeness fails) ===")
StubNLLBTranslator.reset()
StubNLLBTranslator.FAIL_MODE = "always"
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv_fail": {"q1": "Aquest text català sempre fallarà en la traducció per a la prova de validesa."}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
check("structural_validity stays True (the response itself is not lost)", report["structural_validity"] is True)
check("translation_completeness_validity is False when a required translation fails", report["translation_completeness_validity"] is False)
check("STEP2_VALID is False when completeness fails, even with structural_validity True", report["STEP2_VALID"] is False)
check("the failed response's text_es is None, never wrong-language leftover text", responses_out["iv_fail::q1"]["text_es"] is None)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 20: diagnostic-only sanity REVIEW (length-ratio outlier) does not fail STEP2_VALID ===")
# Deliberately NOT a language mismatch or a degenerate output -- this is
# the "unusual length ratio, otherwise-correct translation" case the
# reviewer's fix #2 explicitly says must stay non-gating: a long Catalan
# source translated (by the default stub) into the same fixed, short,
# genuinely-Spanish, non-repetitive STUB_ES_OUTPUT is exactly the kind of
# real-world case (heavy compression by the model) that should surface as
# a REVIEW diagnostic, never as a reason to invalidate the whole run.
#
# Post-frozen-sentence-architecture note: translation is now one call per
# FROZEN SOURCE SENTENCE, not one call per response chunk. This text is
# deliberately built with NO internal periods (only a single trailing
# one), so StubPreprocessor's naive split-on-"." still segments it as
# exactly ONE sentence -- one NLLB call, one STUB_ES_OUTPUT. If this were
# split into multiple sentences instead, each would independently come
# back as the same fixed stub string, and the reconstructed response text
# would be STUB_ES_OUTPUT repeated N times -- itself a (correctly)
# degenerate-flagged repetition loop, which is not what this test is
# checking.
long_low_ratio_text = " ".join(
    f"aquesta es una resposta llarga en catala escrita expressament per a la prova numero {i} de la ratio de longitud"
    for i in range(4)
) + "."
check(
    "the long source segments as exactly one sentence for this test's premise",
    len(preprocess_v2.segment_source_sentences(long_low_ratio_text, "ca", preprocessor)[0]) == 1,
)
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv_review": {"q1": long_low_ratio_text}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
tqs_review = report["translation_quality_summary"]
check("translation completed (non-empty), nothing failed", report["translation_failed_count"] == 0)
check("translation_completeness_validity True (the translation itself did not fail)", report["translation_completeness_validity"] is True)
check("no language mismatch was flagged for this case", tqs_review["language_mismatch_count"] == 0)
check("no degenerate output was flagged for this case", tqs_review["degenerate_output_count"] == 0)
check("a length-ratio outlier WAS flagged (much shorter output than source)", tqs_review["length_ratio_outlier_count"] > 0)
check("translation_output_validity True (no genuinely serious failure)", report["translation_output_validity"] is True)
check("translation_sanity_status is REVIEW (diagnostic-only, non-gating)", report["translation_sanity_status"] == "REVIEW")
check("STEP2_VALID is still True -- length-ratio outliers never gate on their own", report["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 21: terminal.py stops before Stage 3 when STEP2_VALID is False ===")
import terminal  # noqa: E402

# Regression check (added after the fourth external review): merely
# `import terminal` used to eagerly drag in topic_modeling.py (bertopic/
# sentence_transformers) and sentiment_analysis.py (transformers directly)
# at MODULE level -- so this exact import, run by this exact test, is what
# reached an incompatible TensorFlow install on a real reviewer machine and
# aborted the whole test process before a single assertion below ran (see
# ml_backend.py). terminal.py now imports those three only lazily, inside
# per-component getter methods, so importing terminal itself must never
# pull any of them (or their heavy transitive dependencies) into
# sys.modules. This is what makes this reliability suite genuinely offline
# and portable, independent of whatever ML backend happens to be installed.
_heavy_modules_that_must_stay_unimported = [
    "topic_modeling", "sentiment_analysis", "visualization",
    "bertopic", "sentence_transformers", "umap", "hdbscan",
]
check(
    "import terminal does NOT eagerly import topic_modeling/sentiment_analysis/"
    "visualization or their heavy ML dependencies (critical fix)",
    all(name not in sys.modules for name in _heavy_modules_that_must_stay_unimported),
)

_tmp_output_dir = tempfile.mkdtemp(prefix="terminal_step2_invalid_")
_orig_process = preprocess_v2.process


def _always_invalid_process(*args, **kwargs):
    responses_out, sentences_out, report = _orig_process(*args, **kwargs)
    report["STEP2_VALID"] = False
    report["structural_validity"] = False
    report["translation_completeness_validity"] = False
    return responses_out, sentences_out, report


class _FakeTerminalForInvalidRun:
    """Exercises the real TerminalInterface._create_processed_versions
    logic without constructing a real TerminalInterface (whose __init__
    downloads NLTK data and loads a real Preprocessor/SentimentAnalyzer/
    Visualizer -- unnecessary weight for this unit test).
    """

    def __init__(self):
        self.preprocessor = StubPreprocessor()
        self.file_path = _tmp_output_dir
        self._nllb_translator = StubNLLBTranslator()
        self._nllb_cache = preprocess_v2.TranslationCache(os.path.join(_tmp_output_dir, "cache.json"))
        self._nllb_similarity_scorer = StubSimilarityScorer()

    _get_nllb_translator = terminal.TerminalInterface._get_nllb_translator
    _get_nllb_cache = terminal.TerminalInterface._get_nllb_cache
    _get_nllb_similarity_scorer = terminal.TerminalInterface._get_nllb_similarity_scorer
    _create_processed_versions = terminal.TerminalInterface._create_processed_versions


preprocess_v2.process = _always_invalid_process
try:
    fake_terminal = _FakeTerminalForInvalidRun()
    source_data = {"iv1": {"q1": "Aquest text serveix només per activar el pipeline en aquesta prova."}}
    result = fake_terminal._create_processed_versions(source_data)
    check("_create_processed_versions returns False when STEP2_VALID is False", result is False)
    check(
        "fully_processed_ca.json is NOT written when STEP2_VALID is False",
        not os.path.exists(os.path.join(_tmp_output_dir, "fully_processed_ca.json")),
    )
    check(
        "sentiment_input_ca.json is NOT written when STEP2_VALID is False",
        not os.path.exists(os.path.join(_tmp_output_dir, "sentiment_input_ca.json")),
    )
    check(
        "topic_modeling_input.json is NOT written when STEP2_VALID is False",
        not os.path.exists(os.path.join(_tmp_output_dir, "topic_modeling_input.json")),
    )
    check(
        "the diagnostic report IS still written even on an invalid run (no silent failure)",
        os.path.exists(os.path.join(_tmp_output_dir, "preprocessing_language_report.json")),
    )
finally:
    preprocess_v2.process = _orig_process
    shutil.rmtree(_tmp_output_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 22: terminal.py continues to Stage 3/5 input files only when STEP2_VALID is True ===")
_tmp_output_dir = tempfile.mkdtemp(prefix="terminal_step2_valid_")


class _FakeTerminalForValidRun:
    def __init__(self):
        self.preprocessor = StubPreprocessor()
        self.file_path = _tmp_output_dir
        self._nllb_translator = StubNLLBTranslator()
        self._nllb_cache = preprocess_v2.TranslationCache(os.path.join(_tmp_output_dir, "cache.json"))
        self._nllb_similarity_scorer = StubSimilarityScorer()

    _get_nllb_translator = terminal.TerminalInterface._get_nllb_translator
    _get_nllb_cache = terminal.TerminalInterface._get_nllb_cache
    _get_nllb_similarity_scorer = terminal.TerminalInterface._get_nllb_similarity_scorer
    _create_processed_versions = terminal.TerminalInterface._create_processed_versions
    _step2_bridge_is_fresh = terminal.TerminalInterface._step2_bridge_is_fresh


StubNLLBTranslator.reset()
fake_terminal = _FakeTerminalForValidRun()
source_data = {
    # No internal "." beyond the single trailing sentence-final one --
    # StubPreprocessor's naive split-on-"." would otherwise carve an
    # embedded period (e.g. inside a "terminal.py" reference) into an
    # extra spurious sentence, doubling the fixed stub output below.
    "iv1": {"q1": "Aquesta és una resposta en català prou llarga per a la prova d'integració amb aquest programa de terminal."},
    "iv2": {"q1": "Esta es una respuesta en español suficientemente larga para la prueba de integracion con terminal."},
}
result = fake_terminal._create_processed_versions(source_data)
check("_create_processed_versions returns True when STEP2_VALID is True", result is True)

fully_processed_path = os.path.join(_tmp_output_dir, "fully_processed_ca.json")
sentiment_input_path = os.path.join(_tmp_output_dir, "sentiment_input_ca.json")
topic_input_path = os.path.join(_tmp_output_dir, "topic_modeling_input.json")
check("fully_processed_ca.json IS written when STEP2_VALID is True", os.path.exists(fully_processed_path))
check("sentiment_input_ca.json IS written when STEP2_VALID is True", os.path.exists(sentiment_input_path))
check("topic_modeling_input.json IS written when STEP2_VALID is True", os.path.exists(topic_input_path))

with open(fully_processed_path, encoding="utf-8") as f:
    fully_processed_content = json.load(f)
check("fully_processed_ca.json contains both interviews", set(fully_processed_content.keys()) == {"iv1", "iv2"})
check(
    "fully_processed_ca.json's Catalan-sourced entry holds Spanish topic text, not Catalan",
    fully_processed_content["iv1"]["q1"] == STUB_ES_OUTPUT.lower().strip(),
)

with open(sentiment_input_path, encoding="utf-8") as f:
    sentiment_content = json.load(f)
check(
    "sentiment_input_ca.json's entries are lists of sentences",
    isinstance(sentiment_content["iv1"]["q1"], list) and len(sentiment_content["iv1"]["q1"]) > 0,
)

# Fallback path: source_data=None must load the corpus from
# data/input/interviews.json rather than raising a TypeError as the old
# required-argument signature would have.
import inspect  # noqa: E402
check(
    "_create_processed_versions accepts source_data=None (fixes the old fallback crash)",
    inspect.signature(terminal.TerminalInterface._create_processed_versions).parameters["source_data"].default is None,
)

metadata_path = os.path.join(_tmp_output_dir, terminal.STEP2_BRIDGE_METADATA_FILENAME)
check(
    "a validated Step 2 run writes the bridge-freshness metadata sidecar",
    os.path.exists(metadata_path),
)
with open(metadata_path, encoding="utf-8") as f:
    written_metadata = json.load(f)
check(
    "the sidecar records a real input_sha256 (reproducibility fix)",
    isinstance(written_metadata.get("input_sha256"), str) and len(written_metadata["input_sha256"]) == 64,
)
check(
    "the sidecar's input_sha256 matches hashing data/input/interviews.json directly",
    written_metadata["input_sha256"] == terminal._hash_file(terminal.STEP2_INTERVIEWS_INPUT_FILE),
)
check(
    "the sidecar records the NLLB model name and generation version for auditability",
    written_metadata.get("nllb_model_name") == preprocess_v2.NLLB_MODEL_NAME
    and written_metadata.get("nllb_generation_version") == preprocess_v2.GENERATION_VERSION,
)
check(
    "_step2_bridge_is_fresh() is True right after a validated run wrote it",
    fake_terminal._step2_bridge_is_fresh() is True,
)
shutil.rmtree(_tmp_output_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 23: stale/historical bridge files cannot bypass Step 2 (critical fix) ===")
_tmp_output_dir = tempfile.mkdtemp(prefix="terminal_stale_bridge_")


class _FakeTerminalForStaleness:
    def __init__(self):
        self.preprocessor = StubPreprocessor()
        self.file_path = _tmp_output_dir

    _step2_bridge_is_fresh = terminal.TerminalInterface._step2_bridge_is_fresh


staleness_terminal = _FakeTerminalForStaleness()

# Scenario A: nothing on disk at all.
check(
    "no bridge file, no metadata -> not fresh (Step 2 must run)",
    staleness_terminal._step2_bridge_is_fresh() is False,
)

# Scenario B: the exact failure mode this fix targets -- a bridge file
# (e.g. topic_modeling_input.json, left over from a previous pipeline
# version) exists on disk, but with NO metadata sidecar at all. The old
# `if not os.path.exists(topic_modeling_input.json)` gate would have
# accepted this silently and skipped Step 2 entirely.
with open(os.path.join(_tmp_output_dir, "topic_modeling_input.json"), "w", encoding="utf-8") as f:
    json.dump({"documents": [{"text": "old Catalan-era content", "metadata": {}}]}, f)
check(
    "a stale bridge file with NO metadata sidecar is NOT treated as fresh",
    staleness_terminal._step2_bridge_is_fresh() is False,
)

# Scenario C: a metadata sidecar exists but says the run was invalid --
# must still not be treated as fresh.
with open(metadata_path_placeholder := os.path.join(_tmp_output_dir, terminal.STEP2_BRIDGE_METADATA_FILENAME), "w", encoding="utf-8") as f:
    json.dump({"schema_version": terminal.STEP2_BRIDGE_SCHEMA_VERSION, "step2_valid": False}, f)
check(
    "a metadata sidecar with step2_valid=False is NOT treated as fresh",
    staleness_terminal._step2_bridge_is_fresh() is False,
)

# Scenario D: a metadata sidecar from an OLDER/different bridge schema --
# must not be treated as fresh even if it happens to say step2_valid=True.
# (This also covers a real v1-era sidecar predating the input-hash field:
# it necessarily has a different/older schema_version, so it correctly
# fails here without needing a separate no-hash-field scenario.)
with open(metadata_path_placeholder, "w", encoding="utf-8") as f:
    json.dump({"schema_version": "some_other_older_schema", "step2_valid": True}, f)
check(
    "a metadata sidecar with a mismatched schema_version is NOT treated as fresh",
    staleness_terminal._step2_bridge_is_fresh() is False,
)

# Scenario E: current schema + step2_valid True, but NO input_sha256 at all
# -- must not be treated as fresh. Guards against a sidecar that somehow
# has the current schema_version but was never given a hash to check
# (defensive: should not happen from _create_processed_versions() itself,
# but _step2_bridge_is_fresh() must not silently trust a hash it never
# received).
with open(metadata_path_placeholder, "w", encoding="utf-8") as f:
    json.dump({"schema_version": terminal.STEP2_BRIDGE_SCHEMA_VERSION, "step2_valid": True}, f)
check(
    "a current-schema, step2_valid=True sidecar with NO input_sha256 is NOT treated as fresh",
    staleness_terminal._step2_bridge_is_fresh() is False,
)

# Scenario F: current schema, step2_valid True, but input_sha256 does NOT
# match the current data/input/interviews.json -- the reproducibility fix
# from the third external review: the corpus changed since this sidecar
# was written, so it must not be treated as fresh even though everything
# else about it looks valid.
with open(metadata_path_placeholder, "w", encoding="utf-8") as f:
    json.dump({
        "schema_version": terminal.STEP2_BRIDGE_SCHEMA_VERSION,
        "step2_valid": True,
        "input_sha256": "0" * 64,  # deliberately wrong
    }, f)
check(
    "a sidecar whose input_sha256 does NOT match the current corpus is NOT treated as fresh",
    staleness_terminal._step2_bridge_is_fresh() is False,
)

# Scenario G: the real thing -- current schema, step2_valid True, and the
# ACTUAL current hash of data/input/interviews.json (tests run with cwd
# at the project root, so this is the same file _step2_bridge_is_fresh()
# itself would hash).
real_input_hash = terminal._hash_file(terminal.STEP2_INTERVIEWS_INPUT_FILE)
check("the real interviews.json is hashable in this test environment", real_input_hash is not None)
with open(metadata_path_placeholder, "w", encoding="utf-8") as f:
    json.dump({
        "schema_version": terminal.STEP2_BRIDGE_SCHEMA_VERSION,
        "step2_valid": True,
        "input_sha256": real_input_hash,
    }, f)
check(
    "a current-schema, step2_valid=True sidecar with the MATCHING current input hash IS treated as fresh",
    staleness_terminal._step2_bridge_is_fresh() is True,
)
shutil.rmtree(_tmp_output_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 24: SentimentAnalyzer (BART/nlptown) is lazy, not loaded at construction ===")


class StubSentimentAnalyzer:
    INSTANTIATION_COUNT = 0

    def __init__(self):
        StubSentimentAnalyzer.INSTANTIATION_COUNT += 1

    def analyze(self, sentence):
        return {"score": 0.5, "confusion": None}


# NOTE (fixed after the fourth external review): _get_analyzer() now does
# `from sentiment_analysis import SentimentAnalyzer` INSIDE the method body
# (a genuinely lazy IMPORT, not just lazy construction -- see terminal.py),
# so terminal.py no longer has a module-level `SentimentAnalyzer` name at
# all. Patching `terminal.SentimentAnalyzer` (the old approach) would
# silently fail to intercept anything: the `from sentiment_analysis import
# SentimentAnalyzer` statement resolves the name from the sentiment_analysis
# module's own namespace, not terminal's. Patch it there instead, so this
# test still never constructs (or downloads) the real BART/nlptown models.
import sentiment_analysis  # noqa: E402
_real_sentiment_analyzer_cls = sentiment_analysis.SentimentAnalyzer
sentiment_analysis.SentimentAnalyzer = StubSentimentAnalyzer


class _FakeTerminalForLazyAnalyzer:
    def __init__(self):
        self.analyzer = None

    _get_analyzer = terminal.TerminalInterface._get_analyzer


try:
    lazy_terminal = _FakeTerminalForLazyAnalyzer()
    check("analyzer starts as None (not constructed eagerly)", lazy_terminal.analyzer is None)
    first = lazy_terminal._get_analyzer()
    check("first _get_analyzer() call constructs it", isinstance(first, StubSentimentAnalyzer))
    second = lazy_terminal._get_analyzer()
    check("a second call reuses the same instance (not reloaded)", second is first)
    check("SentimentAnalyzer was only ever instantiated once", StubSentimentAnalyzer.INSTANTIATION_COUNT == 1)
    check(
        "the stub intercepted construction -- the real BART/nlptown SentimentAnalyzer "
        "was never instantiated by this test",
        not isinstance(first, _real_sentiment_analyzer_cls) if isinstance(_real_sentiment_analyzer_cls, type) else True,
    )
finally:
    # Restore the real class on the module so later code (none currently
    # follows in this file, but future tests might) never observes the stub.
    sentiment_analysis.SentimentAnalyzer = _real_sentiment_analyzer_cls


# ===========================================================================
print("\n=== Test 25: boundary-aware chunking splits long responses at sentence boundaries ===")


class _RealSentenceSplitter:
    """A closer-to-real sentence splitter than StubPreprocessor's naive
    split-on-period, for exercising _split_into_translation_chunks with
    sentences that look like real interview text (multiple sentences per
    period, no ambiguity about boundaries for this test).
    """

    def split_sentences_strict(self, text, lang):
        return [s.strip() for s in text.split(". ") if s.strip()]

    def full_preprocess_v2(self, text, lang):
        return text.lower().strip()


splitter = _RealSentenceSplitter()

short_text = "Una frase curta que hi cap en un sol tros sense cap problema."
chunks, hard_splits = preprocess_v2._split_into_translation_chunks(short_text, splitter, "ca")
check("a short text produces exactly one chunk", len(chunks) == 1)
check("no hard splits needed for a short text", hard_splits == 0)

# 40 short sentences (~8 words each = ~320 words total) forces multiple
# chunks under the 150-word-per-chunk budget, without needing any single
# sentence anywhere near that limit.
long_sentences = [f"Aquesta es la frase numero {i} del text molt llarg de prova." for i in range(40)]
long_text = " ".join(long_sentences)
chunks, hard_splits = preprocess_v2._split_into_translation_chunks(long_text, splitter, "ca")
check("a long multi-sentence text is split into more than one chunk", len(chunks) > 1)
check("no chunk exceeds the per-chunk word budget", all(len(c.split()) <= preprocess_v2.MAX_TRANSLATE_CHUNK_WORDS for c in chunks))
check("no hard splits needed when every sentence is short", hard_splits == 0)


def _normalize_words(text):
    # Strips sentence-final punctuation before comparing word content --
    # _RealSentenceSplitter splits on ". ", which consumes the separating
    # period between sentences, so a naive word-for-word (including
    # punctuation) comparison would flag lost trailing periods as if they
    # were lost WORDS. This test cares about content preservation, not
    # punctuation reproduction.
    return [w.strip(".,;:!?") for w in text.split()]


check(
    "every word of the source text is preserved across the chunks (nothing dropped)",
    _normalize_words(" ".join(chunks)) == _normalize_words(long_text),
)

# A single pathological sentence far longer than the whole chunk budget --
# must trigger the last-resort hard split rather than ever being handed to
# NLLB as one oversized piece.
huge_sentence = " ".join(f"paraula{i}" for i in range(500))
chunks, hard_splits = preprocess_v2._split_into_translation_chunks(huge_sentence, splitter, "ca")
check("an oversized single sentence triggers at least one hard split", hard_splits > 0)
check("every hard-split piece stays within the word budget", all(len(c.split()) <= preprocess_v2.MAX_TRANSLATE_CHUNK_WORDS for c in chunks))
check(
    "hard-splitting still preserves every word of the oversized sentence",
    " ".join(chunks).split() == huge_sentence.split(),
)


# ===========================================================================
print("\n=== Test 26: a long response is fully translated across chunks, nothing truncated (critical fix) ===")


class StubEchoNLLBTranslator(StubNLLBTranslator):
    """Echoes each input chunk back verbatim (tagged by target language),
    rather than a fixed sentence -- lets this test verify that EVERY
    sentence of a long source response survives translation and
    rejoining, not just that some output was produced.
    """

    ECHO_CALL_LOG = []

    def translate_batch(self, texts, source_lang, target_lang):
        StubEchoNLLBTranslator.ECHO_CALL_LOG.append((source_lang, target_lang, list(texts)))
        return [f"[{target_lang}]{t}" for t in texts]

    @classmethod
    def reset(cls):
        super().reset()
        cls.ECHO_CALL_LOG = []


StubEchoNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
echo_translator = StubEchoNLLBTranslator(batch_size=50)

# A ~700-word Catalan response (comparable to the longest real responses
# in data/input/interviews.json, ~740 words) made of uniquely-markable
# sentences so truncation would be directly detectable.
long_sentences = [f"Aquesta es la frase numero {i} amb un marcador unic MARCADOR{i:03d} per a la prova." for i in range(70)]
long_response_text = " ".join(long_sentences)
word_count = len(long_response_text.split())
check("the synthetic long response is realistically long (comparable to the corpus max)", word_count > 500)

raw_data = {"iv_long": {"q1": long_response_text}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, splitter, cache, echo_translator, similarity_scorer)
response = responses_out["iv_long::q1"]

# Post-frozen-sentence-architecture note: translation is now one call per
# sentence (batched across sentences, not chunked within a response), so
# the "more than one chunk" invariant becomes "more than one sentence" /
# "more than one translate_batch call" -- same underlying guarantee this
# test targets (a long response's translation isn't silently truncated
# partway through).
check("a long response segmented into more than one sentence", response["sentence_count"] > 1)
check(
    "translating that many sentences required more than one batch call",
    len(StubEchoNLLBTranslator.ECHO_CALL_LOG) > 1,
)
check("text_es_status is TRANSLATED for the long response", response["text_es_status"] == "TRANSLATED")
missing_markers = [i for i in range(70) if f"MARCADOR{i:03d}" not in (response["text_es"] or "")]
check("every sentence marker from the 70-sentence response survives translation (no silent truncation)", missing_markers == [])
check(
    "translation_completeness_validity True -- nothing failed or went missing across chunks",
    report["translation_completeness_validity"] is True,
)
# NOT asserting STEP2_VALID here: StubEchoNLLBTranslator echoes the
# CATALAN source back (tagged "[es]") rather than producing real Spanish
# output, purely so this test can verify exact content survives chunking
# and rejoining -- so, correctly, the real check_target_language() flags
# it as a language mismatch (it genuinely isn't Spanish). Through Round
# 10 that made translation_output_validity (and therefore STEP2_VALID)
# False on its own; Round 11 demoted language mismatch to REVIEW-only at
# every length (see Test 33/60/61), so this no longer affects STEP2_VALID
# by itself -- it is still correctly detected and reported, which is all
# this check verifies. STEP2_VALID's dependence on genuinely non-
# degenerate output is covered on its own terms by Test 34.
check(
    "the echo stub's non-Spanish output is correctly flagged as a language mismatch (still detected/reported, not a truncation bug)",
    report["translation_quality_summary"]["language_mismatch_count"] > 0,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 27: evenly-spaced round-trip sampling (not just the first N in corpus order) ===")
sample = preprocess_v2._evenly_spaced_sample([f"id{i}" for i in range(100)], 10)
check("evenly-spaced sample of 10 from 100 returns 10 items", len(sample) == 10)
check("evenly-spaced sample is not just the first N in corpus order", sample != [f"id{i}" for i in range(10)])
check("evenly-spaced sample spans close to the full range, not just the front", int(sample[-1][2:]) >= 80)

full = preprocess_v2._evenly_spaced_sample(["a", "b", "c"], 10)
check("asking for more samples than available items returns everything", full == ["a", "b", "c"])

none_sample = preprocess_v2._evenly_spaced_sample(["a", "b", "c"], 0)
check("a sample size of 0 returns nothing", none_sample == [])


# ===========================================================================
print("\n=== Test 28: semantic-preservation similarity is computed in one batched call, not per-response ===")


class CountingSimilarityScorer(StubSimilarityScorer):
    CALL_COUNT = 0

    def batch_cosine_similarity(self, texts_a, texts_b):
        CountingSimilarityScorer.CALL_COUNT += 1
        return super().batch_cosine_similarity(texts_a, texts_b)


StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
counting_scorer = CountingSimilarityScorer()
raw_data = {
    f"iv_batch_{i}": {"q1": f"Aquesta és una resposta prou llarga en català per a la prova de traducció número {i}."}
    for i in range(5)
}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, counting_scorer)
translated_count = sum(1 for r in responses_out.values() if r["translation_required"])
check("5 Catalan responses were translated", translated_count == 5)
# Exactly 2 calls total: one batched call covering all 5 responses'
# semantic-preservation similarity, and one separate batched call for the
# round-trip diagnostic (a distinct computation, sampled independently) --
# never one call per response (which would be 5, or 10 counting both
# diagnostics).
check(
    "semantic-preservation similarity uses one batched call for all 5 responses, not one per response",
    CountingSimilarityScorer.CALL_COUNT == 2,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 29: count_tokens / _assert_within_token_limit (real methods, no model load) ===")


class _FakeTokenizerForCounting:
    """Duck-types just enough of the real NLLB tokenizer's interface for
    count_tokens(): assigning .src_lang, and being callable with a single
    text argument to return a dict with "input_ids". Token count is
    derived deterministically from the text so tests can control it
    precisely without a real SentencePiece tokenizer.
    """

    def __init__(self, tokens_per_word=1, extra_special_tokens=2):
        self.src_lang = None
        self.tokens_per_word = tokens_per_word
        self.extra_special_tokens = extra_special_tokens

    def __call__(self, text):
        n_words = len(text.split())
        n_tokens = n_words * self.tokens_per_word + self.extra_special_tokens
        return {"input_ids": list(range(n_tokens))}


class _FakeSelfForTokenCounting:
    """Duck-typed fake `self` for exercising NLLBTranslator.count_tokens
    and NLLBTranslator._assert_within_token_limit directly, as real
    (unmodified) methods, without ever constructing a real NLLBTranslator
    (whose __init__ downloads/loads the real model)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # _assert_within_token_limit calls self.count_tokens(...) -- bind the
    # REAL, unmodified NLLBTranslator.count_tokens onto this fake self so
    # both methods under test are exercised as their actual implementation,
    # not reimplemented here under a different name.
    count_tokens = _RealNLLBTranslator.count_tokens


fake_tokenizer = _FakeTokenizerForCounting(tokens_per_word=1, extra_special_tokens=2)
fake_self = _FakeSelfForTokenCounting(fake_tokenizer)

check(
    "count_tokens sets tokenizer.src_lang from the NLLB language code",
    (_RealNLLBTranslator.count_tokens(fake_self, "hola que tal", "es"), fake_tokenizer.src_lang) == (5, "spa_Latn"),
)
check(
    "count_tokens returns len(input_ids), including special tokens",
    _RealNLLBTranslator.count_tokens(fake_self, "one two three four", "ca") == 4 + 2,
)

# _assert_within_token_limit: within-limit texts must not raise.
try:
    _RealNLLBTranslator._assert_within_token_limit(fake_self, ["short text here"], "es")
    within_limit_raised = False
except preprocess_v2.NLLBTranslationError:
    within_limit_raised = True
check("_assert_within_token_limit does not raise when every text is within the token limit", not within_limit_raised)

# Force an oversized text: GENERATION_PARAMS["max_length"] tokens is the
# limit: 1 token/word + 2 special tokens, so 511 words -> 513 tokens, over
# the 512 limit.
oversized_text = " ".join(["palabra"] * 511)
try:
    _RealNLLBTranslator._assert_within_token_limit(fake_self, ["short text", oversized_text], "es")
    oversized_raised = False
    oversized_error_message = None
except preprocess_v2.NLLBTranslationError as exc:
    oversized_raised = True
    oversized_error_message = str(exc)
check("_assert_within_token_limit raises NLLBTranslationError when a text exceeds the token limit", oversized_raised)
check(
    "the raised error names the actual oversized token count, not just a generic message",
    oversized_error_message is not None and "513" in oversized_error_message,
)


# ===========================================================================
print("\n=== Test 30: _enforce_real_token_budget (tokenizer-verified second pass) ===")


def _words_over_threshold_counter(threshold):
    # A fake token_counter reporting an inflated count (10x the word
    # count) ONLY for chunks longer than `threshold` words -- lets tests
    # force specific chunks over the limit without a real tokenizer.
    def _count(text):
        n_words = len(text.split())
        return n_words * 10 if n_words > threshold else n_words
    return _count

# A single chunk of 20 words, reported as over-budget (200 "tokens") by
# the fake counter -- must be recursively bisected until every resulting
# piece is at or under the 50-token limit.
oversized_chunk = " ".join(f"word{i}" for i in range(20))
safe_chunks, extra_splits = preprocess_v2._enforce_real_token_budget(
    [oversized_chunk], _words_over_threshold_counter(threshold=5), token_limit=50,
)
check("an over-budget chunk is split into more than one piece", len(safe_chunks) > 1)
check("every resulting piece is at or under the token limit per the real counter", all(
    _words_over_threshold_counter(threshold=5)(c) <= 50 for c in safe_chunks
))
check("splitting the oversized chunk required at least one extra split", extra_splits > 0)
check(
    "every word of the oversized chunk is preserved across the re-split pieces",
    " ".join(safe_chunks).split() == oversized_chunk.split(),
)

# A chunk already within budget must pass through unchanged, with zero
# extra splits -- the whole point of this being a SECOND pass, not a
# replacement for the word-heuristic first pass. Kept at or under the
# counter's own threshold (5 words) so it is never inflated.
fine_chunk = "una frase curta"
safe_chunks, extra_splits = preprocess_v2._enforce_real_token_budget(
    [fine_chunk], _words_over_threshold_counter(threshold=5), token_limit=50,
)
check("a chunk already within budget passes through unchanged", safe_chunks == [fine_chunk])
check("no extra splits are counted for a chunk already within budget", extra_splits == 0)

# Pathological single "word" alone over budget: nothing left to split on,
# must be handed through as-is rather than looping forever or crashing --
# NLLBTranslator._assert_within_token_limit is what refuses it downstream.
huge_single_token = "x" * 5000
safe_chunks, extra_splits = preprocess_v2._enforce_real_token_budget(
    [huge_single_token], lambda t: 99999, token_limit=50,
)
check("a single unsplittable oversized 'word' is passed through rather than looping forever", safe_chunks == [huge_single_token])


# ===========================================================================
print("\n=== Test 31: _split_into_translation_chunks engages the tokenizer-verified second pass ===")

# 30 short words -- well under MAX_TRANSLATE_CHUNK_WORDS (150), so the
# word-count heuristic alone produces exactly one chunk. A fake
# token_counter that reports this single chunk as over a tight token_limit
# forces the second pass to re-split it -- proving the tokenizer-verified
# pass is a real, additional safety net, not a no-op when wired in.
moderate_text = " ".join(f"paraula{i}" for i in range(30))
chunks_no_counter, hard_splits_no_counter = preprocess_v2._split_into_translation_chunks(
    moderate_text, splitter, "ca",
)
check("without a token_counter, the word heuristic alone produces one chunk", len(chunks_no_counter) == 1)
check("without a token_counter, no hard splits are counted", hard_splits_no_counter == 0)

chunks_with_counter, hard_splits_with_counter = preprocess_v2._split_into_translation_chunks(
    moderate_text, splitter, "ca",
    token_counter=lambda t: len(t.split()) * 10,  # inflate token count 10x
    token_limit=100,
)
check("with a strict token_counter, the same text is split into more than one chunk", len(chunks_with_counter) > 1)
check("the tokenizer-verified second pass counts as hard splits", hard_splits_with_counter > 0)
check(
    "every word is still preserved once the second pass re-splits the chunk",
    " ".join(chunks_with_counter).split() == moderate_text.split(),
)


# ===========================================================================
print("\n=== Test 32: translate_responses_batched wires a real translator's count_tokens into chunking ===")


class StubTokenAwareNLLBTranslator(StubNLLBTranslator):
    """A StubNLLBTranslator that additionally exposes count_tokens(), so
    translate_responses_batched's hasattr(translator, "count_tokens")
    check picks it up and engages the tokenizer-verified second pass --
    without ever loading a real tokenizer/model.
    """

    def count_tokens(self, text, source_lang):
        # Deliberately inflate the count so a moderate response that the
        # word heuristic alone would keep as one chunk gets forced into
        # more than one -- proof the callback is actually being used, not
        # just accepted and ignored. 60 words * 10 = 600, over
        # GENERATION_PARAMS["max_length"] (512), while still well under
        # MAX_TRANSLATE_CHUNK_WORDS (150) so the word heuristic alone
        # would never have split it.
        return len(text.split()) * 10


StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
token_aware_translator = StubTokenAwareNLLBTranslator()

moderate_ca_text = " ".join(f"paraula{i}" for i in range(60)) + "."
results = preprocess_v2.translate_responses_batched(
    [moderate_ca_text], "ca", "es", splitter, token_aware_translator, cache,
    preprocess_v2.ConsecutiveBatchFailureTracker(),
)
_, status, chunk_count, hard_split_count = results[0]
check("a translator exposing count_tokens causes more chunks than the word heuristic alone would produce", chunk_count > 1)
check("the extra chunking from the tokenizer-verified pass is counted as hard splits", hard_split_count > 0)
check("the response was still fully translated despite the forced re-chunking", status == "TRANSLATED")
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()

plain_translator_without_count_tokens = StubNLLBTranslator()
check(
    "a plain StubNLLBTranslator (no count_tokens) is unaffected -- hasattr guard keeps old tests' behavior unchanged",
    not hasattr(plain_translator_without_count_tokens, "count_tokens"),
)


# ===========================================================================
print("\n=== Test 33: a language-mismatch-only failure no longer prevents STEP2_VALID=True (Round 11) ===")
# The translation itself "succeeds" (non-empty, not a retry failure) but
# lands in the wrong language entirely -- STUB_EN_OUTPUT is a genuinely
# fluent, varied English sentence, so it is NOT flagged as degenerate
# (empty/identical/repetitive); the only thing wrong with it is that it
# isn't Spanish. Through Round 10 this was fatal on its own (the
# reviewer's original fix #2). Round 11 reverses that: a manual review of
# the first real 950-response corpus run's 141 language-mismatch flags
# found every one was either a langdetect false positive on genuinely
# correct Spanish, or already independently caught by check_degenerate_
# output -- language detection alone never caught a real failure
# check_degenerate_output didn't already catch. So this is now REVIEW-
# only, at every length, never FAIL.
StubNLLBTranslator.FAIL_MODE = "wrong_language_one"
mismatch_text = "Aquest text es tradueix accidentalment a un idioma que no toca en aquesta prova"
StubNLLBTranslator.WRONG_LANGUAGE_FOR_TEXTS = frozenset([mismatch_text])
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv_mismatch": {"q1": mismatch_text}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
tqs_mismatch = report["translation_quality_summary"]
check("translation completed (non-empty), nothing failed at the completeness level", report["translation_failed_count"] == 0)
check("translation_completeness_validity True (the translation itself did not fail)", report["translation_completeness_validity"] is True)
check("a language mismatch WAS flagged (still fully computed/reported)", tqs_mismatch["language_mismatch_count"] > 0)
check("no degenerate output was flagged for this case (fluent, varied, non-empty text)", tqs_mismatch["degenerate_output_count"] == 0)
check("the sentence itself is NOT marked FLAGGED -- language mismatch alone no longer flags a sentence", report["sentence_translation_summary"]["flagged_sentence_count"] == 0)
check(
    "...but IS counted in the new sentence-level language-mismatch rollup (still visible, just not fatal)",
    report["sentence_translation_summary"]["sentence_language_mismatch_count"] == 1,
)
check("translation_output_validity is True (language mismatch alone is no longer fatal)", report["translation_output_validity"] is True)
check("translation_sanity_status is REVIEW, not FAIL -- surfaced for a human, never blocking on its own", report["translation_sanity_status"] == "REVIEW")
check("STEP2_VALID is True", report["STEP2_VALID"] is True)
check("structural_validity stays True (the response itself was not lost)", report["structural_validity"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 34: a degenerate-output-only failure now prevents STEP2_VALID=True ===")
# STUB_REPETITIVE_OUTPUT is confirmed (Test 11, and via real langdetect
# above) to be detected as genuinely Spanish -- so this case isolates
# degeneracy (n-gram repetition) from language mismatch, the mirror image
# of Test 33.
#
# Round 12: marked REPETITIVE_STAYS_BAD_ON_RETRY -- this test specifically
# needs the new repetition-fallback-retry mechanism (Test 63/64) to NOT
# quietly fix this case, so it keeps proving degenerate output is fatal
# even when a repair attempt is made and doesn't help. Test 63 covers the
# "retry resolves it" path on its own terms.
StubNLLBTranslator.FAIL_MODE = "repetition_one"
degenerate_text = "Aquest text es tradueix de manera repetitiva per accident en aquesta prova"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([degenerate_text])
StubNLLBTranslator.REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS = frozenset([degenerate_text])
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv_degenerate": {"q1": degenerate_text}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
tqs_degenerate = report["translation_quality_summary"]
check("translation completed (non-empty), nothing failed at the completeness level", report["translation_failed_count"] == 0)
check("translation_completeness_validity True (the translation itself did not fail)", report["translation_completeness_validity"] is True)
check("no language mismatch was flagged for this case (output is genuinely Spanish)", tqs_degenerate["language_mismatch_count"] == 0)
check("a degenerate (repetitive) output WAS flagged", tqs_degenerate["degenerate_output_count"] > 0)
check("a fallback retry WAS attempted, and correctly did not resolve it", report["sentence_translation_summary"]["repetition_fallback_retry_unresolved_count"] == 1)
check("translation_output_validity is False (degenerate output alone is fatal, even after an unsuccessful repair attempt)", report["translation_output_validity"] is False)
check("translation_sanity_status is FAIL, not just REVIEW, for a genuinely serious failure", report["translation_sanity_status"] == "FAIL")
check("STEP2_VALID is False even though structural_validity and completeness are both True", report["STEP2_VALID"] is False)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 35: terminal._hash_file (input reproducibility helper) ===")
_hash_tmp_dir = tempfile.mkdtemp(prefix="hash_file_test_")
_hash_file_path = os.path.join(_hash_tmp_dir, "sample.json")
with open(_hash_file_path, "w", encoding="utf-8") as f:
    f.write('{"a": 1}')
hash_1 = terminal._hash_file(_hash_file_path)
check("_hash_file returns a 64-char hex SHA-256 digest for an existing file", isinstance(hash_1, str) and len(hash_1) == 64)

hash_2 = terminal._hash_file(_hash_file_path)
check("_hash_file is deterministic -- hashing the same unchanged file twice matches", hash_1 == hash_2)

with open(_hash_file_path, "w", encoding="utf-8") as f:
    f.write('{"a": 2}')
hash_3 = terminal._hash_file(_hash_file_path)
check("_hash_file changes when the file's content changes", hash_3 != hash_1)

check(
    "_hash_file returns None (never raises) for a missing file",
    terminal._hash_file(os.path.join(_hash_tmp_dir, "does_not_exist.json")) is None,
)
shutil.rmtree(_hash_tmp_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 36: short/ambiguous translations no longer produce false-positive STEP2_VALID failures ===")
# The corpus's real short responses ("Sí.", "No.", "PC.", "2009", ...) can
# legitimately translate to byte-identical Catalan/Spanish output. Before
# this fix, that alone would have been flagged as a fatal degenerate
# output and invalidated STEP2_VALID -- exactly the false-positive the
# third external review caught. No trailing period, deliberately (see
# Test 20's note on StubPreprocessor's naive splitter stripping it).
StubNLLBTranslator.FAIL_MODE = "identical_one"
short_identical_text = "Sí"
StubNLLBTranslator.IDENTICAL_FOR_TEXTS = frozenset([short_identical_text])
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv_short": {"q1": short_identical_text}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
tqs_short = report["translation_quality_summary"]
short_response = responses_out["iv_short::q1"]

check("the short response's translation still completed (non-empty)", report["translation_failed_count"] == 0)
check(
    "the short identical output is NOT counted as a language mismatch (not assessed)",
    short_response["quality"]["target_language_check"]["status"] == "NOT_ASSESSED_SHORT_TEXT",
)
check("no fatal language mismatch was recorded for this corpus", tqs_short["language_mismatch_count"] == 0)
check(
    "the short identical output is reported informationally, not as fatal degenerate output",
    tqs_short["short_text_identical_output_count"] == 1 and tqs_short["degenerate_output_count"] == 0,
)
check("translation_output_validity True (no genuinely serious failure)", report["translation_output_validity"] is True)
check(
    "translation_sanity_status is REVIEW (diagnostic-only short-text signal), not FAIL",
    report["translation_sanity_status"] == "REVIEW",
)
check(
    "STEP2_VALID is True -- a correct short translation is no longer a false-positive failure",
    report["STEP2_VALID"] is True,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 37: topic_text_es_clean_empty is aggregated in the report (count reconciliation) ===")


class _StopwordOnlyPreprocessor(StubPreprocessor):
    """Simulates a response whose cleaned topic text becomes empty after
    stopword/lemmatization removal (e.g. a response consisting only of
    stopwords) -- full_preprocess_v2() here deliberately returns "" for
    one specific marker text, everything else passes through unchanged.
    """

    EMPTY_AFTER_CLEANING_FOR_TEXTS = frozenset()

    def full_preprocess_v2(self, text, lang):
        if text in self.EMPTY_AFTER_CLEANING_FOR_TEXTS:
            return ""
        return text.lower().strip()


stopword_only_preprocessor = _StopwordOnlyPreprocessor()
# Matched against the RAW text_es passed into full_preprocess_v2() (i.e.
# STUB_ES_OUTPUT itself, before this stub's own lowering/stripping) --
# iv_a's Catalan response translates to exactly STUB_ES_OUTPUT via the
# default StubNLLBTranslator below.
stopword_only_preprocessor.EMPTY_AFTER_CLEANING_FOR_TEXTS = frozenset([STUB_ES_OUTPUT])
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {
    "iv_a": {"q1": "Aquesta és una resposta prou llarga en català per a la prova de neteja de text de temes."},
    "iv_b": {"q1": "Esta es una respuesta en español que produce un texto de tema limpio y no vacio para esta prueba."},
}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, stopword_only_preprocessor, cache, translator, similarity_scorer)
check(
    "exactly one response's topic_text_es_clean became empty after cleaning",
    report["topic_text_es_clean_empty_count"] == 1,
)
check(
    "the empty one is identified by response ID in the report",
    report["topic_text_es_clean_empty_ids"] == ["iv_a::q1"],
)
check(
    "expected_topic_model_document_count excludes exactly the empty-after-cleaning response",
    report["expected_topic_model_document_count"] == len(responses_out) - 1,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 38: the on-disk cache is created and updated incrementally during a run ===")
# Fifth external review: a real multi-hour run only saved the cache once, at
# the very end -- an interruption partway through lost everything translated
# so far. translate_texts_batched now calls cache.save() after every batch
# that reaches the model, not just once at the end.
cache, cache_dir = fresh_cache()
save_calls = {"n": 0}
_real_save = cache.save


def _counting_save():
    save_calls["n"] += 1
    return _real_save()


cache.save = _counting_save
translator = StubNLLBTranslator(batch_size=1)  # batch_size=1 -> one save per text
texts = [f"Text number {i} for incremental cache saving." for i in range(5)]
preprocess_v2.translate_texts_batched(texts, "ca", "es", translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker())
check("cache.save() was called more than once across a multi-batch run (incremental, not end-of-run-only)", save_calls["n"] >= 5)
check("the cache file actually exists on disk after the run", os.path.exists(cache.cache_path))
with open(cache.cache_path, encoding="utf-8") as f:
    saved_cache_contents = json.load(f)
check("every translated text landed in the on-disk cache file, not just in memory", len(saved_cache_contents) == len(texts))
StubNLLBTranslator.reset()
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 39: an interrupted run resumes from the cache without re-sending cached items to NLLB ===")
cache_dir = tempfile.mkdtemp(prefix="nllb_test_cache_")
cache_path = os.path.join(cache_dir, "cache.json")

# "Run 1": everything is a fresh translation, written to disk as it goes.
cache_run1 = preprocess_v2.TranslationCache(cache_path)
translator_run1 = StubNLLBTranslator(batch_size=2)
resume_texts = [f"Interrupted-run text {i}." for i in range(6)]
results_run1 = preprocess_v2.translate_texts_batched(
    resume_texts, "ca", "es", translator_run1, cache_run1, preprocess_v2.ConsecutiveBatchFailureTracker(),
)
check("run 1: every text was freshly translated (nothing cached yet)", all(status == "TRANSLATED" for _, status in results_run1))
check("run 1: the translator was actually called for these texts", len(StubNLLBTranslator.CALL_LOG) > 0)

# Simulate the process being killed and restarted: a brand-new
# TranslationCache instance reading the SAME on-disk file, and a brand-new
# translator instance (as a real restart would construct) whose call log we
# can inspect to prove it was never invoked for these already-cached texts.
StubNLLBTranslator.CALL_LOG = []
cache_run2 = preprocess_v2.TranslationCache(cache_path)
translator_run2 = StubNLLBTranslator(batch_size=2)
results_run2 = preprocess_v2.translate_texts_batched(
    resume_texts, "ca", "es", translator_run2, cache_run2, preprocess_v2.ConsecutiveBatchFailureTracker(),
)
check(
    "run 2 (simulated restart): every text is loaded as CACHE_HIT, not re-translated",
    all(status == "CACHE_HIT" for _, status in results_run2),
)
check(
    "run 2's translated text matches run 1's exactly (a real resume, not a coincidence)",
    [t for t, _ in results_run2] == [t for t, _ in results_run1],
)
check(
    "cached translations are NOT sent to NLLB again on resume -- translate_batch was never called in run 2",
    len(StubNLLBTranslator.CALL_LOG) == 0,
)
StubNLLBTranslator.reset()
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 40: batch_size=2 (the new MPS/low-memory default) produces the same deterministic output as batch_size=8 ===")
# Reducing batch_size only changes how many model calls are made, never the
# NLLB model, decoding parameters, chunking, or translation output --
# exactly the fifth review's "runtime/memory change only" constraint.
#
# Deliberately built as ONE sentence per response (no internal periods, a
# single trailing one), not 40 separately-punctuated sentences: the
# default StubNLLBTranslator returns the same fixed STUB_ES_OUTPUT for
# every input regardless of content, so 40 independently-translated
# sentences in one response would legitimately (and correctly) trip the
# response-level repetition/degenerate-output diagnostic -- this test is
# about batch-size determinism, not translation quality, and batching
# across the 3 separate responses is still enough to exercise different
# batch counts at batch_size=2 vs. batch_size=8.
long_response = " ".join(f"Aquesta és la frase numero {i} d'una resposta llarga" for i in range(40)) + "."
resume_source = {"iv1": {"q1": long_response}, "iv2": {"q1": long_response}, "iv3": {"q1": long_response}}

StubNLLBTranslator.reset()
cache_b8, cache_dir_b8 = fresh_cache()
translator_b8 = StubNLLBTranslator(batch_size=8)
responses_b8, _, report_b8 = preprocess_v2.process(resume_source, preprocessor, cache_b8, translator_b8, similarity_scorer)

StubNLLBTranslator.reset()
cache_b2, cache_dir_b2 = fresh_cache()
translator_b2 = StubNLLBTranslator(batch_size=2)
responses_b2, _, report_b2 = preprocess_v2.process(resume_source, preprocessor, cache_b2, translator_b2, similarity_scorer)

check(
    "batch_size=8 and batch_size=2 produce identical text_es for every response",
    all(responses_b8[rid]["text_es"] == responses_b2[rid]["text_es"] for rid in responses_b8),
)
check(
    "batch_size=8 and batch_size=2 produce the same sentence counts (segmentation is batch-size-independent)",
    all(responses_b8[rid]["sentence_count"] == responses_b2[rid]["sentence_count"] for rid in responses_b8),
)
check("batch_size=8 run is STEP2_VALID", report_b8["STEP2_VALID"] is True)
check("batch_size=2 run is STEP2_VALID (smaller batch size doesn't affect validity)", report_b2["STEP2_VALID"] is True)
check("the report records the batch_size actually used (8)", report_b8["batch_size"] == 8)
check("the report records the batch_size actually used (2)", report_b2["batch_size"] == 2)
shutil.rmtree(cache_dir_b8, ignore_errors=True)
shutil.rmtree(cache_dir_b2, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 41: NLLB progress reporting reaches 100% with correct fresh/cache_hit/failed counts ===")
progress_snapshots = []


def _collect_progress(snapshot):
    progress_snapshots.append(dict(snapshot))


cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=2)
# No trailing "." -- StubPreprocessor.split_sentences_strict splits (and
# strips) on ".", so a period here would make the chunk text actually
# looked up differ from what's pre-cached below by a trailing period.
progress_texts = [f"Short progress test sentence {i}" for i in range(7)]
# Pre-cache the first two so the run is a realistic fresh/cache-hit mix.
for t in progress_texts[:2]:
    cache.set(t, "ca", "es", f"Frase corta de progreso {t}", model_name=preprocess_v2.NLLB_MODEL_NAME, generation_version=preprocess_v2.GENERATION_VERSION)
cache.save()

results = preprocess_v2.translate_responses_batched(
    progress_texts, "ca", "es", preprocessor, translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker(),
    progress_hook=_collect_progress,
)
check("progress_hook fired at least once", len(progress_snapshots) > 0)
final_snapshot = progress_snapshots[-1]
check("progress reaches 100% -- responses_done equals the total", final_snapshot["responses_done"] == final_snapshot["total"] == len(progress_texts))
check("responses_done is monotonically non-decreasing across snapshots", all(
    progress_snapshots[i]["responses_done"] <= progress_snapshots[i + 1]["responses_done"]
    for i in range(len(progress_snapshots) - 1)
))
check("exactly 2 cache hits were counted (the pre-cached texts)", final_snapshot["cache_hits"] == 2)
check("exactly 5 fresh translations were counted (the remaining texts)", final_snapshot["fresh"] == 5)
check("no failures were counted", final_snapshot["failed"] == 0)
check("translate_responses_batched itself still returns one result per input text", len(results) == len(progress_texts))
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 42: a cache-write failure is reported safely, without raising or losing data ===")
cache_dir = tempfile.mkdtemp(prefix="nllb_test_cache_write_fail_")
# A filename component this long exceeds the filesystem's NAME_MAX (255
# bytes on Linux) no matter what user/permissions this test runs as --
# a permission-based sabotage (chmod 0o000) would NOT reliably fail here,
# since this suite may run as root, which bypasses normal write-permission
# checks entirely. A too-long name fails at the OS level regardless.
too_long_name = "x" * 300 + ".json"
unwritable_path = os.path.join(cache_dir, too_long_name)
broken_cache = preprocess_v2.TranslationCache(unwritable_path)
broken_cache.set("texto de prueba", "ca", "es", "resultado", model_name="m", generation_version="v")
try:
    broken_cache.save()
    save_raised = False
except Exception:
    save_raised = True
check("TranslationCache.save() does not raise when the target path can't be written", not save_raised)
check("the cache stays marked dirty after a failed save (nothing is falsely considered persisted)", broken_cache._dirty is True)
check(
    "no leftover .tmp file is left behind after a failed save",
    not any(name.startswith(too_long_name + ".tmp") for name in os.listdir(cache_dir)),
)
shutil.rmtree(cache_dir, ignore_errors=True)


# ===========================================================================
print("\n=== Test 43: runtime-architecture detection is diagnostic-only and never affects STEP2_VALID ===")
_real_get_runtime_architecture_info = preprocess_v2.get_runtime_architecture_info


def _fake_rosetta_info():
    return {
        "runtime_architecture": "x86_64",
        "running_under_rosetta": True,
        "warning": "Running as x86_64 Python translated by Rosetta 2 on Apple Silicon (simulated for this test).",
    }


preprocess_v2.get_runtime_architecture_info = _fake_rosetta_info
try:
    cache, cache_dir = fresh_cache()
    translator = StubNLLBTranslator()
    arch_source = {"iv1": {"q1": "Esta es una respuesta en español, suficientemente larga para esta prueba de arquitectura."}}
    _, _, arch_report = preprocess_v2.process(arch_source, preprocessor, cache, translator, similarity_scorer)
finally:
    preprocess_v2.get_runtime_architecture_info = _real_get_runtime_architecture_info

check("a simulated Rosetta/x86_64 environment is recorded in the report", arch_report["runtime_architecture"] == "x86_64")
check("the Rosetta warning is recorded in runtime_info", arch_report["runtime_info"]["running_under_rosetta"] is True)
check(
    "STEP2_VALID is unaffected by runtime-architecture detection -- it's a performance warning only",
    arch_report["STEP2_VALID"] is True,
)
check(
    "structural/completeness/output validity are all unaffected too",
    arch_report["structural_validity"] is True
    and arch_report["translation_completeness_validity"] is True
    and arch_report["translation_output_validity"] is True,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 44: progress fresh/cache_hit/failed counts are RESPONSE-level, not chunk-level (sixth review fix) ===")
# Sixth external review: the live postfix's fresh/cache_hits/failed counts
# were incrementing once per CHUNK inside _on_chunk_done, so a response
# spanning several chunks got counted multiple times -- silently inflating
# these numbers relative to the bar's own "n/total" reading, which was
# always response-level. This test mixes short (1-chunk) responses with a
# genuinely long (multi-chunk) one and proves the counts stay response-
# level: exactly one increment per response, regardless of how many chunks
# it took.
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=3)
long_multi_chunk_sentences = [f"Aquesta es la frase numero {i} d'una resposta llarga per la prova de progres." for i in range(40)]
long_multi_chunk_text = " ".join(long_multi_chunk_sentences)
progress_mix_texts = [
    "Frase curta u",
    long_multi_chunk_text,
    "Frase curta dos",
]
progress_snapshots_44 = []
results_44 = preprocess_v2.translate_responses_batched(
    progress_mix_texts, "ca", "es", splitter, translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker(),
    progress_hook=lambda snap: progress_snapshots_44.append(dict(snap)),
)
long_response_chunk_count = results_44[1][2]
check("the long response actually required more than one chunk (test is exercising the real bug)", long_response_chunk_count > 1)
final_snapshot_44 = progress_snapshots_44[-1]
check("responses_done reaches exactly 3 (the number of RESPONSES, not chunks)", final_snapshot_44["responses_done"] == 3)
check(
    "fresh count is exactly 3 -- the multi-chunk response counted ONCE, not once per chunk",
    final_snapshot_44["fresh"] == 3,
)
check(
    "fresh + cache_hits + failed always equals responses_done (each response classified exactly once)",
    final_snapshot_44["fresh"] + final_snapshot_44["cache_hits"] + final_snapshot_44["failed"] == final_snapshot_44["responses_done"],
)
check("no failures were counted", final_snapshot_44["failed"] == 0)
check("no snapshot ever shows fresh exceeding the total response count (the exact regression this test guards against)", all(
    snap["fresh"] + snap["cache_hits"] + snap["failed"] <= snap["total"] for snap in progress_snapshots_44
))
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 45: merge_ellipsis_continuations() -- source-side segmentation repair (Round 7) ===")
# Round 7 segmentation audit (evidence/round7_full_corpus_run_2026-09-07/):
# split_sentences_strict() on this corpus's transcribed speech routinely
# splits a speaker's mid-clause trailing-off ("..."/"…") into its own
# sentence, with the continuation split off separately. 38 confirmed cases
# across the real 950-response corpus, all same-speaker disfluency
# continuations -- none a topic/speaker shift. merge_ellipsis_
# continuations() repairs this BEFORE sentence IDs are frozen, since once
# frozen, segmentation becomes part of the corpus definition. These tests
# lock down that behavior, per the explicit review before freezing IDs.

import re as _re  # noqa: E402 (test-local; keeps the module-level import list unchanged for the rest of the suite)


def _word_tokens(text):
    """Word/number tokens only, ignoring punctuation, spacing, and the
    dash/ellipsis markers this whole feature is about removing -- used to
    verify lexical content is preserved (not reworded, not reordered),
    independent of exactly how boundary punctuation was stripped.
    """
    return _re.findall(r"[^\W\d_]+|\d+", text, flags=_re.UNICODE)


# --- one ellipsis + lowercase continuation -> merge ---
case1 = ["Jo estava parlant amb ell perquè...", "veiem que no funcionava bé."]
merged1, log1 = preprocess_v2.merge_ellipsis_continuations(case1)
check("single ellipsis+lowercase continuation merges into one sentence", len(merged1) == 1)
check("exactly one merge is logged", len(log1) == 1)
check(
    "merged text reads as one clean sentence, no stray ellipsis/markers in the middle",
    merged1[0] == "Jo estava parlant amb ell perquè veiem que no funcionava bé.",
)
check(
    "merge log records the original before/after text",
    log1[0]["before"] == case1 and log1[0]["after"] == merged1[0],
)

# --- literal "…" (single Unicode ellipsis char), not just "..." ---
case1b = ["Ell va dir que…", "no hi havia cap problema."]
merged1b, log1b = preprocess_v2.merge_ellipsis_continuations(case1b)
check("the single-character Unicode ellipsis (…) triggers the same merge as '...'", len(merged1b) == 1 and len(log1b) == 1)

# --- chained ellipsis continuations -> iterative merge into ONE sentence ---
# Mirrors the real corpus example (n23_Marcal...::4.2): one utterance split
# into three fragments by two separate ellipsis breaks.
case2 = [
    "-I llavors vam decidir posar sensors...",
    "-...de temperatura per tot arreu...",
    "-...i vam millorar molt.",
]
merged2, log2 = preprocess_v2.merge_ellipsis_continuations(case2)
check("a 3-fragment ellipsis chain collapses into exactly one sentence", len(merged2) == 1)
check("collapsing a 3-fragment chain takes exactly 2 merges", len(log2) == 2)
check(
    "the fully-collapsed chain reads as one clean sentence",
    merged2[0] == "-I llavors vam decidir posar sensors de temperatura per tot arreu i vam millorar molt.",
)
check(
    "count invariant holds for the chain: new_count = old_count - merges_performed",
    len(merged2) == len(case2) - len(log2),
)

# Same shape as the actual real-corpus n23_Marcal...::4.2 example, used
# verbatim as a direct regression check against the finding that motivated
# this whole feature.
real_example = [
    "-I allà van acabar ficant sensors...",
    "-...de camions com a tal, per saber que realment arriba a granja el que s'havia...",
    "-...enviat.",
]
merged_real, log_real = preprocess_v2.merge_ellipsis_continuations(real_example)
check("real-corpus n23_Marcal::4.2 chain collapses to exactly one sentence", len(merged_real) == 1)
check(
    "real-corpus chain collapses to the expected clean text",
    merged_real[0] == "-I allà van acabar ficant sensors de camions com a tal, per saber que realment arriba a granja el que s'havia enviat.",
)

# --- ellipsis + UPPERCASE next sentence -> do NOT merge ---
# An ellipsis-ending sentence followed by a capitalized, independent
# sentence is a genuine sentence boundary (or at least not the disfluency
# pattern this rule targets) -- must be left untouched.
case3 = ["Ell va marxar molt aviat...", "Després vam parlar una estona."]
merged3, log3 = preprocess_v2.merge_ellipsis_continuations(case3)
check("ellipsis followed by an uppercase-starting sentence is NOT merged", merged3 == case3)
check("no merge is logged when the next sentence starts uppercase", log3 == [])

# --- ordinary full stop + lowercase next unit -> do NOT merge automatically ---
# The trigger is specifically an ELLIPSIS ending -- an ordinary full stop
# must never trigger a merge, even if (rarely) the following unit happens
# to start lowercase, since that is not evidence of a trailing-off
# disfluency.
case4 = ["Ell va marxar molt aviat.", "però jo vaig quedar-me una estona."]
merged4, log4 = preprocess_v2.merge_ellipsis_continuations(case4)
check("an ordinary full stop never triggers a merge, regardless of next sentence's case", merged4 == case4)
check("no merge is logged for an ordinary full stop", log4 == [])

# --- response boundaries can never be crossed ---
# merge_ellipsis_continuations() has no concept of "response" at all -- it
# only ever sees the single sentence list passed to it. Demonstrated here
# with two lists that WOULD merge if concatenated into one list, but never
# merge when kept as two separate per-response calls (exactly how
# process() would call this: once per response, on that response's own
# sentence list only).
response_a_sentences = ["Al final vam decidir marxar perquè..."]
response_b_sentences = ["veiem que no hi havia futur allà."]
merged_a, log_a = preprocess_v2.merge_ellipsis_continuations(response_a_sentences)
merged_b, log_b = preprocess_v2.merge_ellipsis_continuations(response_b_sentences)
check("a response's trailing ellipsis sentence is untouched when it has no next sentence in ITS OWN list", merged_a == response_a_sentences)
check("a following response's sentence is untouched by another response's ellipsis", merged_b == response_b_sentences)
check("no merges are ever logged across what would be a response boundary", log_a == [] and log_b == [])
# Confirm this ISN'T just because the text itself is unmergeable -- the
# same two sentences DO merge when they legitimately belong to one list.
merged_concat, log_concat = preprocess_v2.merge_ellipsis_continuations(response_a_sentences + response_b_sentences)
check("the same two sentences DO merge when they are genuinely one list (proves the boundary case above is the list boundary, not the text)", len(merged_concat) == 1 and len(log_concat) == 1)

# --- original lexical content/order preserved exactly, except removal of
# the continuation boundary marker (dash/ellipsis) itself ---
original_words = []
for frag in case2:
    original_words.extend(_word_tokens(frag))
merged_words = _word_tokens(merged2[0])
check("merging never drops, reorders, or invents a word -- only the boundary markers are removed", merged_words == original_words)

# --- leading genuine turn marker on the FIRST fragment is retained ---
case7 = ["-Doncs jo crec que...", "el que hauríem de fer és esperar."]
merged7, log7 = preprocess_v2.merge_ellipsis_continuations(case7)
check("the first fragment's own leading turn marker is preserved in the merged sentence", merged7[0].startswith("-Doncs"))

# --- internal "-..." continuation marker removed ONLY when the rule
# actually merges -- never edited when it doesn't ---
case8_merge = ["Vam parlar molt perquè...", "-...necessitàvem aclarir coses."]
merged8m, log8m = preprocess_v2.merge_ellipsis_continuations(case8_merge)
check("when a merge happens, the internal '-...' continuation marker is gone from the result", "-..." not in merged8m[0] and len(merged8m) == 1)

# A leading "-..." with nothing before it in the list (first element) can
# never be a merge TARGET (there's no preceding sentence to trigger it) --
# must be left completely untouched, marker and all.
case8_no_merge = ["-...comença amb punts suspensius aquí.", "Una altra frase normal."]
merged8n, log8n = preprocess_v2.merge_ellipsis_continuations(case8_no_merge)
check("a '-...' marker is left untouched when there is no preceding ellipsis to trigger a merge into it", merged8n == case8_no_merge)
check("no merge is logged for the untouched '-...' case", log8n == [])

# --- count invariant across a larger, mixed stress case: new_count =
# old_count - number_of_merges ---
stress_case = (
    case1 + case2 + case3 + case4 + case7
    + ["Una frase completament independent sense cap ellipsi."]
)
merged_stress, log_stress = preprocess_v2.merge_ellipsis_continuations(stress_case)
check(
    "count invariant holds on a larger mixed list: new_count == old_count - merges_performed",
    len(merged_stress) == len(stress_case) - len(log_stress),
)

# --- re-scan after merge gives ZERO remaining instances of the targeted
# pattern (and the merge is idempotent -- re-running finds nothing left
# to do) ---
def _has_unresolved_ellipsis_continuation(sentences):
    ellipsis_trailing = _re.compile(r"(\.\.\.|…)\s*$")
    turn_marker = _re.compile(r"^[-–—]\s*")
    leading_ellipsis = _re.compile(r"^(\.\.\.|…)\s*")

    def looks_like_continuation(s):
        s = turn_marker.sub("", s)
        s = leading_ellipsis.sub("", s)
        for ch in s:
            if ch.isalpha():
                return ch.islower()
            if ch.isdigit() or ch in "\"'‘’“”(¿¡":
                return False
        return False

    return any(
        ellipsis_trailing.search(sentences[i]) and looks_like_continuation(sentences[i + 1])
        for i in range(len(sentences) - 1)
    )


check("the targeted pattern is fully gone from the merged stress case", not _has_unresolved_ellipsis_continuation(merged_stress))
merged_stress_again, log_stress_again = preprocess_v2.merge_ellipsis_continuations(merged_stress)
check("merging is idempotent -- re-running on already-merged output performs zero further merges", log_stress_again == [] and merged_stress_again == merged_stress)


# ===========================================================================
print("\n=== Test 46: segment_source_sentences() -- the single canonical split+merge path ===")
# The frozen sentence-ID structure and the eventual one-sentence-per-
# translation redesign must segment text identically -- this composition
# function is the one place that combines split_sentences_strict() and
# merge_ellipsis_continuations(), so neither caller can drift by forgetting
# the merge step or calling things in the wrong order.


class _EllipsisAwareSplitter:
    """A splitter whose split_sentences_strict() reproduces the real
    ellipsis-continuation artifact (splits on ". " AND on a trailing
    ellipsis), so this test exercises segment_source_sentences() actually
    invoking the merge step -- not just trusting it's wired in.
    """

    def split_sentences_strict(self, text, lang):
        import re as _re2
        parts = _re2.split(r"(?<=[.])\s+|(?<=\.\.\.)\s+|(?<=…)\s+", text)
        return [p.strip() for p in parts if p.strip()]


ellipsis_splitter = _EllipsisAwareSplitter()
combined_text = "Ell va marxar molt aviat perquè... teníem pressa per arribar."
segmented, seg_log = preprocess_v2.segment_source_sentences(combined_text, "ca", ellipsis_splitter)
check("segment_source_sentences merges an ellipsis-continuation produced by the real split step", len(segmented) == 1)
check("segment_source_sentences returns the merge log from the underlying merge step", len(seg_log) == 1)

# Falls back to [text] (matching _split_into_translation_chunks' own
# existing defensive pattern) if the preprocessor's splitter misbehaves,
# rather than propagating the exception into a corpus-wide run.
class _BrokenSplitter:
    def split_sentences_strict(self, text, lang):
        raise RuntimeError("simulated spaCy failure")


segmented_broken, seg_log_broken = preprocess_v2.segment_source_sentences("Qualsevol text.", "ca", _BrokenSplitter())
check("a splitter failure falls back to treating the whole text as one sentence, not a crash", segmented_broken == ["Qualsevol text."])
check("no merges are logged on the fallback path", seg_log_broken == [])

# On the REAL splitter (StubPreprocessor is a naive split-on-period stand-
# in, so use the real, spaCy-backed Preprocessor path indirectly via the
# actual corpus-derived example instead): confirm the composed function
# reproduces exactly what split_sentences_strict() + merge_ellipsis_
# continuations() would give when called separately, for the same input.
manual_split = ellipsis_splitter.split_sentences_strict(combined_text, "ca")
manual_merged, manual_log = preprocess_v2.merge_ellipsis_continuations(manual_split)
check(
    "segment_source_sentences produces exactly the same result as calling split then merge by hand",
    segmented == manual_merged and seg_log == manual_log,
)


# ===========================================================================
print("\n=== Test 47: translate_frozen_sentences_batched() progress is genuinely incremental ===")
# Regression test for the "0/5792 for a long time, then jumps to 100%"
# report: translate_frozen_sentences_batched() used to call translate_
# texts_batched() for the WHOLE first-pass piece list without wiring its
# progress_callback through, so every sentence's progress tick was deferred
# to a single bulk loop AFTER that entire call returned -- correct final
# counts, but no visible progress while it ran. This test proves ticks
# happen DURING the call (via progress_hook), not only once at the end,
# and that cache hits, fresh translations, and a first-pass failure that
# recovers on retry are all counted correctly and immediately.
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
# batch_size=1 -- one NLLB call per sentence -- deliberately, so FAIL_MODE
# "n_times" (a whole-BATCH failure) fails exactly one specific sentence's
# call (whichever cache-miss is processed first) rather than an arbitrary
# mix, making the "fails once, then the retry pass succeeds" scenario
# exactly reproducible.
translator = StubNLLBTranslator(batch_size=1)
StubNLLBTranslator.FAIL_MODE = "n_times"
StubNLLBTranslator.FAILS_REMAINING = 1

# rA's sentences are NOT pre-cached, so they become misses processed (in
# this order) before rB's -- s000 is deliberately first, so it's the one
# whose lone batch call the "n_times" failure consumes.
retry_text = "Aquesta frase falla la primera vegada pero es recupera"
fresh_text_1 = "Aquesta es una frase fresca numero u"
fresh_text_2 = "Aquesta es una frase fresca numero dos"
cached_text_1 = "Aquesta frase ja esta traduida i cachejada u"
cached_text_2 = "Aquesta frase ja esta traduida i cachejada dos"
spanish_text = "Esta frase ya esta en español y no debe enviarse a NLLB"

for t in (cached_text_1, cached_text_2):
    cache.set(t, "ca", "es", f"Cache: {t}", model_name=preprocess_v2.NLLB_MODEL_NAME, generation_version=preprocess_v2.GENERATION_VERSION)
cache.save()

segmented_responses = {
    "ivP::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": "ivP::qA::s000", "sentence_index": 0, "text_source": retry_text},
            {"sentence_id": "ivP::qA::s001", "sentence_index": 1, "text_source": fresh_text_1},
            {"sentence_id": "ivP::qA::s002", "sentence_index": 2, "text_source": fresh_text_2},
        ],
    },
    "ivP::qB": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": "ivP::qB::s000", "sentence_index": 0, "text_source": cached_text_1},
            {"sentence_id": "ivP::qB::s001", "sentence_index": 1, "text_source": cached_text_2},
        ],
    },
    "ivP::qC": {
        "source_language": "es",
        "sentences": [
            {"sentence_id": "ivP::qC::s000", "sentence_index": 0, "text_source": spanish_text},
        ],
    },
}

progress_calls_47 = []
sentence_results_47 = preprocess_v2.translate_frozen_sentences_batched(
    segmented_responses, translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker(),
    show_progress=False, progress_hook=lambda snap: progress_calls_47.append(dict(snap)),
)

check(
    "exactly one progress_hook call per Catalan sentence (5), not one bulk call at the end",
    len(progress_calls_47) == 5,
)
check(
    "the FIRST progress tick fires with a partial count, not the final total -- proof it's live, not a single end-of-call burst",
    progress_calls_47[0]["sentences_done"] == 1 and progress_calls_47[0]["total"] == 5,
)
check(
    "sentences_done is monotonically increasing across ticks, one per tick",
    [c["sentences_done"] for c in progress_calls_47] == [1, 2, 3, 4, 5],
)
check(
    "a cache hit is what produces the very first tick (cached sentences advance progress immediately)",
    progress_calls_47[0]["cache_hits"] == 1,
)
check(
    "final progress count equals the number of Catalan sentence units (5 -- the Spanish sentence is excluded)",
    progress_calls_47[-1]["sentences_done"] == progress_calls_47[-1]["total"] == 5,
)
final_47 = progress_calls_47[-1]
check("final tally: 2 cache hits", final_47["cache_hits"] == 2)
check("final tally: 2 fresh translations", final_47["fresh"] == 2)
check("final tally: 1 retried-OK translation (the first-pass failure recovered)", final_47["retried_ok"] == 1)
check("final tally: 0 permanent failures", final_47["failed"] == 0)

check(
    "the retried sentence's actual translation_provenance in the returned results is RETRY_OK",
    sentence_results_47["ivP::qA::s000"]["translation_provenance"] == "RETRY_OK",
)
check(
    "the retried sentence still has real translated text, not None",
    sentence_results_47["ivP::qA::s000"]["text_es"] not in (None, ""),
)
check(
    "the two cached sentences report CACHE_HIT provenance",
    sentence_results_47["ivP::qB::s000"]["translation_provenance"] == "CACHE_HIT"
    and sentence_results_47["ivP::qB::s001"]["translation_provenance"] == "CACHE_HIT",
)
check(
    "the Spanish sentence was never sent to NLLB (SOURCE_ES, untouched)",
    sentence_results_47["ivP::qC::s000"] == {"text_es": spanish_text, "translation_provenance": "SOURCE_ES", "hard_split_count": 0},
)
check(
    "translate_frozen_sentences_batched still returns exactly one record per frozen sentence ID (6 total, including the Spanish one)",
    len(sentence_results_47) == 6,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 48: a retry pass that fails permanently still reaches the correct final count ===")
# Companion to Test 47's "retry succeeds" case: here every retried
# sentence stays FAILED. This is what the safety-net reconciliation pass
# at the end of translate_frozen_sentences_batched exists for -- progress
# must still land on exactly total_ca_sentences, with every failure
# correctly attributed, never silently dropped or double-counted.
StubNLLBTranslator.reset()
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator(batch_size=8)
StubNLLBTranslator.FAIL_MODE = "always"  # every batch call fails, first pass AND retry

fail_texts = [f"Aquesta frase sempre falla numero {i}" for i in range(4)]
segmented_responses_48 = {
    "ivQ::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": f"ivQ::qA::s{i:03d}", "sentence_index": i, "text_source": t}
            for i, t in enumerate(fail_texts)
        ],
    },
}

progress_calls_48 = []
results_48 = preprocess_v2.translate_frozen_sentences_batched(
    segmented_responses_48, translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker(),
    show_progress=False, progress_hook=lambda snap: progress_calls_48.append(dict(snap)),
)

check(
    "final progress count still equals the number of Catalan sentence units, even though all 4 failed",
    progress_calls_48[-1]["sentences_done"] == progress_calls_48[-1]["total"] == 4,
)
check("all 4 are counted as failed, not lost or miscounted", progress_calls_48[-1]["failed"] == 4)
check(
    "every sentence's stored result is FAILED with no translated text",
    all(r["translation_provenance"] == "FAILED" and r["text_es"] is None for r in results_48.values()),
)
check("no sentence_id is missing from the results", set(results_48.keys()) == {f"ivQ::qA::s{i:03d}" for i in range(4)})
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 49: a retry pass that itself hits the consecutive-failure abort recovers cleanly ===")
# The sharpest edge case the idempotent-tick + end-of-function reconciliation
# design exists for: the RETRY pass's own translate_texts_batched call can
# raise NLLBTranslationError partway through (its fresh ConsecutiveBatch
# FailureTracker hits MAX_CONSECUTIVE_BATCH_FAILURES), AFTER some of its
# own per-piece callbacks already fired for sentences that failed just
# before the abort. translate_frozen_sentences_batched's except-branch
# then treats every retried sentence as FAILED (including ones never even
# attempted, like a 4th sentence never reached before the abort) -- this
# proves that fallback and the live callbacks never disagree or double
# count, and that a never-attempted sentence still gets exactly one tick.
StubNLLBTranslator.reset()


class _SelectiveFailTranslator(StubNLLBTranslator):
    """Fails the whole batch call only when one of FAIL_TEXTS is in it,
    independent of StubNLLBTranslator's own FAIL_MODE machinery. Lets a
    test construct failures that are non-consecutive across the first
    pass (each interspersed with an unrelated successful sentence) but
    ARE consecutive once only the failing subset is retried together --
    the only way to make the retry pass hit its OWN abort threshold
    without the first pass hitting it first (see comment below).
    """
    FAIL_TEXTS: frozenset = frozenset()

    def translate_batch(self, texts, source_lang, target_lang):
        StubNLLBTranslator.CALL_LOG.append((source_lang, target_lang, list(texts)))
        if any(t in self.__class__.FAIL_TEXTS for t in texts):
            raise NLLBTranslationError("stubbed: selective failure")
        default_output = STUB_ES_OUTPUT if target_lang == "es" else STUB_CA_OUTPUT
        return [default_output for _ in texts]


cache, cache_dir = fresh_cache()
bad_texts = [f"Aquesta frase dolenta sempre falla numero {i}" for i in range(4)]
good_texts = [f"Aquesta frase bona sempre funciona numero {i}" for i in range(3)]
_SelectiveFailTranslator.FAIL_TEXTS = frozenset(bad_texts)
translator = _SelectiveFailTranslator(batch_size=1)  # one call per sentence

# Interleaved so first-pass failures are never 3-in-a-row (each bad_i is
# immediately followed by a good_i that resets the tracker) -- MAX_
# CONSECUTIVE_BATCH_FAILURES (3) is never reached in the first pass, so it
# completes normally and defers all 4 bad sentences to retry. The retry
# pass then attempts them back-to-back with no successes in between, so
# its OWN fresh tracker DOES hit the threshold on the 3rd retried failure.
interleaved = [bad_texts[0], good_texts[0], bad_texts[1], good_texts[1], bad_texts[2], good_texts[2], bad_texts[3]]
sentences_49 = [
    {"sentence_id": f"ivR::qA::s{i:03d}", "sentence_index": i, "text_source": t}
    for i, t in enumerate(interleaved)
]
segmented_responses_49 = {"ivR::qA": {"source_language": "ca", "sentences": sentences_49}}

progress_calls_49a = []
results_49 = preprocess_v2.translate_frozen_sentences_batched(
    segmented_responses_49, translator, cache, preprocess_v2.ConsecutiveBatchFailureTracker(),
    show_progress=False, progress_hook=lambda snap: progress_calls_49a.append(dict(snap)),
)

check(
    "the run completes without raising -- the retry-pass abort is caught and handled, not propagated",
    True,  # reaching this line at all is the check: a propagated exception would have stopped the test script
)
check(
    "final progress count still equals all 7 sentence units (3 good + 4 bad), none lost to the internal abort",
    progress_calls_49a[-1]["sentences_done"] == progress_calls_49a[-1]["total"] == 7,
)
check("the 3 interleaved good sentences are correctly counted as fresh", progress_calls_49a[-1]["fresh"] == 3)
check(
    "all 4 bad sentences -- including the 4th, never actually attempted before the retry pass aborted -- are counted as failed",
    progress_calls_49a[-1]["failed"] == 4,
)
check(
    "every bad sentence's stored result is FAILED with no translated text, the 4th included",
    all(results_49[f"ivR::qA::s{i:03d}"]["translation_provenance"] == "FAILED" and results_49[f"ivR::qA::s{i:03d}"]["text_es"] is None for i in (0, 2, 4, 6)),
)
check(
    "every good sentence's stored result is FRESH_OK with real translated text",
    all(results_49[f"ivR::qA::s{i:03d}"]["translation_provenance"] == "FRESH_OK" and results_49[f"ivR::qA::s{i:03d}"]["text_es"] for i in (1, 3, 5)),
)
check(
    "no sentence was ticked twice -- exactly one progress_hook call per sentence, 7 total",
    len(progress_calls_49a) == 7,
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 50: the frozen 6,396-sentence structure is untouched by the progress-reporting fix ===")
# Runs translate_frozen_sentences_batched() directly against the REAL,
# on-disk frozen segmentation file (the one Round 7 froze and main() now
# requires) with a fast, deterministic, offline stand-in translator --
# this is a structural regression check, not a translation-quality run.
# It exists specifically to confirm Test 47/48's rewrite didn't disturb
# the one invariant that matters most: one frozen source sentence in, one
# downstream sentence record out, for the real corpus at real scale.
_frozen_path = os.path.join(os.path.dirname(os.path.abspath(preprocess_v2.__file__)), preprocess_v2.FROZEN_SEGMENTATION_PATH)
if os.path.exists(_frozen_path):
    with open(_frozen_path, encoding="utf-8") as f:
        _frozen_file = json.load(f)
    _frozen_segmentation = _frozen_file["responses"]
    _expected_total = _frozen_file["total_sentence_count"]

    class _FastDeterministicTranslator(StubNLLBTranslator):
        """Same shape as StubNLLBTranslator, just given its own class
        identity so this test's FAIL_MODE/CALL_LOG state can never leak
        into or out of any other test's StubNLLBTranslator class state.
        """
        FAIL_MODE = "none"
        FAILS_REMAINING = 0
        EMPTY_FOR_TEXTS = frozenset()
        IDENTICAL_FOR_TEXTS = frozenset()
        REPETITIVE_FOR_TEXTS = frozenset()
        WRONG_LANGUAGE_FOR_TEXTS = frozenset()
        CALL_LOG = []

    fast_translator = _FastDeterministicTranslator(batch_size=64)
    fast_cache, fast_cache_dir = fresh_cache()

    progress_calls_50 = []
    real_results = preprocess_v2.translate_frozen_sentences_batched(
        _frozen_segmentation, fast_translator, fast_cache, preprocess_v2.ConsecutiveBatchFailureTracker(),
        show_progress=False, progress_hook=lambda snap: progress_calls_50.append(dict(snap)),
    )

    _frozen_ids = {sid for r in _frozen_segmentation.values() for sid in r["sentence_ids"]}
    check("real frozen file total is still 6,396 sentences (unchanged by this fix)", _expected_total == 6396)
    check(
        "translate_frozen_sentences_batched produces exactly one record per frozen sentence ID",
        set(real_results.keys()) == _frozen_ids and len(real_results) == _expected_total,
    )
    _ca_sentence_count = sum(1 for r in real_results.values() if r["translation_provenance"] != "SOURCE_ES")
    check(
        "progress reached exactly the number of Catalan (non-SOURCE_ES) sentences at real corpus scale",
        progress_calls_50[-1]["sentences_done"] == progress_calls_50[-1]["total"] == _ca_sentence_count,
    )
    check(
        "progress ticked incrementally at real scale too (thousands of separate calls, not one)",
        len(progress_calls_50) == _ca_sentence_count and _ca_sentence_count > 1000,
    )
    check("no sentence was left without a translation_provenance", all(r.get("translation_provenance") for r in real_results.values()))
    shutil.rmtree(fast_cache_dir, ignore_errors=True)
else:
    check("frozen segmentation file not present in this environment -- skipped, not failed", True)


# ===========================================================================
print("\n=== Test 51: validate_frozen_segmentation_against_input() -- hard pre-run freshness gate ===")
# A seventh external review found process()'s own per-response frozen
# lookup (missing_from_frozen_segmentation / frozen_segmentation_source_
# language_mismatch) cannot catch a frozen file that has quietly gone
# stale for the corpus as a WHOLE -- a response silently removed from the
# frozen file (never iterated, so never checked), or one whose original
# text changed while its ID/language happened to stay the same. Each of
# the five checks below is exercised in isolation (only one class of
# problem present at a time), confirming both that each one actually
# fires AND that a genuinely fresh frozen file produces zero problems.
_v51_hash_tmp_dir = tempfile.mkdtemp(prefix="frozen_validate_test_")
_v51_input_path = os.path.join(_v51_hash_tmp_dir, "interviews.json")


def _v51_write_input_and_hash(raw_data_dict):
    with open(_v51_input_path, "w", encoding="utf-8") as f:
        json.dump(raw_data_dict, f, ensure_ascii=False)
    return preprocess_v2._hash_file(_v51_input_path)


# -- Clean baseline: everything agrees, zero problems. --
v51_raw_data = {"iv1": {"q1": "Text de referencia per la prova de validacio del fitxer congelat."}}
v51_hash = _v51_write_input_and_hash(v51_raw_data)
v51_frozen = {
    "schema_version": preprocess_v2.FROZEN_SEGMENTATION_SCHEMA_VERSION,
    "input_sha256": v51_hash,
    "response_count": 1,
    "total_sentence_count": 1,
    "responses": {
        "iv1::q1": {
            "original_text": "Text de referencia per la prova de validacio del fitxer congelat.",
            "sentences": [{"sentence_id": "iv1::q1::s000", "sentence_index": 0, "text_source": "Text de referencia per la prova de validacio del fitxer congelat."}],
        },
    },
}
v51_problems = preprocess_v2.validate_frozen_segmentation_against_input(v51_frozen, v51_raw_data, _v51_input_path)
check("a genuinely fresh, self-consistent frozen file produces zero problems", v51_problems == [])

# -- Check 1: stale schema_version -- short-circuits, nothing else checked. --
v51_stale_schema = dict(v51_frozen, schema_version=1)
del v51_stale_schema["input_sha256"]  # a real schema_version=1 file never had this field at all
v51_problems_1 = preprocess_v2.validate_frozen_segmentation_against_input(v51_stale_schema, v51_raw_data, _v51_input_path)
check("a stale schema_version is caught", len(v51_problems_1) == 1 and "schema_version" in v51_problems_1[0])
check("a stale schema_version short-circuits -- nothing else is checked on top of it", len(v51_problems_1) == 1)

# -- Check 2: declared counts don't match what's actually inside the file. --
v51_bad_count = dict(v51_frozen, response_count=99)
v51_problems_2 = preprocess_v2.validate_frozen_segmentation_against_input(v51_bad_count, v51_raw_data, _v51_input_path)
check("a self-inconsistent declared response_count is caught", any("response_count" in p for p in v51_problems_2))
check("nothing else is wrongly flagged alongside the count mismatch", len(v51_problems_2) == 1)

# -- Check 3: input_sha256 recorded in the frozen file no longer matches the actual input file. --
v51_stale_hash = dict(v51_frozen, input_sha256="0" * 64)
v51_problems_3 = preprocess_v2.validate_frozen_segmentation_against_input(v51_stale_hash, v51_raw_data, _v51_input_path)
check("a stale input_sha256 (input file changed) is caught", any("input_sha256 mismatch" in p for p in v51_problems_3))
check("nothing else is wrongly flagged alongside the hash mismatch", len(v51_problems_3) == 1)

# -- Check 4a: a response in the CURRENT corpus is missing from the frozen file entirely. --
v51_raw_extra = dict(v51_raw_data, iv2={"q1": "Una segona resposta que no existeix al fitxer congelat de referencia."})
v51_hash_extra = _v51_write_input_and_hash(v51_raw_extra)
v51_frozen_missing = dict(v51_frozen, input_sha256=v51_hash_extra)
v51_problems_4a = preprocess_v2.validate_frozen_segmentation_against_input(v51_frozen_missing, v51_raw_extra, _v51_input_path)
check(
    "a response present in the current corpus but absent from the frozen file is caught",
    any("not in the frozen segmentation at all" in p for p in v51_problems_4a),
)
check("nothing else is wrongly flagged alongside the missing-response case", len(v51_problems_4a) == 1)

# -- Check 4b: a response in the frozen file no longer exists in the current corpus (the case process()'s own per-response lookup can never catch by itself). --
v51_hash_shrunk = _v51_write_input_and_hash(v51_raw_data)  # back to the 1-response baseline file on disk
v51_frozen_stale_extra = {
    "schema_version": preprocess_v2.FROZEN_SEGMENTATION_SCHEMA_VERSION,
    "input_sha256": v51_hash_shrunk,
    "response_count": 2,
    "total_sentence_count": 2,
    "responses": dict(
        v51_frozen["responses"],
        **{
            "iv2::q1": {
                "original_text": "Una resposta congelada que ja no existeix a l'entrevista actual.",
                "sentences": [{"sentence_id": "iv2::q1::s000", "sentence_index": 0, "text_source": "Una resposta congelada que ja no existeix a l'entrevista actual."}],
            },
        },
    ),
}
v51_problems_4b = preprocess_v2.validate_frozen_segmentation_against_input(v51_frozen_stale_extra, v51_raw_data, _v51_input_path)
check(
    "a response present in the frozen file but no longer in the current corpus is caught "
    "(this is exactly the case process()'s own per-response lookup can never detect by itself)",
    any("no longer exist in the current corpus" in p for p in v51_problems_4b),
)
check("nothing else is wrongly flagged alongside the stale-in-frozen case", len(v51_problems_4b) == 1)

# -- Check 5: same response ID exists in both, but its recorded text differs. --
v51_frozen_text_drift = {
    "schema_version": preprocess_v2.FROZEN_SEGMENTATION_SCHEMA_VERSION,
    "input_sha256": v51_hash_shrunk,
    "response_count": 1,
    "total_sentence_count": 1,
    "responses": {
        "iv1::q1": {
            "original_text": "Aquest text ha canviat silenciosament des que es va congelar la segmentacio.",
            "sentences": [{"sentence_id": "iv1::q1::s000", "sentence_index": 0, "text_source": "Aquest text ha canviat silenciosament des que es va congelar la segmentacio."}],
        },
    },
}
v51_problems_5 = preprocess_v2.validate_frozen_segmentation_against_input(v51_frozen_text_drift, v51_raw_data, _v51_input_path)
check(
    "a response whose text quietly drifted (same ID, different content) is caught",
    any("text differs from what the frozen segmentation was built from" in p for p in v51_problems_5),
)
shutil.rmtree(_v51_hash_tmp_dir, ignore_errors=True)

# -- Wiring: main() refuses to run (returns False, no model load attempted) when this check fails. --
# Note: "run_preflight(" appears TWICE in main()'s source -- once in the
# early `mode == "preflight"` branch (irrelevant here; that branch returns
# before ever reaching the frozen-file logic below), and once in the real
# full-run path, which is the occurrence that must come AFTER the
# validation call. rindex (last occurrence) is deliberately used instead
# of index (first occurrence) to land on that second one.
_v51_main_source = inspect.getsource(preprocess_v2.main)
check(
    "main() calls validate_frozen_segmentation_against_input() before the (real, full-run) model load, and returns False on any problem",
    "validate_frozen_segmentation_against_input(" in _v51_main_source
    and _v51_main_source.index("validate_frozen_segmentation_against_input(") < _v51_main_source.rindex("run_preflight("),
)


# ===========================================================================
print("\n=== Test 52: sentence_count_invariant's exact ID-set fields (not just cardinality) ===")
# An eighth external review noted the production report's invariant only
# ever checked expected_count == actual_count -- two numbers matching by
# coincidence would still pass even if a real sentence ID were silently
# dropped while a duplicate of another one was produced instead. This
# proves the new fields (a) report a clean match/[]/[] on ordinary runs,
# for both the frozen-file path and the live-segmentation path, and (b)
# actually catch a case a pure count check handles differently than a
# pure set-equality check would -- see the duplicate-sentence-id case
# below, where the SET of IDs matches (a duplicate collapses in a set)
# but the cardinality does not, which is exactly why `match` requires
# both conditions together, not either alone.
StubNLLBTranslator.reset()

# -- (a) live-segmentation path (no frozen_segmentation given), multi-response, all-correct. --
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
live_raw = {
    "ivL1": {"q1": "Esta es una respuesta normal en español para la prueba del invariante de conteo."},
    "ivL2": {"q1": "Aquesta és una resposta normal en català per la prova de l'invariant de recompte."},
}
_, sentences_out_52a, report_52a = preprocess_v2.process(live_raw, preprocessor, cache, translator, similarity_scorer)
sci_52a = report_52a["sentence_count_invariant"]
check("live-segmentation path: expected == actual", sci_52a["expected"] == sci_52a["actual"] == len(sentences_out_52a))
check("live-segmentation path: match is True", sci_52a["match"] is True)
check("live-segmentation path: sentence_id_set_match is True with empty missing/unexpected lists", sci_52a["sentence_id_set_match"] is True and sci_52a["missing_sentence_ids"] == [] and sci_52a["unexpected_sentence_ids"] == [])
shutil.rmtree(cache_dir, ignore_errors=True)

# -- (b) frozen-segmentation path, multi-response, all-correct. --
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
frozen_raw_52b = {
    "ivF1": {"q1": "Primera resposta per la prova de l'invariant amb segmentacio congelada."},
    "ivF2": {"q1": "Segunda respuesta en español para la prueba del invariante con segmentacion congelada."},
}
frozen_seg_52b = {
    "ivF1::q1": {"source_language": "ca", "sentences": [{"sentence_id": "ivF1::q1::s000", "sentence_index": 0, "text_source": "Primera resposta per la prova de l'invariant amb segmentacio congelada."}]},
    "ivF2::q1": {"source_language": "es", "sentences": [{"sentence_id": "ivF2::q1::s000", "sentence_index": 0, "text_source": "Segunda respuesta en español para la prueba del invariante con segmentacion congelada."}]},
}
_, sentences_out_52b, report_52b = preprocess_v2.process(frozen_raw_52b, preprocessor, cache, translator, similarity_scorer, frozen_segmentation=frozen_seg_52b)
sci_52b = report_52b["sentence_count_invariant"]
check("frozen-segmentation path: expected == actual == 2", sci_52b["expected"] == sci_52b["actual"] == 2)
check("frozen-segmentation path: match and sentence_id_set_match are both True", sci_52b["match"] is True and sci_52b["sentence_id_set_match"] is True)
check("frozen-segmentation path: the exact expected sentence IDs are the ones produced", set(sentences_out_52b.keys()) == {"ivF1::q1::s000", "ivF2::q1::s000"})
shutil.rmtree(cache_dir, ignore_errors=True)

# -- (c) a corrupted frozen file with a duplicate sentence_id within one response: sets collapse to equal, but cardinality does not -- proving why `match` needs BOTH checks, not either alone. --
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
dup_text_1 = "Primera frase en català per la prova de duplicats de identificador."
dup_text_2 = "Segona frase amb el mateix identificador de frase per error de congelacio."
raw_52c = {"ivD": {"qA": dup_text_1 + " " + dup_text_2}}
frozen_seg_52c = {
    "ivD::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": "ivD::qA::s000", "sentence_index": 0, "text_source": dup_text_1},
            {"sentence_id": "ivD::qA::s000", "sentence_index": 1, "text_source": dup_text_2},  # duplicate ID, distinct text -- a corrupted-freeze scenario
        ],
    },
}
_, sentences_out_52c, report_52c = preprocess_v2.process(raw_52c, preprocessor, cache, translator, similarity_scorer, frozen_segmentation=frozen_seg_52c)
sci_52c = report_52c["sentence_count_invariant"]
check("duplicate-sentence-id case: the duplicate is caught as a FATAL error", report_52c["duplicate_sentence_id_count"] == 1)
check("duplicate-sentence-id case: expected (2, raw list length) != actual (1, one key overwrote the other)", sci_52c["expected"] == 2 and sci_52c["actual"] == 1)
check(
    "duplicate-sentence-id case: sentence_id_set_match is True (a duplicate collapses to an equal SET) even though the run is NOT actually sound",
    sci_52c["sentence_id_set_match"] is True,
)
check(
    "duplicate-sentence-id case: match is correctly False anyway, because it also requires the cardinality check -- proving sentence_id_set_match alone would have missed this",
    sci_52c["match"] is False,
)
check("duplicate-sentence-id case: structural_validity is False (both the fatal_errors count and the invariant mismatch independently say so)", report_52c["structural_validity"] is False)
check("duplicate-sentence-id case: STEP2_VALID is False", report_52c["STEP2_VALID"] is False)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 53: response-level-only repetition across sentence boundaries no longer FAILs STEP2_VALID (demoted to REVIEW) ===")
# The core of the eighth review's fix #7/#8: translation_output_validity is
# now gated on sentence-level FLAGGED status, not on the response-level
# diagnostics computed on the RECONSTRUCTED (joined) text_es. This builds a
# response from six DISTINCT, individually-fine Catalan sentences that
# each happen to translate to the same short 3-word Spanish phrase --
# short enough that no single sentence's own degenerate-output check ever
# sees the repetition threshold (len(tokens) >= 3*6=18 (Round 11's raised
# REPETITION_MIN_REPEATS) is never met by a 3-token sentence) -- but whose
# CONCATENATION contains that exact 3-gram six times in a row, which is
# precisely what used to trip the old response-level-only gate as a false
# positive (see the sixth/seventh reviews this same failure mode was
# already fixed for once, at a different grain -- this is the same class
# of bug recurring one level up). Six sentences, not four, because Round
# 11 raised REPETITION_MIN_REPEATS to 6 -- four repeats of a 3-gram no
# longer trips this diagnostic at all, response-level or sentence-level.
StubNLLBTranslator.reset()


class _RepeatedShortPhraseTranslator(StubNLLBTranslator):
    """Every sentence handed to this translator comes back as the exact
    same short, valid, non-empty Spanish phrase -- individually harmless,
    but repetitive once several such sentences are joined into one
    response's reconstructed text_es.
    """
    FAIL_MODE = "none"
    FAILS_REMAINING = 0
    EMPTY_FOR_TEXTS = frozenset()
    IDENTICAL_FOR_TEXTS = frozenset()
    REPETITIVE_FOR_TEXTS = frozenset()
    WRONG_LANGUAGE_FOR_TEXTS = frozenset()
    CALL_LOG = []

    def translate_batch(self, texts, source_lang, target_lang):
        _RepeatedShortPhraseTranslator.CALL_LOG.append((source_lang, target_lang, list(texts)))
        return ["gracias por todo" for _ in texts]


cache, cache_dir = fresh_cache()
translator_53 = _RepeatedShortPhraseTranslator()
sent_texts_53 = [
    "Primera frase catalana per la prova de repeticio a nivell de resposta.",
    "Segona frase catalana, diferent de la primera, per la mateixa prova.",
    "Tercera frase catalana, tambe diferent, per continuar la prova.",
    "Quarta frase catalana, diferent de les anteriors, per continuar la prova.",
    "Cinquena frase catalana, tambe diferent, per continuar la prova.",
    "Sisena frase catalana, l'ultima del grup, per completar la prova.",
]
raw_53 = {"ivR": {"qA": " ".join(sent_texts_53)}}
frozen_seg_53 = {
    "ivR::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": f"ivR::qA::s{i:03d}", "sentence_index": i, "text_source": t}
            for i, t in enumerate(sent_texts_53)
        ],
    },
}
_, sentences_out_53, report_53 = preprocess_v2.process(raw_53, preprocessor, cache, translator_53, similarity_scorer, frozen_segmentation=frozen_seg_53)

check(
    "none of the six individual sentences is FLAGGED (each one's own output is too short to trip the per-sentence repetition threshold)",
    all(sentences_out_53[f"ivR::qA::s{i:03d}"]["sentence_status"] != "FLAGGED" for i in range(6)),
)
check("sentence_translation_summary reports zero flagged sentences", report_53["sentence_translation_summary"]["flagged_sentence_count"] == 0)
check(
    "the RECONSTRUCTED response-level text_es DOES trip the response-level repetition diagnostic (proving this is a genuine response-level-only false positive, not a vacuous test)",
    report_53["translation_quality_summary"]["degenerate_output_count"] == 1,
)
check("translation_output_validity is True -- the sentence-level gate is what decides this now, and no sentence was flagged", report_53["translation_output_validity"] is True)
check(
    "translation_sanity_status is REVIEW, not FAIL -- the response-level repetition is surfaced for a human to look at, but does not by itself invalidate Step 2",
    report_53["translation_sanity_status"] == "REVIEW",
)
check("STEP2_VALID is True -- this response-level-only diagnostic no longer recreates the false-positive gate the review flagged", report_53["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 54: a single FLAGGED sentence among good neighbors in the SAME response still correctly FAILs STEP2_VALID ===")
# The mirror image of Test 53, and the other half of fix #7/#8: moving the
# gate to sentence-level must not accidentally become a majority-vote or
# an average -- one genuinely bad sentence, diluted among otherwise-good
# ones in the very same response, must still fail the run. (Test 34
# already covers this for a single-sentence response, where sentence-level
# and response-level are the same thing by construction; this is the
# multi-sentence case that one cannot distinguish.)
#
# Round 11: this now uses a DEGENERATE (repetitive) bad sentence, not a
# wrong-language one -- a language-mismatch-only sentence is no longer
# fatal at all (see Test 33/60/61), so it can no longer serve as "the one
# bad sentence" this test needs. check_degenerate_output is now the only
# sentence-level fatal signal.
#
# Round 12: marked REPETITIVE_STAYS_BAD_ON_RETRY -- same reasoning as
# Test 34, so the new fallback-retry mechanism doesn't quietly fix this
# sentence and undermine what this test is actually proving.
StubNLLBTranslator.reset()
good1_text_54 = "Aquesta és la primera frase bona en català per la prova de senyal aïllat."
bad_text_54 = "Aquesta és la frase dolenta que fallarà la comprovació de repetició en aquesta prova."
good2_text_54 = "Aquesta és la segona frase bona en català per la prova de senyal aïllat."
StubNLLBTranslator.FAIL_MODE = "repetition_one"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([bad_text_54])
StubNLLBTranslator.REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS = frozenset([bad_text_54])
cache, cache_dir = fresh_cache()
translator_54 = StubNLLBTranslator()
raw_54 = {"ivF": {"qA": " ".join([good1_text_54, bad_text_54, good2_text_54])}}
frozen_seg_54 = {
    "ivF::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": "ivF::qA::s000", "sentence_index": 0, "text_source": good1_text_54},
            {"sentence_id": "ivF::qA::s001", "sentence_index": 1, "text_source": bad_text_54},
            {"sentence_id": "ivF::qA::s002", "sentence_index": 2, "text_source": good2_text_54},
        ],
    },
}
_, sentences_out_54, report_54 = preprocess_v2.process(raw_54, preprocessor, cache, translator_54, similarity_scorer, frozen_segmentation=frozen_seg_54)

check("the two good sentences are NOT flagged", sentences_out_54["ivF::qA::s000"]["sentence_status"] != "FLAGGED" and sentences_out_54["ivF::qA::s002"]["sentence_status"] != "FLAGGED")
check("the one bad sentence (degenerate/repetitive output) IS flagged, even though it's outnumbered 2-to-1 by good neighbors in the same response", sentences_out_54["ivF::qA::s001"]["sentence_status"] == "FLAGGED")
check("sentence_translation_summary reports exactly one flagged sentence, correctly identified by ID", report_54["sentence_translation_summary"]["flagged_sentence_count"] == 1 and report_54["sentence_translation_summary"]["flagged_sentence_ids"] == ["ivF::qA::s001"])
check("translation_output_validity is False -- one bad sentence in three is still fatal, not averaged away", report_54["translation_output_validity"] is False)
check("translation_sanity_status is FAIL", report_54["translation_sanity_status"] == "FAIL")
check("STEP2_VALID is False", report_54["STEP2_VALID"] is False)
check("structural_validity stays True -- nothing was structurally lost, only flagged", report_54["structural_validity"] is True)
check("translation_completeness_validity stays True -- the sentence WAS translated, just flagged, not failed", report_54["translation_completeness_validity"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 55: per-sentence length_diagnostics -- diagnostic-only, correctly identifies the outlier sentence by ID ===")
# Added because a response-level length ratio can statistically absorb one
# sentence collapsing to a fraction of its source length while the rest of
# the response compensates -- this is the per-sentence version, and it is
# deliberately NOT part of the FLAGGED/fatal-quality condition (see the
# comment at its call site in preprocess_v2.py): source<->target length
# legitimately varies a lot sentence-to-sentence, so this stays REVIEW-only.
StubNLLBTranslator.reset()
long_source_55 = (
    "Aquesta és una frase molt llarga en català que hauria de produir una traducció "
    "desproporcionadament curta en aquesta prova de diagnòstic de longitud per frase."
)
short_target_55 = "Traducción corta."
normal_source_55 = (
    "Aquesta és una altra frase normal en català que hauria de tenir una traducció "
    "de longitud raonable per aquesta prova de diagnòstic."
)


class _LengthOutlierTranslator(StubNLLBTranslator):
    FAIL_MODE = "none"
    FAILS_REMAINING = 0
    EMPTY_FOR_TEXTS = frozenset()
    IDENTICAL_FOR_TEXTS = frozenset()
    REPETITIVE_FOR_TEXTS = frozenset()
    WRONG_LANGUAGE_FOR_TEXTS = frozenset()
    CALL_LOG = []

    def translate_batch(self, texts, source_lang, target_lang):
        _LengthOutlierTranslator.CALL_LOG.append((source_lang, target_lang, list(texts)))
        return [short_target_55 if t == long_source_55 else STUB_ES_OUTPUT for t in texts]


cache, cache_dir = fresh_cache()
translator_55 = _LengthOutlierTranslator()
raw_55 = {"ivO": {"qA": long_source_55 + " " + normal_source_55}}
frozen_seg_55 = {
    "ivO::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": "ivO::qA::s000", "sentence_index": 0, "text_source": long_source_55},
            {"sentence_id": "ivO::qA::s001", "sentence_index": 1, "text_source": normal_source_55},
        ],
    },
}
_, sentences_out_55, report_55 = preprocess_v2.process(raw_55, preprocessor, cache, translator_55, similarity_scorer, frozen_segmentation=frozen_seg_55)

ld_outlier = sentences_out_55["ivO::qA::s000"]["quality"]["length_diagnostics"]
ld_normal = sentences_out_55["ivO::qA::s001"]["quality"]["length_diagnostics"]
check("the drastically-shortened sentence's char_ratio is correctly computed and below the low bound", ld_outlier is not None and ld_outlier["char_ratio"] < preprocess_v2.LENGTH_RATIO_LOW)
check("that sentence is correctly flagged as a length_ratio_outlier", ld_outlier["length_ratio_outlier"] is True)
check("the normally-proportioned sentence is NOT flagged as a length_ratio_outlier", ld_normal is not None and ld_normal["length_ratio_outlier"] is False)
check(
    "length_diagnostics is diagnostic-only -- the outlier sentence's sentence_status is still FRESH_OK, not FLAGGED",
    sentences_out_55["ivO::qA::s000"]["sentence_status"] == "FRESH_OK",
)
check("sentence_translation_summary's corpus-wide rollup counts exactly this one outlier, by ID", report_55["sentence_translation_summary"]["sentence_length_ratio_outlier_count"] == 1 and report_55["sentence_translation_summary"]["sentence_length_ratio_outlier_ids"] == ["ivO::qA::s000"])
check("translation_output_validity stays True -- a length outlier alone never gates STEP2_VALID", report_55["translation_output_validity"] is True)
check("translation_sanity_status is REVIEW -- surfaced for a human, not fatal", report_55["translation_sanity_status"] == "REVIEW")
check("STEP2_VALID is True", report_55["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 56: a single NLLB model load, not two (run_preflight/main reuse) ===")
# Previously, run_preflight() constructed a real NLLBTranslator purely to
# verify the environment (tokenizer/model load + one real translation),
# then main() constructed a SECOND one for the actual corpus run --
# loading a ~600M-parameter model twice back-to-back on the exact
# low-memory Mac that had already crashed once under memory pressure. Now
# run_preflight() hands its already-loaded translator back for reuse.
preprocess_v2_preflight_passed, preprocess_v2_preflight_result = preprocess_v2.run_preflight(verbose=False, batch_size=4)
check("run_preflight succeeds against the stub translator", preprocess_v2_preflight_passed is True)
check(
    "run_preflight's result dict carries the constructed translator for reuse, only on success",
    isinstance(preprocess_v2_preflight_result.get("translator"), StubNLLBTranslator),
)
check("the reused translator was built with the batch_size this preflight was asked for", preprocess_v2_preflight_result["translator"].batch_size == 4)


class _AlwaysFailsToConstruct:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("stubbed: model construction failed")


_original_nllb_translator_for_test56 = preprocess_v2.NLLBTranslator
preprocess_v2.NLLBTranslator = _AlwaysFailsToConstruct
try:
    preflight_fail_passed, preflight_fail_result = preprocess_v2.run_preflight(verbose=False)
    check("run_preflight fails cleanly (returns False) when model construction raises", preflight_fail_passed is False)
    check("no 'translator' key is set in the result on a failed preflight -- a caller must never try to reuse a translator from a failed run", "translator" not in preflight_fail_result)
finally:
    preprocess_v2.NLLBTranslator = _original_nllb_translator_for_test56

_main_source_56 = inspect.getsource(preprocess_v2.main)
check(
    "main()'s own source never constructs an NLLBTranslator directly -- the only real construction site left is inside run_preflight/run_smoke_test",
    "NLLBTranslator(" not in _main_source_56,
)
check(
    "main() reuses preflight_result['translator'] instead of building a new one for the real run",
    'preflight_result["translator"]' in _main_source_56 or "preflight_result['translator']" in _main_source_56,
)


# ===========================================================================
print("\n=== Test 57: SemanticSimilarityScorer is genuinely lazy -- no model load until first real use ===")
# The docstring already claimed this before the eighth review, but
# __init__ eagerly constructed the real SentenceTransformer regardless --
# on the same low-memory Mac, that meant paying an embedding-model load
# every run even when the corpus had nothing for it to score (an all-
# Spanish-original corpus needs no ca<->es similarity at all). Now
# construction does nothing, and the first real use (`.available` or
# `.batch_cosine_similarity`) is what triggers the (at most once) load
# attempt. Uses a fake `sentence_transformers` module injected into
# sys.modules so this is deterministic and needs no network access either
# way, regardless of what's actually installed in this environment.
import types as _types  # noqa: E402 (test-local; keeps the module-level import list unchanged for the rest of the suite)

lazy_scorer_no_module_swap = _RealSemanticSimilarityScorer(model_name="never-actually-loaded-in-this-test")
check(
    "constructing SemanticSimilarityScorer does nothing -- no model, no load attempted yet",
    lazy_scorer_no_module_swap.model is None and lazy_scorer_no_module_swap.load_attempted is False,
)

_fake_st_module = _types.ModuleType("sentence_transformers")


class _FakeSentenceTransformerCtor:
    calls = []

    def __init__(self, model_name):
        _FakeSentenceTransformerCtor.calls.append(model_name)
        self.model_name = model_name


_fake_st_module.SentenceTransformer = _FakeSentenceTransformerCtor
_real_st_module_57 = sys.modules.get("sentence_transformers")
sys.modules["sentence_transformers"] = _fake_st_module
try:
    lazy_scorer_57 = _RealSemanticSimilarityScorer(model_name="fake-embedding-model-57")
    check("no load attempted immediately after construction", lazy_scorer_57.load_attempted is False and _FakeSentenceTransformerCtor.calls == [])
    first_available = lazy_scorer_57.available
    check("the FIRST .available access triggers exactly one load attempt, with the right model name", _FakeSentenceTransformerCtor.calls == ["fake-embedding-model-57"])
    check(".available reflects a successful load", first_available is True and lazy_scorer_57.model is not None)
    _ = lazy_scorer_57.available
    _ = lazy_scorer_57.available
    check("subsequent .available calls do not load again -- at most once, ever", _FakeSentenceTransformerCtor.calls == ["fake-embedding-model-57"])

    # Failure path: the (fake) import/construction itself raises.
    class _FakeSentenceTransformerCtorFails:
        def __init__(self, model_name):
            raise RuntimeError("stubbed: could not load embedding model")

    _fake_st_module.SentenceTransformer = _FakeSentenceTransformerCtorFails
    lazy_scorer_fail_57 = _RealSemanticSimilarityScorer(model_name="fake-embedding-model-that-fails")
    check("a load failure degrades gracefully -- available is False, not a raised exception", lazy_scorer_fail_57.available is False)
    check("unavailable_reason explains why", lazy_scorer_fail_57.unavailable_reason is not None and "could not load embedding model" in lazy_scorer_fail_57.unavailable_reason)
    check("load_attempted is True even after a failure -- a second call must not retry", lazy_scorer_fail_57.load_attempted is True)
finally:
    if _real_st_module_57 is not None:
        sys.modules["sentence_transformers"] = _real_st_module_57
    else:
        sys.modules.pop("sentence_transformers", None)

# Integration: a corpus with nothing for the similarity scorer to do must
# never force the lazy load just to fill in the report's summary fields --
# this is the exact call site (translation_quality_summary construction)
# an eighth-review-style regression could silently reintroduce by reading
# `.available` instead of `.load_attempted` there.
all_spanish_raw_57 = {"ivS": {"q1": "Esta es una respuesta completamente en español, sin nada en catalán en absoluto para esta prueba de carga diferida."}}
lazy_scorer_integration_57 = _RealSemanticSimilarityScorer(model_name="should-never-be-loaded-this-run")
cache, cache_dir = fresh_cache()
translator_57 = StubNLLBTranslator()
_, _, report_57 = preprocess_v2.process(all_spanish_raw_57, preprocessor, cache, translator_57, lazy_scorer_integration_57)
check(
    "an all-Spanish corpus needs no similarity scoring at all, and process() never forces the lazy load just to report on it",
    lazy_scorer_integration_57.load_attempted is False,
)
check("similarity_model_available correctly reports False without forcing a load", report_57["translation_quality_summary"]["similarity_model_available"] is False)
check(
    "the unavailable_reason correctly explains WHY (not needed this run, not a real failure) rather than misleadingly implying a load was tried and failed",
    report_57["translation_quality_summary"]["similarity_model_unavailable_reason"] == "not_needed_for_this_run_no_similarity_pairs_computed",
)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 58: Round 10 -- raised short-text language-detection threshold ===")

# Pins the specific values chosen this round (see preprocess_v2.py's
# comment above these constants for the full data-driven tradeoff
# analysis, against the real 950-response corpus run, that justified
# 50/8 over the prior 30/5).
check("MIN_CHARS_FOR_DIRECT_DETECTION was raised to 50 (Round 10)", preprocess_v2.MIN_CHARS_FOR_DIRECT_DETECTION == 50)
check("MIN_WORDS_FOR_DIRECT_DETECTION was raised to 8 (Round 10)", preprocess_v2.MIN_WORDS_FOR_DIRECT_DETECTION == 8)

# A short, real, grammatically-correct Spanish phrase that would have
# cleared the OLD 30-char/5-word bars (and so would have been directly
# language-assessed under the old thresholds) but does not clear the NEW
# 50/8 bars. This is exactly the shape of the 105 real false-positive
# flags found in the actual corpus run: short, informal, correct Spanish
# that langdetect's statistical model misjudges at these lengths.
word_boundary_text = "No sé, la verdad, era bastante raro."
check("the probe text would have cleared the OLD char bar (30)", len(word_boundary_text) >= 30)
check("the probe text would have cleared the OLD word bar (5)", preprocess_v2._word_count(word_boundary_text) >= 5)
check("...but does NOT clear the NEW char bar (50)", len(word_boundary_text) < preprocess_v2.MIN_CHARS_FOR_DIRECT_DETECTION)
check("...and does NOT clear the NEW word bar (8)", preprocess_v2._word_count(word_boundary_text) < preprocess_v2.MIN_WORDS_FOR_DIRECT_DETECTION)
check("_is_sufficiently_long is now False for this old-threshold-passing text", preprocess_v2._is_sufficiently_long(word_boundary_text) is False)

boundary_lang_check = preprocess_v2.check_target_language(word_boundary_text, "es")
check("...so check_target_language now skips it entirely (NOT_ASSESSED_SHORT_TEXT)", boundary_lang_check["status"] == "NOT_ASSESSED_SHORT_TEXT")
check("...and passed is None -- never a false-positive False", boundary_lang_check["passed"] is None)

# The raise must not blind the check to REAL mismatches or stop assessing
# real long, correct text -- it only exempts short text. Reuses the
# suite's existing real-language stub outputs (both well over 50
# chars / 8 words) rather than inventing new ones.
en_check_58 = preprocess_v2.check_target_language(STUB_EN_OUTPUT, "es")
check("a genuinely long, wrong-language text is STILL assessed and fails (threshold raise did not blind real mismatches)",
      en_check_58["status"] == "ASSESSED" and en_check_58["passed"] is False)
es_check_58 = preprocess_v2.check_target_language(STUB_ES_OUTPUT, "es")
check("a genuinely long, correct translation is still assessed and passes",
      es_check_58["status"] == "ASSESSED" and es_check_58["passed"] is True)


# ===========================================================================
print("\n=== Test 59: Round 12 -- anti-repetition params are RETRY-only; PRIMARY config reverted to pre-Round-10 ===")

# Round 10 originally added repetition_penalty/no_repeat_ngram_size
# directly to GENERATION_PARAMS (applied to every sentence) and bumped
# GENERATION_VERSION, which would have forced the real corpus's entire
# ~5.8k-entry cache to be discarded just to fix ~37 sentences. Round 12
# replaced that with a targeted fallback-retry mechanism (see Test 63/64)
# and reverted GENERATION_PARAMS/GENERATION_VERSION to their exact
# pre-Round-10 values, so the real corpus's existing cache is fully valid
# again. This test locks in that reversion plus the new RETRY_* constants.
check("GENERATION_PARAMS no longer includes repetition_penalty (reverted -- primary calls stay unmodified)", "repetition_penalty" not in preprocess_v2.GENERATION_PARAMS)
check("GENERATION_PARAMS no longer includes no_repeat_ngram_size (reverted)", "no_repeat_ngram_size" not in preprocess_v2.GENERATION_PARAMS)
check(
    "GENERATION_VERSION is back to its pre-Round-10 value -- no 'antirep' marker -- so the real corpus's "
    "existing cache (built entirely under this exact configuration) is fully valid again",
    "antirep" not in preprocess_v2.GENERATION_VERSION,
)
check(
    "the primary decoding parameters (beam search, determinism) are otherwise untouched",
    preprocess_v2.GENERATION_PARAMS.get("num_beams") == 4
    and preprocess_v2.GENERATION_PARAMS.get("do_sample") is False
    and preprocess_v2.GENERATION_PARAMS.get("max_length") == 512,
)

check("RETRY_GENERATION_PARAMS carries repetition_penalty", "repetition_penalty" in preprocess_v2.RETRY_GENERATION_PARAMS)
check("RETRY_GENERATION_PARAMS carries no_repeat_ngram_size", "no_repeat_ngram_size" in preprocess_v2.RETRY_GENERATION_PARAMS)
check(
    "repetition_penalty is a genuine (>1.0) penalty -- 1.0 or below would be a no-op or would reward repetition",
    preprocess_v2.RETRY_GENERATION_PARAMS["repetition_penalty"] > 1.0,
)
check(
    "no_repeat_ngram_size is set larger than REPETITION_NGRAM_SIZE (the 3-gram size check_degenerate_output() "
    "flags on) -- a hard generation-time block at exactly 3 would forbid even short, legitimate repeated "
    "phrases ('el tema de ... el tema de') that show up naturally in this corpus's rambling interview speech",
    preprocess_v2.RETRY_GENERATION_PARAMS["no_repeat_ngram_size"] > preprocess_v2.REPETITION_NGRAM_SIZE,
)
check(
    "RETRY_GENERATION_PARAMS otherwise matches the primary decoding settings (beam search, determinism, max_length)",
    preprocess_v2.RETRY_GENERATION_PARAMS.get("num_beams") == preprocess_v2.GENERATION_PARAMS.get("num_beams")
    and preprocess_v2.RETRY_GENERATION_PARAMS.get("do_sample") == preprocess_v2.GENERATION_PARAMS.get("do_sample")
    and preprocess_v2.RETRY_GENERATION_PARAMS.get("max_length") == preprocess_v2.GENERATION_PARAMS.get("max_length"),
)
check(
    "RETRY_GENERATION_VERSION is a DISTINCT cache namespace from GENERATION_VERSION -- never the same string",
    preprocess_v2.RETRY_GENERATION_VERSION != preprocess_v2.GENERATION_VERSION
    and "antirep" in preprocess_v2.RETRY_GENERATION_VERSION,
)


# ===========================================================================
print("\n=== Test 60: Round 11 -- a language mismatch among good neighbors in the SAME response no longer fails STEP2_VALID ===")
# The direct multi-sentence mirror of the old Test 54 scenario, now with
# the OPPOSITE expected outcome: a wrong-language sentence sitting among
# good neighbors must NOT be marked FLAGGED and must NOT fail STEP2_VALID
# any more -- only degenerate output does that (see Test 54's current
# form). It should still be visible, just non-fatal: named in the new
# per-sentence sentence_language_mismatch_ids rollup and surfaced at
# translation_sanity_status="REVIEW".
StubNLLBTranslator.reset()
good1_text_60 = "Aquesta és la primera frase bona en català per la prova de senyal aïllat."
mismatch_text_60 = "Aquesta és la frase que es tradueix a un idioma incorrecte en aquesta prova."
good2_text_60 = "Aquesta és la segona frase bona en català per la prova de senyal aïllat."
StubNLLBTranslator.FAIL_MODE = "wrong_language_one"
StubNLLBTranslator.WRONG_LANGUAGE_FOR_TEXTS = frozenset([mismatch_text_60])
cache, cache_dir = fresh_cache()
translator_60 = StubNLLBTranslator()
raw_60 = {"ivG": {"qA": " ".join([good1_text_60, mismatch_text_60, good2_text_60])}}
frozen_seg_60 = {
    "ivG::qA": {
        "source_language": "ca",
        "sentences": [
            {"sentence_id": "ivG::qA::s000", "sentence_index": 0, "text_source": good1_text_60},
            {"sentence_id": "ivG::qA::s001", "sentence_index": 1, "text_source": mismatch_text_60},
            {"sentence_id": "ivG::qA::s002", "sentence_index": 2, "text_source": good2_text_60},
        ],
    },
}
_, sentences_out_60, report_60 = preprocess_v2.process(raw_60, preprocessor, cache, translator_60, similarity_scorer, frozen_segmentation=frozen_seg_60)

check(
    "none of the three sentences is FLAGGED -- a language mismatch alone no longer flags a sentence",
    all(sentences_out_60[f"ivG::qA::s{i:03d}"]["sentence_status"] != "FLAGGED" for i in range(3)),
)
check(
    "the mismatched sentence's OWN target_language_check still correctly recorded passed=False (still computed, just not fatal)",
    sentences_out_60["ivG::qA::s001"]["quality"]["target_language_check"]["passed"] is False,
)
check(
    "sentence_translation_summary's new sentence_language_mismatch_ids correctly names exactly the mismatched sentence",
    report_60["sentence_translation_summary"]["sentence_language_mismatch_count"] == 1
    and report_60["sentence_translation_summary"]["sentence_language_mismatch_ids"] == ["ivG::qA::s001"],
)
check("sentence_translation_summary reports zero FLAGGED sentences", report_60["sentence_translation_summary"]["flagged_sentence_count"] == 0)
check("translation_output_validity is True", report_60["translation_output_validity"] is True)
check("translation_sanity_status is REVIEW, not FAIL", report_60["translation_sanity_status"] == "REVIEW")
check("STEP2_VALID is True", report_60["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 61: Round 11 -- REPETITION_MIN_REPEATS raised to 6, at the exact boundary ===")
check("REPETITION_MIN_REPEATS was raised to 6 (Round 11)", preprocess_v2.REPETITION_MIN_REPEATS == 6)

# A short natural phrase repeating exactly 5 times -- below the new bar --
# must NOT be flagged, the direct regression test for the 4 real false
# positives (natural phrases repeating exactly 4x, now with margin to 5)
# found in the first real corpus run.
five_reps_text = " ".join(["tema de trabajo"] * 5)
degenerate_five = preprocess_v2.check_degenerate_output("font catalana original prou llarga", five_reps_text)
check("a 3-gram repeating 5 times (below the new bar) is NOT flagged as degenerate", degenerate_five["flagged"] is False and degenerate_five["repetition_flag"] is False)

# The same phrase repeating exactly 6 times -- AT the new bar -- must
# still be flagged. Real repetition-loop failures repeated their worst
# 3-gram far more than this (15-93+ times), so this is a deliberately
# tight boundary check, not just a comfortably-large example.
six_reps_text = " ".join(["tema de trabajo"] * 6)
degenerate_six = preprocess_v2.check_degenerate_output("font catalana original prou llarga", six_reps_text)
check("the same 3-gram repeating 6 times (at the new bar) IS flagged as degenerate", degenerate_six["flagged"] is True and degenerate_six["repetition_flag"] is True)
check("...with the correct repeated n-gram and count recorded", degenerate_six["repetition_detail"]["count"] == 6)


# ===========================================================================
print("\n=== Test 62: Round 12 -- a fallback retry that RESOLVES a repetition-flagged sentence, end to end ===")
# The core new mechanism: a PRIMARY translation that check_degenerate_
# output() flags as a repetition loop gets exactly one targeted retry
# under RETRY_GENERATION_PARAMS, and when that retry is clean, its output
# REPLACES the sentence's text_es -- the sentence is no longer FLAGGED,
# and the full provenance trail (primary vs. retry, resolved, which
# generation_version was actually selected) is recorded.
StubNLLBTranslator.reset()
resolve_text_62 = "Aquest text es tradueix de manera repetitiva pero es corregeix en el reintent."
StubNLLBTranslator.FAIL_MODE = "repetition_one"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([resolve_text_62])
# (not in REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS -- default stub behavior
# is that a retry call fixes it, which is exactly what this test wants.)
cache, cache_dir = fresh_cache()
translator_62 = StubNLLBTranslator()
raw_62 = {"iv62": {"q1": resolve_text_62}}
frozen_seg_62 = {
    "iv62::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv62::q1::s000", "sentence_index": 0, "text_source": resolve_text_62}],
    },
}
_, sentences_out_62, report_62 = preprocess_v2.process(raw_62, preprocessor, cache, translator_62, similarity_scorer, frozen_segmentation=frozen_seg_62)
sent_62 = sentences_out_62["iv62::q1::s000"]
retry_record_62 = sent_62["quality"]["repetition_fallback_retry"]

check("the sentence is NOT flagged -- the retry's clean output was selected", sent_62["sentence_status"] != "FLAGGED")
check("text_es was REPLACED with the retry's output, not the original repetitive one", sent_62["text_es"] == STUB_ES_OUTPUT)
check("the FINAL degenerate_output (on the selected text) is clean", sent_62["quality"]["degenerate_output"]["flagged"] is False)
check("repetition_fallback_retry.attempted is True", retry_record_62["attempted"] is True)
check("repetition_fallback_retry.resolved is True", retry_record_62["resolved"] is True)
check("repetition_fallback_retry.cache_status is FRESH (first time this text was retried)", retry_record_62["cache_status"] == "FRESH")
check(
    "the ORIGINAL primary diagnostic is preserved for audit, and correctly shows the repetition that triggered the retry",
    retry_record_62["primary_degenerate_output"]["repetition_flag"] is True,
)
check("primary_generation_version matches the module's current primary GENERATION_VERSION", retry_record_62["primary_generation_version"] == preprocess_v2.GENERATION_VERSION)
check("retry_generation_version matches RETRY_GENERATION_VERSION", retry_record_62["retry_generation_version"] == preprocess_v2.RETRY_GENERATION_VERSION)
check("selected_generation_version correctly names the RETRY version (its output is what was kept)", retry_record_62["selected_generation_version"] == preprocess_v2.RETRY_GENERATION_VERSION)
check(
    "exactly one PRIMARY call and one RETRY call were made for this text (no wasted extra calls)",
    sum(1 for call in StubNLLBTranslator.CALL_LOG if resolve_text_62 in call[2] and call[3] is None) == 1
    and sum(1 for call in StubNLLBTranslator.CALL_LOG if resolve_text_62 in call[2] and call[3] is not None) == 1,
)
check(
    "sentence_translation_summary correctly counts this as one attempted, one resolved, zero unresolved",
    report_62["sentence_translation_summary"]["repetition_fallback_retry_attempted_count"] == 1
    and report_62["sentence_translation_summary"]["repetition_fallback_retry_resolved_count"] == 1
    and report_62["sentence_translation_summary"]["repetition_fallback_retry_unresolved_count"] == 0
    and report_62["sentence_translation_summary"]["repetition_fallback_retry_resolved_ids"] == ["iv62::q1::s000"],
)
check("translation_output_validity is True -- the repaired sentence no longer gates STEP2_VALID", report_62["translation_output_validity"] is True)
check("STEP2_VALID is True", report_62["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 63: Round 12 -- a fallback retry that does NOT resolve the sentence keeps the ORIGINAL primary text ===")
# The mirror of Test 62: when the retry itself is still bad (or fails),
# text_es must NOT be silently replaced with a second, possibly-different
# but still-wrong translation -- it stays exactly what the primary call
# produced, and the sentence correctly stays FLAGGED.
StubNLLBTranslator.reset()
unresolved_text_63 = "Aquest altre text tambe es tradueix de manera repetitiva i el reintent tampoc ho arregla."
StubNLLBTranslator.FAIL_MODE = "repetition_one"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([unresolved_text_63])
StubNLLBTranslator.REPETITIVE_STAYS_BAD_ON_RETRY_FOR_TEXTS = frozenset([unresolved_text_63])
cache, cache_dir = fresh_cache()
translator_63 = StubNLLBTranslator()
raw_63 = {"iv63": {"q1": unresolved_text_63}}
frozen_seg_63 = {
    "iv63::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv63::q1::s000", "sentence_index": 0, "text_source": unresolved_text_63}],
    },
}
_, sentences_out_63, report_63 = preprocess_v2.process(raw_63, preprocessor, cache, translator_63, similarity_scorer, frozen_segmentation=frozen_seg_63)
sent_63 = sentences_out_63["iv63::q1::s000"]
retry_record_63 = sent_63["quality"]["repetition_fallback_retry"]

check("the sentence IS still flagged -- the retry did not produce a clean result", sent_63["sentence_status"] == "FLAGGED")
check("text_es is UNCHANGED -- still the original primary (repetitive) output, not a second bad translation", sent_63["text_es"] == STUB_REPETITIVE_OUTPUT)
check("repetition_fallback_retry.attempted is True", retry_record_63["attempted"] is True)
check("repetition_fallback_retry.resolved is False", retry_record_63["resolved"] is False)
check("selected_generation_version correctly names the PRIMARY version (retry's output was NOT selected)", retry_record_63["selected_generation_version"] == preprocess_v2.GENERATION_VERSION)
check(
    "sentence_translation_summary correctly counts this as one attempted, zero resolved, one unresolved",
    report_63["sentence_translation_summary"]["repetition_fallback_retry_attempted_count"] == 1
    and report_63["sentence_translation_summary"]["repetition_fallback_retry_resolved_count"] == 0
    and report_63["sentence_translation_summary"]["repetition_fallback_retry_unresolved_count"] == 1
    and report_63["sentence_translation_summary"]["repetition_fallback_retry_unresolved_ids"] == ["iv63::q1::s000"],
)
check("translation_output_validity is False -- an unresolved repetition loop is still fatal", report_63["translation_output_validity"] is False)
check("STEP2_VALID is False", report_63["STEP2_VALID"] is False)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 64: Round 12 -- the retry cache namespace is reused across sentences sharing the same bad source text ===")
# This corpus's real repetition-loop cases showed heavy cache reuse (e.g.
# "No, no." behind 14 flagged sentence IDs -- see STEP2_NLLB_CHANGES.md's
# Round 10 diagnosis). Two DIFFERENT sentences with the IDENTICAL source
# text must only trigger ONE real retry call between them -- the second
# is served from the retry cache namespace, never re-sent to NLLB.
StubNLLBTranslator.reset()
shared_bad_text_64 = "Aquest text compartit es tradueix de manera repetitiva en dues respostes diferents."
StubNLLBTranslator.FAIL_MODE = "repetition_one"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([shared_bad_text_64])
cache, cache_dir = fresh_cache()
translator_64 = StubNLLBTranslator()
raw_64 = {"iv64a": {"q1": shared_bad_text_64}, "iv64b": {"q1": shared_bad_text_64}}
frozen_seg_64 = {
    "iv64a::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv64a::q1::s000", "sentence_index": 0, "text_source": shared_bad_text_64}],
    },
    "iv64b::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv64b::q1::s000", "sentence_index": 0, "text_source": shared_bad_text_64}],
    },
}
_, sentences_out_64, report_64 = preprocess_v2.process(raw_64, preprocessor, cache, translator_64, similarity_scorer, frozen_segmentation=frozen_seg_64)

check(
    "both sentences resolved -- the retry's clean output was selected for each",
    sentences_out_64["iv64a::q1::s000"]["sentence_status"] != "FLAGGED"
    and sentences_out_64["iv64b::q1::s000"]["sentence_status"] != "FLAGGED",
)
retry_statuses_64 = sorted(
    sentences_out_64[sid]["quality"]["repetition_fallback_retry"]["cache_status"]
    for sid in ("iv64a::q1::s000", "iv64b::q1::s000")
)
check("exactly one FRESH retry and one CACHE_HIT retry, across the two sentences (in either order)", retry_statuses_64 == ["CACHE_HIT", "FRESH"])
check(
    "only ONE real retry-mode translate_batch call was made for this text, despite two sentences needing it",
    sum(1 for call in StubNLLBTranslator.CALL_LOG if shared_bad_text_64 in call[2] and call[3] is not None) == 1,
)
check("STEP2_VALID is True", report_64["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 65: Round 12 -- NLLBTranslator.translate_batch's generation_params override (real class, no model load) ===")


class _FakeTensorForGeneration:
    """Duck-types just `.to(device)` -- returns itself, since none of this
    test cares about actual tensor placement."""

    def to(self, device):
        return self


class _FakeTokenizerForGeneration:
    lang_code_to_id = {"spa_Latn": 42, "cat_Latn": 7}

    def __init__(self):
        self.src_lang = None

    def __call__(self, text_or_texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        if isinstance(text_or_texts, str):
            # count_tokens()'s call shape (used by _assert_within_token_limit,
            # which translate_batch calls before ever reaching generate()).
            return {"input_ids": list(range(len(text_or_texts.split()) + 2))}
        # translate_batch()'s own batch-tokenize call shape.
        return {"input_ids": _FakeTensorForGeneration(), "attention_mask": _FakeTensorForGeneration()}

    def convert_tokens_to_ids(self, token):
        raise AssertionError("should not be called -- lang_code_to_id has the key")

    def batch_decode(self, generated, skip_special_tokens=True):
        return generated


class _FakeModelForGeneration:
    def __init__(self):
        self.generate_calls = []  # each entry is the full kwargs dict generate() was called with

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return ["<fake-generated>"]


class _FakeSelfForGeneration:
    """Duck-typed fake `self` for exercising the REAL, unmodified
    NLLBTranslator.translate_batch directly -- never constructs a real
    NLLBTranslator (whose __init__ downloads/loads the real model)."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = "cpu"

    count_tokens = _RealNLLBTranslator.count_tokens
    _assert_within_token_limit = _RealNLLBTranslator._assert_within_token_limit
    _forced_bos_token_id = _RealNLLBTranslator._forced_bos_token_id


fake_model_65 = _FakeModelForGeneration()
fake_self_65 = _FakeSelfForGeneration(_FakeTokenizerForGeneration(), fake_model_65)

_RealNLLBTranslator.translate_batch(fake_self_65, ["hola"], "ca", "es")
check("a default call (no generation_params) uses the module-level GENERATION_PARAMS", all(
    fake_model_65.generate_calls[-1].get(k) == v for k, v in preprocess_v2.GENERATION_PARAMS.items()
))
check(
    "...and does NOT carry the retry-only anti-repetition keys",
    "repetition_penalty" not in fake_model_65.generate_calls[-1]
    and "no_repeat_ngram_size" not in fake_model_65.generate_calls[-1],
)

_RealNLLBTranslator.translate_batch(fake_self_65, ["hola"], "ca", "es", generation_params=preprocess_v2.RETRY_GENERATION_PARAMS)
check("passing generation_params overrides what reaches model.generate() for that one call", all(
    fake_model_65.generate_calls[-1].get(k) == v for k, v in preprocess_v2.RETRY_GENERATION_PARAMS.items()
))
check(
    "...and the override is genuinely per-call -- the module-level GENERATION_PARAMS itself is untouched",
    "repetition_penalty" not in preprocess_v2.GENERATION_PARAMS,
)
check("forced_bos_token_id was still correctly resolved for the target language in both calls", all(
    call.get("forced_bos_token_id") == 42 for call in fake_model_65.generate_calls
))


# ===========================================================================
print("\n=== Test 66: Round 12 (external review fix) -- a resolved repair correctly updates the TOP-LEVEL translation_provenance, not just the nested selected_generation_version ===")
# Before this fix, `provenance` (CACHE_HIT/FRESH_OK/etc, describing only
# the PRIMARY attempt) was never reassigned when a retry resolved the
# sentence -- sentences_out[...]["translation_provenance"] and every
# corpus-wide provenance rollup silently kept the stale PRIMARY value
# even though the sentence's actual final text_es came from the retry.
StubNLLBTranslator.reset()
repair_text_66 = "Aquest text primari ja estava en cache pero era repetitiu i cal reparar-lo."
StubNLLBTranslator.FAIL_MODE = "repetition_one"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([repair_text_66])
cache, cache_dir = fresh_cache()
translator_66 = StubNLLBTranslator()
# Pre-populate the PRIMARY cache entry directly, so this sentence's
# primary translation is served as CACHE_HIT (not FRESH_OK) -- exactly
# the scenario the external review's own bug report used ("CACHE_HIT"
# left in place after a successful repair).
cache.set(
    repair_text_66, "ca", "es", STUB_REPETITIVE_OUTPUT,
    model_name=translator_66.model_name, generation_version=preprocess_v2.GENERATION_VERSION,
)
raw_66 = {"iv66": {"q1": repair_text_66}}
frozen_seg_66 = {
    "iv66::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv66::q1::s000", "sentence_index": 0, "text_source": repair_text_66}],
    },
}
_, sentences_out_66, report_66 = preprocess_v2.process(raw_66, preprocessor, cache, translator_66, similarity_scorer, frozen_segmentation=frozen_seg_66)
sent_66 = sentences_out_66["iv66::q1::s000"]
retry_record_66 = sent_66["quality"]["repetition_fallback_retry"]

check(
    "the primary translation really was served as CACHE_HIT (not FRESH_OK) before any repair",
    retry_record_66["primary_translation_provenance"] == "CACHE_HIT",
)
check(
    "the TOP-LEVEL translation_provenance is REPETITION_REPAIRED, not left stale as CACHE_HIT",
    sent_66["translation_provenance"] == "REPETITION_REPAIRED",
)
check("sentence_status also correctly reflects the repair, not the stale primary provenance", sent_66["sentence_status"] == "REPETITION_REPAIRED")
check(
    "the nested record's selected_translation_provenance matches the top-level field exactly",
    retry_record_66["selected_translation_provenance"] == sent_66["translation_provenance"] == "REPETITION_REPAIRED",
)
check(
    "the corpus-wide by_translation_provenance rollup counts this sentence under REPETITION_REPAIRED, not CACHE_HIT",
    report_66["sentence_translation_summary"]["by_translation_provenance"].get("REPETITION_REPAIRED") == 1
    and report_66["sentence_translation_summary"]["by_translation_provenance"].get("CACHE_HIT", 0) == 0,
)
check("the flat repetition_repaired_translations rollup counts it too", report_66["repetition_repaired_translations"] == 1)
check("the nested runtime_info mirror agrees", report_66["runtime_info"]["repetition_repaired_translations"] == 1)
check("REPETITION_REPAIRED is a distinct value from the pre-existing pass-1 RETRY_OK label (no collision)", "REPETITION_REPAIRED" != "RETRY_OK")
check("STEP2_VALID is True -- the repaired sentence no longer gates validity", report_66["STEP2_VALID"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 67: Round 12 (external review fix) -- a blank/empty fallback retry result is never cached and is reported FAILED ===")


class _BlankRetryTranslator:
    model_name = "fake-model-67"

    def translate_batch(self, texts, source_lang, target_lang, generation_params=None):
        return [""]  # blank, even though this is a "successful" call (no exception)


cache_67, cache_dir_67 = fresh_cache()
blank_text_67 = "Aquest text produeix una sortida buida en el reintent."
result_67 = preprocess_v2.attempt_repetition_fallback_retry(blank_text_67, _BlankRetryTranslator(), cache_67)
check("a blank retry result is reported as FAILED, not a success", result_67["cache_status"] == "FAILED")
check("text_es is None for a blank retry result", result_67["text_es"] is None)
check(
    "the blank result was NEVER cached under the retry namespace",
    not cache_67.contains(blank_text_67, "ca", "es", "fake-model-67", preprocess_v2.RETRY_GENERATION_VERSION),
)
shutil.rmtree(cache_dir_67, ignore_errors=True)

# End-to-end: the same scenario through process() itself -- the sentence
# correctly stays FLAGGED (same observable outcome as an unresolved retry
# in Test 63), and specifically because the retry came back blank, not
# because it merely "stayed repetitive".
StubNLLBTranslator.reset()
blank_e2e_text_67 = "Un altre text que en el reintent nomes produeix una cadena buida."
StubNLLBTranslator.FAIL_MODE = "repetition_one"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([blank_e2e_text_67])
StubNLLBTranslator.REPETITIVE_RETRY_RETURNS_BLANK_FOR_TEXTS = frozenset([blank_e2e_text_67])
cache_67b, cache_dir_67b = fresh_cache()
translator_67b = StubNLLBTranslator()
raw_67b = {"iv67b": {"q1": blank_e2e_text_67}}
frozen_seg_67b = {
    "iv67b::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv67b::q1::s000", "sentence_index": 0, "text_source": blank_e2e_text_67}],
    },
}
_, sentences_out_67b, report_67b = preprocess_v2.process(raw_67b, preprocessor, cache_67b, translator_67b, similarity_scorer, frozen_segmentation=frozen_seg_67b)
sent_67b = sentences_out_67b["iv67b::q1::s000"]
retry_record_67b = sent_67b["quality"]["repetition_fallback_retry"]
check("end to end: the sentence stays FLAGGED when its retry comes back blank", sent_67b["sentence_status"] == "FLAGGED")
check(
    "end to end: text_es is unchanged -- still the original primary output, never replaced with blank",
    sent_67b["text_es"] == STUB_REPETITIVE_OUTPUT,
)
check("end to end: repetition_fallback_retry.cache_status is FAILED, not FRESH", retry_record_67b["cache_status"] == "FAILED")
check("end to end: resolved is False", retry_record_67b["resolved"] is False)
check(
    "end to end: nothing was cached under the retry namespace for this text -- a future run will try again, not silently serve blank",
    not cache_67b.contains(blank_e2e_text_67, "ca", "es", translator_67b.model_name, preprocess_v2.RETRY_GENERATION_VERSION),
)
shutil.rmtree(cache_dir_67b, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 68: Round 12 (external review fix) -- a successful fallback retry is persisted to disk immediately, not left for a later batch save ===")


class _OneShotRetryTranslator:
    model_name = "fake-model-68"

    def translate_batch(self, texts, source_lang, target_lang, generation_params=None):
        return [STUB_ES_OUTPUT]


cache_68, cache_dir_68 = fresh_cache()
persist_text_68 = "Aquest text de reintent s'ha de desar immediatament al disc."
result_68 = preprocess_v2.attempt_repetition_fallback_retry(persist_text_68, _OneShotRetryTranslator(), cache_68)
check("the retry itself reports FRESH (a genuine new call, not a cache hit)", result_68["cache_status"] == "FRESH")

# Simulate a crash right after this call: open a BRAND NEW TranslationCache
# instance pointed at the SAME on-disk file, without ever calling save()
# on cache_68 again. If attempt_repetition_fallback_retry() really did
# call cache.save() itself (not just cache.set(), which only updates the
# in-memory dict), the entry must already be durably on disk.
reloaded_cache_68 = preprocess_v2.TranslationCache(cache_68.cache_path)
check(
    "the retry-repaired translation is visible from a FRESH TranslationCache instance reading the same file "
    "-- proof it was actually persisted, not just held in memory",
    reloaded_cache_68.contains(persist_text_68, "ca", "es", "fake-model-68", preprocess_v2.RETRY_GENERATION_VERSION),
)
check(
    "the reloaded record's content matches what was generated",
    reloaded_cache_68.get_text(persist_text_68, "ca", "es", "fake-model-68", preprocess_v2.RETRY_GENERATION_VERSION) == STUB_ES_OUTPUT,
)
shutil.rmtree(cache_dir_68, ignore_errors=True)


# ===========================================================================
print("\n=== Test 69: Round 12 (external review fix) -- require_warm_primary_cache refuses to proceed on incomplete "
      "primary-cache coverage, and does not block once the cache is genuinely warm ===")
StubNLLBTranslator.reset()
gate_text_69 = "Aquesta frase ja hauria d'estar traduida abans de tornar a executar el programa."
raw_69 = {"iv69": {"q1": gate_text_69}}
frozen_seg_69 = {
    "iv69::q1": {
        "source_language": "ca",
        "sentences": [{"sentence_id": "iv69::q1::s000", "sentence_index": 0, "text_source": gate_text_69}],
    },
}

# --- Negative case: cache is cold (nothing pre-populated), gate is ON. ---
cold_cache_69, cold_cache_dir_69 = fresh_cache()
translator_69a = StubNLLBTranslator()
_, sentences_out_69a, report_69a = preprocess_v2.process(
    raw_69, preprocessor, cold_cache_69, translator_69a, similarity_scorer,
    frozen_segmentation=frozen_seg_69, require_warm_primary_cache=True,
)
check("a cold cache with the gate on: process() reports translation_aborted_early", report_69a["translation_aborted_early"] is True)
check("a cold cache with the gate on: abort_reason names the coverage gate", "primary cache coverage" in (report_69a["abort_reason"] or ""))
check("a cold cache with the gate on: primary_cache_coverage correctly reports missing=1", report_69a["primary_cache_coverage"]["missing"] == 1)
check(
    "a cold cache with the gate on: SAFE TO REUSE PRIMARY CACHE is False",
    report_69a["primary_cache_coverage"]["safe_to_reuse_primary_cache"] is False,
)
check(
    "a cold cache with the gate on: ZERO NLLB calls were made -- the gate tripped before any translate_batch call",
    len(StubNLLBTranslator.CALL_LOG) == 0,
)
check(
    "a cold cache with the gate on: the sentence is recorded as FAILED, not silently translated anyway",
    sentences_out_69a["iv69::q1::s000"]["translation_provenance"] == "FAILED",
)
check("a cold cache with the gate on: STEP2_VALID is False", report_69a["STEP2_VALID"] is False)
shutil.rmtree(cold_cache_dir_69, ignore_errors=True)
StubNLLBTranslator.reset()

# --- Positive case: cache IS warm (pre-populated), gate is ON -- must NOT block. ---
warm_cache_69, warm_cache_dir_69 = fresh_cache()
translator_69b = StubNLLBTranslator()
warm_cache_69.set(
    gate_text_69, "ca", "es", STUB_ES_OUTPUT,
    model_name=translator_69b.model_name, generation_version=preprocess_v2.GENERATION_VERSION,
)
_, sentences_out_69b, report_69b = preprocess_v2.process(
    raw_69, preprocessor, warm_cache_69, translator_69b, similarity_scorer,
    frozen_segmentation=frozen_seg_69, require_warm_primary_cache=True,
)
check("a warm cache with the gate on: the run is NOT aborted", report_69b["translation_aborted_early"] is False)
check(
    "a warm cache with the gate on: coverage correctly reports missing=0 / SAFE=True",
    report_69b["primary_cache_coverage"]["missing"] == 0 and report_69b["primary_cache_coverage"]["safe_to_reuse_primary_cache"] is True,
)
check(
    "a warm cache with the gate on: the sentence is served as CACHE_HIT, exactly as normal",
    sentences_out_69b["iv69::q1::s000"]["translation_provenance"] == "CACHE_HIT",
)
check(
    "a warm cache with the gate on: no ca->es (primary-direction) NLLB calls were needed -- it really was all "
    "cache hits (any CALL_LOG entries here are the unrelated es->ca round-trip diagnostic, which legitimately "
    "still runs against the cached translation)",
    all(not (call[0] == "ca" and call[1] == "es") for call in StubNLLBTranslator.CALL_LOG),
)
check("a warm cache with the gate on: STEP2_VALID is True", report_69b["STEP2_VALID"] is True)
shutil.rmtree(warm_cache_dir_69, ignore_errors=True)
StubNLLBTranslator.reset()

# --- Default behavior case: gate OFF (the default for every existing caller) --
# a cold cache is translated normally, unaffected by this round's change. ---
cold_default_cache_69, cold_default_cache_dir_69 = fresh_cache()
translator_69c = StubNLLBTranslator()
_, sentences_out_69c, report_69c = preprocess_v2.process(
    raw_69, preprocessor, cold_default_cache_69, translator_69c, similarity_scorer,
    frozen_segmentation=frozen_seg_69,  # require_warm_primary_cache omitted -- defaults to False
)
check("gate OFF (default): a cold cache is translated normally, not blocked", report_69c["translation_aborted_early"] is False)
check(
    "gate OFF (default): the sentence was actually sent to NLLB and got a real translation",
    sentences_out_69c["iv69::q1::s000"]["translation_provenance"] == "FRESH_OK",
)
check("gate OFF (default): coverage is still computed and reported even though it doesn't gate anything", report_69c["primary_cache_coverage"]["missing"] == 1)
shutil.rmtree(cold_default_cache_dir_69, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 70: Round 12 (external review fix) -- compute_primary_cache_coverage() counts by SENTENCE UNIT "
      "(duplicate source texts included), and never pollutes cache.stats() hit/miss counters ===")
cache_70, cache_dir_70 = fresh_cache()
translator_70 = StubNLLBTranslator()
shared_text_70 = "No, no."
unique_text_70 = "Aquest text nomes apareix una vegada."
# Cache only the shared text's primary entry -- unique_text_70 stays
# uncached, deliberately, so coverage is genuinely partial.
cache_70.set(shared_text_70, "ca", "es", "No, no.", model_name=translator_70.model_name, generation_version=preprocess_v2.GENERATION_VERSION)

pending_70 = [
    {
        "translation_required": True,
        "sentences": [
            {"sentence_id": "r1::s000", "sentence_index": 0, "text_source": shared_text_70},
            {"sentence_id": "r1::s001", "sentence_index": 1, "text_source": shared_text_70},  # duplicate source text
            {"sentence_id": "r1::s002", "sentence_index": 2, "text_source": unique_text_70},
        ],
    },
    {
        # translation_required False -- e.g. a Spanish-primary response --
        # must be excluded from the expected count entirely.
        "translation_required": False,
        "sentences": [{"sentence_id": "r2::s000", "sentence_index": 0, "text_source": "Ja esta en espanyol."}],
    },
]

hits_before_70 = cache_70.stats()["hits_this_run"]
misses_before_70 = cache_70.stats()["misses_this_run"]
coverage_70 = preprocess_v2.compute_primary_cache_coverage(pending_70, cache_70, translator_70)

check("expected counts every Catalan SENTENCE UNIT, not distinct source texts -- 3, not 2 or 4", coverage_70["expected_catalan_sentence_units"] == 3)
check("both sentence IDs sharing the cached shared text count as covered independently", coverage_70["covered"] == 2)
check(
    "the one genuinely-uncached sentence is the only one missing",
    coverage_70["missing"] == 1 and coverage_70["missing_sentence_ids"] == ["r1::s002"],
)
check("SAFE TO REUSE PRIMARY CACHE is correctly False -- coverage is partial", coverage_70["safe_to_reuse_primary_cache"] is False)
check(
    "compute_primary_cache_coverage() used contains(), not get() -- cache.stats() hit/miss counters are completely untouched by the coverage check",
    cache_70.stats()["hits_this_run"] == hits_before_70 and cache_70.stats()["misses_this_run"] == misses_before_70,
)
shutil.rmtree(cache_dir_70, ignore_errors=True)


# ===========================================================================
print("\n=== Test 71: Round 13 (external review fix) -- FROZEN source_language is authoritative over the LIVE "
      "per-response detector, never fatal, never discards frozen sentences ===")
# The bug: a response whose frozen segmentation recorded one source
# language, but whose text the LIVE per-response detector (re-run every
# time process() runs) happened to call something else, was treated as a
# FATAL "frozen_segmentation_source_language_mismatch" -- its frozen
# sentences were discarded entirely (sentence_records = []) and its
# translation_required flag was computed from the LIVE guess instead of
# the frozen one. In the real corpus this silently dropped 3 real frozen
# sentences (6396 -> 6393) and mis-routed 2 genuinely Spanish responses
# into "requires NLLB translation" with zero actual sentences to
# translate. The fix: once a frozen entry exists, ITS source_language
# wins outright; the live detector's disagreement is recorded as a
# REVIEW-only warning and nothing else changes.
StubNLLBTranslator.reset()
# A text independently confirmed (see Test 69's own comment) to be
# LIVE-detected as "ca" via the real determine_response_language() --
# used here specifically so the live/frozen disagreement is real, not
# contrived by mocking the detector itself.
frozen_text_71 = "Aquesta frase ja hauria d'estar traduida abans de tornar a executar el programa."
cache_71, cache_dir_71 = fresh_cache()
translator_71 = StubNLLBTranslator()
raw_71 = {"iv71": {"q1": frozen_text_71}}
frozen_seg_71 = {
    "iv71::q1": {
        "source_language": "es",  # FROZEN says Spanish -- deliberately disagrees with live "ca"
        "sentences": [
            {"sentence_id": "iv71::q1::s000", "sentence_index": 0, "text_source": "Aquesta frase ja hauria d'estar traduida"},
            {"sentence_id": "iv71::q1::s001", "sentence_index": 1, "text_source": "abans de tornar a executar el programa."},
        ],
    },
}
responses_out_71, sentences_out_71, report_71 = preprocess_v2.process(
    raw_71, preprocessor, cache_71, translator_71, similarity_scorer, frozen_segmentation=frozen_seg_71,
)
resp_71 = responses_out_71["iv71::q1"]

check("the response is NOT fatal -- status is OK, not excluded", resp_71["status"] == "OK")
check("FROZEN source_language (es) wins over the live detector's guess (ca)", resp_71["source_language"] == "es")
check("translation_required is derived from the FROZEN language, so it is correctly False", resp_71["translation_required"] is False)
check("BOTH frozen sentences are kept -- none discarded", resp_71["sentence_count"] == 2)
check("both frozen sentence IDs are present in sentences_out", "iv71::q1::s000" in sentences_out_71 and "iv71::q1::s001" in sentences_out_71)
check(
    "each sentence is provenance SOURCE_ES (treated as already-Spanish, never sent to NLLB), text unchanged",
    sentences_out_71["iv71::q1::s000"]["translation_provenance"] == "SOURCE_ES"
    and sentences_out_71["iv71::q1::s000"]["text_es"] == "Aquesta frase ja hauria d'estar traduida"
    and sentences_out_71["iv71::q1::s001"]["translation_provenance"] == "SOURCE_ES"
    and sentences_out_71["iv71::q1::s001"]["text_es"] == "abans de tornar a executar el programa.",
)
check("zero fatal errors -- the old 'frozen_segmentation_source_language_mismatch' reason no longer fires", report_71["fatal_error_count"] == 0)
check("structural_validity is True", report_71["structural_validity"] is True)
check(
    "sentence_count_invariant holds exactly -- no sentence silently dropped",
    report_71["sentence_count_invariant"]["match"] is True
    and report_71["sentence_count_invariant"]["expected"] == 2
    and report_71["sentence_count_invariant"]["actual"] == 2,
)
check(
    "the disagreement is recorded as a REVIEW-only warning, correctly naming both languages",
    any(
        w["id"] == "iv71::q1" and w["reason"] == "live_detector_disagrees_with_frozen_language"
        and w["status"] == "REVIEW" and w["live_detected_language"] == "ca" and w["frozen_language"] == "es"
        for w in report_71["warnings"]
    ),
)
check(
    "translation_quality_summary correctly counts and names exactly this one source-language mismatch",
    report_71["translation_quality_summary"]["source_language_mismatch_count"] == 1
    and report_71["translation_quality_summary"]["source_language_mismatch_ids"] == ["iv71::q1"],
)
check("translation_sanity_status is REVIEW, not FAIL or PASS", report_71["translation_sanity_status"] == "REVIEW")
check("STEP2_VALID is True -- a source-language disagreement alone is never fatal", report_71["STEP2_VALID"] is True)
check(
    "no NLLB call was made for this response -- it was correctly never sent for translation",
    all(not (call[0] == "ca" and call[1] == "es") for call in StubNLLBTranslator.CALL_LOG),
)
shutil.rmtree(cache_dir_71, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print(f"\n{'=' * 60}")
print(f"TOTAL: {PASS} passed, {FAIL} failed")
if FAIL:
    print("SOME CHECKS FAILED")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
