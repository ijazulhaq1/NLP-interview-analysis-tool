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
STUB_REPETITIVE_OUTPUT = "el perro el perro el perro el perro el perro el perro el perro corre"


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
    CALL_LOG = []  # list of (source_lang, target_lang, [texts]) per translate_batch call

    def __init__(self, model_name=preprocess_v2.NLLB_MODEL_NAME, requested_device=None, batch_size=preprocess_v2.DEFAULT_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.requested_device = requested_device or "cpu"
        self.device = self.requested_device
        self.device_fallback = False
        self.fallback_reason = None

    def translate_batch(self, texts, source_lang, target_lang):
        StubNLLBTranslator.CALL_LOG.append((source_lang, target_lang, list(texts)))

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
check("text_es equals original_text (copied, not translated)", response["text_es"] == response["original_text"])
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
# Kept under MAX_TRANSLATE_CHUNK_WORDS (150 words) so this remains a
# single translation chunk -- STUB_ES_OUTPUT repeated across multiple
# chunks would itself look like degenerate repetition, which is not what
# this test is checking.
long_low_ratio_text = " ".join(
    f"Aquesta es una resposta llarga en catala escrita expressament per a la prova numero {i} de la ratio de longitud."
    for i in range(4)
)
check(
    "the long source stays within a single translation chunk for this test's premise",
    len(long_low_ratio_text.split()) <= preprocess_v2.MAX_TRANSLATE_CHUNK_WORDS,
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
    "iv1": {"q1": "Aquesta és una resposta en català prou llarga per a la prova d'integració amb terminal.py."},
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

check("a long response required more than one translation chunk", response["translation_chunk_count"] > 1)
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
# it as a language mismatch (it genuinely isn't Spanish), which per fix
# #2 makes translation_output_validity (and therefore STEP2_VALID) False.
# That is fix #2 working as intended, not a bug in the chunking fix this
# test targets -- STEP2_VALID's dependence on genuinely-Spanish output is
# covered on its own terms by Tests 20, 33, and 34.
check(
    "the echo stub's non-Spanish output is correctly flagged as a language mismatch (fix #2, not a truncation bug)",
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
print("\n=== Test 33: a language-mismatch-only failure now prevents STEP2_VALID=True ===")
# The translation itself "succeeds" (non-empty, not a retry failure) but
# lands in the wrong language entirely -- STUB_EN_OUTPUT is a genuinely
# fluent, varied English sentence, so it is NOT flagged as degenerate
# (empty/identical/repetitive); the only thing wrong with it is that it
# isn't Spanish. Per the reviewer's fix #2, this is a genuinely serious
# failure and must block STEP2_VALID on its own.
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
check("a language mismatch WAS flagged", tqs_mismatch["language_mismatch_count"] > 0)
check("no degenerate output was flagged for this case (fluent, varied, non-empty text)", tqs_mismatch["degenerate_output_count"] == 0)
check("translation_output_validity is False (language mismatch alone is fatal)", report["translation_output_validity"] is False)
check("translation_sanity_status is FAIL, not just REVIEW, for a genuinely serious failure", report["translation_sanity_status"] == "FAIL")
check("STEP2_VALID is False even though structural_validity and completeness are both True", report["STEP2_VALID"] is False)
check("structural_validity stays True (the response itself was not lost)", report["structural_validity"] is True)
shutil.rmtree(cache_dir, ignore_errors=True)
StubNLLBTranslator.reset()


# ===========================================================================
print("\n=== Test 34: a degenerate-output-only failure now prevents STEP2_VALID=True ===")
# STUB_REPETITIVE_OUTPUT is confirmed (Test 11, and via real langdetect
# above) to be detected as genuinely Spanish -- so this case isolates
# degeneracy (n-gram repetition) from language mismatch, the mirror image
# of Test 33.
StubNLLBTranslator.FAIL_MODE = "repetition_one"
degenerate_text = "Aquest text es tradueix de manera repetitiva per accident en aquesta prova"
StubNLLBTranslator.REPETITIVE_FOR_TEXTS = frozenset([degenerate_text])
cache, cache_dir = fresh_cache()
translator = StubNLLBTranslator()
raw_data = {"iv_degenerate": {"q1": degenerate_text}}
responses_out, sentences_out, report = preprocess_v2.process(raw_data, preprocessor, cache, translator, similarity_scorer)
tqs_degenerate = report["translation_quality_summary"]
check("translation completed (non-empty), nothing failed at the completeness level", report["translation_failed_count"] == 0)
check("translation_completeness_validity True (the translation itself did not fail)", report["translation_completeness_validity"] is True)
check("no language mismatch was flagged for this case (output is genuinely Spanish)", tqs_degenerate["language_mismatch_count"] == 0)
check("a degenerate (repetitive) output WAS flagged", tqs_degenerate["degenerate_output_count"] > 0)
check("translation_output_validity is False (degenerate output alone is fatal)", report["translation_output_validity"] is False)
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
long_response = " ".join(f"Aquesta és la frase numero {i} d'una resposta llarga." for i in range(40))
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
    "batch_size=8 and batch_size=2 produce the same chunk counts (chunking is batch-size-independent)",
    all(responses_b8[rid]["translation_chunk_count"] == responses_b2[rid]["translation_chunk_count"] for rid in responses_b8),
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
print(f"\n{'=' * 60}")
print(f"TOTAL: {PASS} passed, {FAIL} failed")
if FAIL:
    print("SOME CHECKS FAILED")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
