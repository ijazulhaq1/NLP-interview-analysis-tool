"""
preprocess_v2.py — Step 2 (Preprocessing/translation), frozen NLLB redesign (rev. 4)

Standalone, non-destructive script. It reads the same raw source the
original pipeline reads (data/input/interviews.json) and writes ONLY new
files:

    data/output/preprocessed_responses_v2.json
    data/output/preprocessed_sentences_v2.json
    data/output/preprocessing_language_report.json

It does NOT touch fully_processed_ca.json, sentiment_input_ca.json,
topic_modeling_input.json, or sentiment_results_ca.json directly — those
are written by terminal.py's Step 2 integration, which calls process()
below and bridges its validated output into those existing filenames only
when STEP2_VALID is True. terminal.py's own Stage 3/5 logic (BERTopic,
the sentiment model, the confusion/NLI model) is unchanged by this file.

===========================================================================
FROZEN METHODOLOGY (this revision)
===========================================================================

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

Compared to the previous (GoogleTranslate-backed, English-included)
revision, this is a deliberate simplification, not an oversight:

  - English disappears from the primary pipeline entirely. There is no
    Spanish->English translation and no BART-English confusion branch in
    Step 2. (A later, separately-tracked Stage 5 change is expected to
    replace facebook/bart-large-mnli with a multilingual NLI model that
    takes Spanish directly -- that is a sentiment_analysis.py concern, not
    a preprocess_v2.py concern, and is out of scope here.)
  - Only ONE translation direction is required: cat_Latn -> spa_Latn.
    Spanish-original responses are copied into text_es unchanged
    (translation_required=False) rather than "translated" Spanish->Spanish.
  - Translation happens ONCE, at the RESPONSE level, before sentence
    splitting -- not once per sentence, and not once per downstream
    branch. Sentence segmentation then runs on the Spanish-standardized
    response text, so the topic-modelling branch and the sentiment/NLI
    branch consume sentences drawn from the exact same translated text
    and the exact same sentence boundaries. Under the previous per-
    sentence-routed design, two branches could in principle see text
    translated via two independent live calls; that source of divergence
    is now structurally impossible, not just unlikely.
  - Per-sentence language detection/routing (the old
    determine_sentence_language() and the code-switch diagnostic built on
    top of it) is removed along with it: that machinery existed
    specifically to decide, per sentence, "which language do I tell the
    translator this came from" when several sentence-level translation
    calls were still in play. With exactly one response-level cat_Latn->
    spa_Latn call per response, there is nothing left for it to route.
  - The translation engine is a local model (facebook/nllb-200-distilled-
    600M via transformers), not a scraped web endpoint. This removes the
    entire network-reliability problem the previous revision's retry/
    backoff/pacing/cooldown/consecutive-failure-tracker machinery existed
    to survive (see git history / STEP2_RELIABILITY_CHANGES.md for that
    now-retired design) -- a local model either loads and runs, or it
    doesn't; it cannot be intermittently throttled by a remote service.
    A local model raises a different concern instead: it can produce
    fluent-looking but wrong output without ever raising an exception.
    That is what the translation_quality_summary in this file's report
    exists to catch (see below) -- completeness (did every required
    translation happen) and quality (was it any good) are tracked and
    reported as two separate, both-visible signals, not conflated into
    one opaque bit.

===========================================================================
ON DETERMINISM -- an intentionally careful claim
===========================================================================

Decoding is configured with do_sample=False and a fixed beam count, which
makes translation output far more reproducible than a live, unversioned
web service. This is NOT the same claim as "byte-identical output on any
hardware" -- PyTorch/device/library-version differences can still affect
floating-point execution in ways that could, in principle, change output.
The accurate claim, and the one used in this project's documentation, is:

    A fixed local NLLB checkpoint and a deterministic decoding
    configuration were used to improve computational reproducibility and
    remove dependence on a changing external translation service.

To make that claim checkable rather than asserted, this script records
model name, transformers/torch/Python versions, the resolved device, and
the exact generation parameters into every report (see
_software_versions() and the "model_info" report block).

===========================================================================
WHAT STAYS FROM THE PREVIOUS REVISION
===========================================================================

  - Stable, non-translation-derived IDs: response_id = f"{interview_id}::
    {question_id}"; sentence_id = f"{response_id}::s{index:03d}".
  - The response-level language-detection hierarchy (interview_primary_
    language derived from that interview's own sufficiently-long
    responses, restricted to {ca, es}; a confident English detection is
    flagged, never silently treated as a legitimate interview language).
  - No-silent-failure discipline: every skipped/failed item is recorded
    with an ID, a stage, a status (WARNING vs FATAL), and a reason.
  - A translation failure never leaves wrong-language text in a field
    named for a different language: on failure, text_es/topic_text_es_raw
    are set to None, status says WARNING_TRANSLATION_FAILED, and
    downstream consumers must check is_translation_usable(status) first.
  - Preprocessor.split_sentences_strict() / full_preprocess_v2() from
    preprocessing.py, unchanged -- both already take a `lang` parameter,
    so calling them with "es" instead of "ca" required no changes to that
    file at all.
  - original_text is NEVER overwritten, for both Catalan and Spanish
    responses -- required for the reviewer-requested original-vs-
    translated sensitivity analysis later.
"""
import ml_backend  # noqa: F401 (import FIRST -- sets USE_TF=0/etc. before torch/transformers below; see ml_backend.py)

import os
import sys
import time
import logging
import argparse
import platform
import subprocess
import statistics
import functools
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect_langs, LangDetectException
from tqdm import tqdm

from json_handler import JsonHandler
from preprocessing import Preprocessor  # noqa: F401 (imported first -> sets DetectorFactory.seed before any detection)
from translation_cache import TranslationCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config / thresholds
# ---------------------------------------------------------------------------

INPUT_FILE = "data/input/interviews.json"
OUTPUT_DIR = "data/output"
SMOKE_TEST_DIR = "data/output/smoke_test"
# A NEW cache, per the frozen redesign -- never the old Google-Translate-
# backed data/output/translation_cache_v2.json. Living under data/cache/
# (not data/output/) also makes it obvious at a glance that this file is
# not a pipeline *output* to be reviewed, but reusable translation memory.
CACHE_PATH = "data/cache/nllb_translation_cache_v1.json"

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANG_TO_NLLB = {"ca": "cat_Latn", "es": "spa_Latn"}
REQUIRED_TRANSLATION_DIRECTION = "ca->es"  # the only direction this corpus requires

# Deterministic decoding: no sampling, fixed beam count. Bump
# GENERATION_VERSION any time these change -- it is baked into the cache
# key (see translation_cache.py) specifically so a settings change can
# never silently serve a translation generated under the old settings.
GENERATION_PARAMS = {
    "num_beams": 4,
    "max_length": 512,
    "do_sample": False,
}
# v2: bumped deliberately when response-level boundary-aware chunking was
# introduced (see _split_into_translation_chunks / translate_responses_
# batched below). Translation now happens per-CHUNK, not necessarily per
# whole response, for any response that would otherwise risk silent
# truncation at NLLB's max_length=512-subword-token generation limit. This
# version bump forces a clean cache break rather than relying on
# incidental text differences to invalidate anything: no cache entry
# written under v1 -- some of which could have come from a since-fixed,
# silently truncated whole-response translation -- can ever be served
# under v2.
GENERATION_VERSION = "nllb600M-beams4-maxlen512-chunked-v2"

DEFAULT_BATCH_SIZE = 8
# Fixed after the fifth external review: on an Apple Silicon Mac, MPS is not
# a discrete GPU with its own VRAM -- it shares unified memory with the rest
# of the OS. A batch size tuned for a discrete-GPU/CPU setup (8) can push an
# 8GB Mac's Python process well past 10GB of resident memory once the model,
# activations, and a large padded batch are all held at once, driving the
# whole machine into memory compression/swapping -- which is what actually
# made a real multi-hour run on such a machine appear "stuck" (see
# STEP2_NLLB_CHANGES.md's fifth-review section; the process itself was not
# hung, it was thrashing). This does not change the model, decoding
# parameters, or any translation output -- a smaller batch produces
# identical translations to a larger one, just more (smaller) model calls.
LOW_MEMORY_DEVICES = {"mps"}
LOW_MEMORY_DEVICE_BATCH_SIZE = 2

# How often (in fully-completed responses) translate_responses_batched logs
# a persistent "[NLLB checkpoint]" progress line, in addition to the live
# tqdm bar -- so progress is visible even when stdout isn't a TTY (e.g.
# piped to a log file) and the tqdm bar's own carriage-return updates don't
# show.
PROGRESS_CHECKPOINT_INTERVAL = 25

# A local generation call is retried once on a transient error (e.g. a
# one-off device hiccup) before being treated as a real failure -- this is
# deliberately much lighter than the old network retry/backoff, because a
# local model failing is not expected to be a rate-limiting problem that
# backoff would help with.
LOCAL_MAX_RETRIES = 1
LOCAL_RETRY_DELAY_SECONDS = 2.0
# If this many consecutive BATCHES fail outright (after their own retry),
# stop the run rather than grinding through the rest of the corpus against
# whatever local problem (e.g. persistent OOM) is causing it.
MAX_CONSECUTIVE_BATCH_FAILURES = 3

# These interviews were conducted in Catalan or Spanish. A response/
# interview detected as English at high confidence is NOT treated as a
# legitimate primary interview language -- flagged instead.
EXPECTED_INTERVIEW_LANGUAGES = {"ca", "es"}

# A response is "sufficiently long" for direct language detection if it
# clears both bars (see the original Step 2 design notes for the
# empirical basis of these thresholds against this corpus).
MIN_CHARS_FOR_DIRECT_DETECTION = 30
MIN_WORDS_FOR_DIRECT_DETECTION = 5

LANGUAGE_TIE_BREAK_ORDER = ["ca", "es"]

# Translation-sanity diagnostics (informational; see STEP2_VALID logic
# below for why these never gate validity on their own).
LENGTH_RATIO_LOW = 0.4
LENGTH_RATIO_HIGH = 3.0
REPETITION_NGRAM_SIZE = 3
REPETITION_MIN_REPEATS = 4  # same 3-gram appearing >=4x flags repetition
SIMILARITY_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Reported alongside the similarity distribution as a count, NEVER used as
# a pass/fail gate -- see the module docstring and STEP2_VALID logic.
SIMILARITY_INFORMATIONAL_LOW_BOUND = 0.50
DEFAULT_ROUNDTRIP_SAMPLE_SIZE = 100

USABLE_TRANSLATION_STATUSES = {"IDENTITY_COPY", "CACHE_HIT", "TRANSLATED"}


def is_translation_usable(status: str) -> bool:
    """Downstream stages (Stage 3 topic modelling, Stage 3/5 sentiment and
    NLI) must call this (or check membership in USABLE_TRANSLATION_STATUSES
    directly) before using text_es / topic_text_es_raw / topic_text_es_clean.
    A WARNING_TRANSLATION_FAILED status means the corresponding field is
    None -- there is nothing usable to read.
    """
    return status in USABLE_TRANSLATION_STATUSES


# ---------------------------------------------------------------------------
# Software / environment provenance
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _software_versions(preprocessor: Optional[object] = None) -> Dict[str, str]:
    """Records the reproducibility-relevant software environment. Every
    lookup here is best-effort -- a missing/unloadable package must never
    crash the run over a provenance field.
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "langdetect": _pkg_version("langdetect"),
        "spacy": _pkg_version("spacy"),
    }
    for attr, key in [("nlp_ca", "ca_core_news_sm"), ("nlp_es", "es_core_news_sm")]:
        model = getattr(preprocessor, attr, None) if preprocessor is not None else None
        meta = getattr(model, "meta", None) if model is not None else None
        versions[key] = meta.get("version", "unknown") if isinstance(meta, dict) else "not_loaded"
    return versions


def _detect_rosetta_translation() -> Optional[bool]:
    """Best-effort detection of x86_64 Python running under Rosetta 2
    translation on an Apple Silicon (arm64) Mac.

    Returns True if translated (Rosetta), False if this is genuinely x86_64
    hardware (an Intel Mac), or None if undeterminable (not macOS, not
    x86_64, or the `sysctl` probe itself failed/isn't present -- e.g. this
    sandbox, or any non-Mac machine). Diagnostic-only: never raises, and
    the caller must never let the result affect STEP2_VALID -- see
    get_runtime_architecture_info() and the fifth external review section
    in STEP2_NLLB_CHANGES.md.

    `sysctl.proc_translated` is the standard macOS mechanism for exactly
    this check (returns "1" under Rosetta, "0" natively, and doesn't exist
    on Intel Macs at all -- both of the latter cases correctly fall through
    to `return None` inside the platform.machine() == "x86_64" branch below
    only when the sysctl name itself is unrecognized, which `sysctl -in`
    reports with empty output rather than an error).
    """
    if sys.platform != "darwin" or platform.machine() != "x86_64":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-in", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=2,
        )
        output = result.stdout.strip()
        if output == "1":
            return True
        if output == "0":
            return False
        return None
    except Exception:
        return None


def get_runtime_architecture_info() -> dict:
    """Reproducibility/diagnostic metadata for the Step 2 report and the
    full-run startup banner: what CPU architecture this Python process is
    actually running as, and whether it's an x86_64 process being
    translated by Rosetta 2 on Apple Silicon hardware.

    Added after the fifth external review: a real multi-hour run on an
    ~8GB Apple Silicon Mac was diagnosed as running an x86_64 Python build
    under Rosetta, with the process's memory footprint reaching ~10.7GB --
    on that little physical RAM, that drove the machine into heavy memory
    compression/swapping, which is what actually made the run appear
    "stuck" for hours (the process was thrashing, not hung). Rosetta
    translation does not change any translation OUTPUT (NLLB still runs
    the same model, same weights, same decoding parameters) -- it is
    purely a performance/memory concern, so this is surfaced as a WARNING
    only and must never affect STEP2_VALID.
    """
    machine = platform.machine()
    running_under_rosetta = _detect_rosetta_translation()
    warning = None
    if running_under_rosetta:
        warning = (
            f"Running as {machine} Python translated by Rosetta 2 on Apple Silicon. "
            "This significantly increases memory usage and slows down NLLB inference, "
            "and can drive a low-memory Mac into swapping on a full corpus run. "
            "Installing a native arm64 Python (e.g. an arm64-native conda/miniforge "
            "environment, or python.org's universal2 installer) is strongly "
            "recommended for full-corpus runs. This is a performance warning only -- "
            "it does not affect translation output or STEP2_VALID."
        )
    return {
        "runtime_architecture": machine,
        "running_under_rosetta": running_under_rosetta,
        "warning": warning,
    }


def _format_hms(seconds: float) -> str:
    """Formats a duration as HH:MM:SS (hours can exceed 99 on a very long
    run -- kept as a plain zero-padded integer rather than wrapping).
    """
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Local NLLB translation backend
# ---------------------------------------------------------------------------

class NLLBTranslationError(RuntimeError):
    """Raised when a batch fails even after its local retry -- distinct
    from a single unusual output, which is a quality diagnostic, not an
    error.
    """


class NLLBTranslator:
    """Loads facebook/nllb-200-distilled-600M once and reuses it for every
    call in the run. No network calls after the model is first
    downloaded/cached by `transformers`' own hub cache; decoding is
    deterministic (do_sample=False, fixed beam count).

    Device handling: resolves cuda > mps > cpu once at construction
    (unless a device is explicitly requested), and if a generation call on
    a non-CPU device fails, falls back to CPU for the REST of the run
    (recorded once, not re-attempted per call) rather than bouncing
    between devices sentence by sentence.
    """

    def __init__(
        self,
        model_name: str = NLLB_MODEL_NAME,
        requested_device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.requested_device = requested_device or self._auto_detect_device()
        self.device = self.requested_device
        self.device_fallback = False
        self.fallback_reason: Optional[str] = None

        logger.info(f"Loading NLLB tokenizer/model ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"NLLB model loaded on device={self.device}")

    @staticmethod
    def _auto_detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _forced_bos_token_id(self, nllb_target_code: str) -> int:
        """Compatibility shim: `NllbTokenizer.lang_code_to_id` was the
        documented way to get a language token's id in the transformers
        versions this project pins (~4.28-4.35). Newer transformers
        releases may not expose that attribute the same way, so this falls
        back to convert_tokens_to_ids(), which is stable across versions.
        """
        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and nllb_target_code in lang_code_to_id:
            return lang_code_to_id[nllb_target_code]
        return self.tokenizer.convert_tokens_to_ids(nllb_target_code)

    def _fallback_to_cpu(self, reason: str) -> None:
        logger.error(
            f"[NLLBTranslator] generation failed on device={self.device}: {reason}. "
            "Falling back to CPU for the remainder of this run."
        )
        self.model.to("cpu")
        self.device = "cpu"
        self.device_fallback = True
        self.fallback_reason = reason

    def count_tokens(self, text: str, source_lang: str) -> int:
        """Actual NLLB subword token count for `text`, using the real
        tokenizer -- including the special tokens the tokenizer adds, so
        this is exactly what would be handed to `generate()` under the
        same call `translate_batch()` makes. This is the ground truth
        `_split_into_translation_chunks()`'s word-count heuristic is
        checked against (see `translate_responses_batched`), and what
        `_assert_within_token_limit()` below uses as a final, authoritative
        safety net before every real generation call.
        """
        self.tokenizer.src_lang = LANG_TO_NLLB[source_lang]
        return len(self.tokenizer(text)["input_ids"])

    def _assert_within_token_limit(self, texts: List[str], source_lang: str) -> None:
        """Hard, fail-closed check: refuses to translate anything that
        would exceed GENERATION_PARAMS['max_length'] NLLB subword tokens,
        rather than letting the tokenizer's own `truncation=True` silently
        discard part of the input.

        This is deliberately NOT inside the retry loop below -- an
        oversized chunk is a deterministic property of that text, not a
        transient failure, so retrying it would just waste time before
        failing the same way again. In normal operation this should never
        actually fire: `_split_into_translation_chunks()` already keeps
        chunks under this limit using this translator's own
        `count_tokens()` (see `translate_responses_batched`). If it does
        fire, that means a chunk reached this point without having been
        measured against the real tokenizer -- a chunking-safety-margin
        bug, not an expected runtime condition -- so this fails loudly
        instead of silently truncating.
        """
        limit = GENERATION_PARAMS["max_length"]
        oversized = [(t, self.count_tokens(t, source_lang)) for t in texts]
        oversized = [(t, n) for t, n in oversized if n > limit]
        if oversized:
            largest = max(n for _, n in oversized)
            raise NLLBTranslationError(
                f"{len(oversized)} chunk(s) in this batch tokenize to more than "
                f"{limit} NLLB subword tokens (largest: {largest}) even after "
                "sentence-boundary chunking -- refusing to translate with "
                "truncation=True silently discarding part of the input."
            )

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translates a batch of texts, all in the same direction. Retries
        once locally on failure; on a non-CPU device, a failure triggers a
        one-time, sticky fallback to CPU (recorded on the instance) before
        the retry. Raises NLLBTranslationError if the batch still fails
        after that, or immediately (no retry) if any input would exceed
        NLLB's token limit -- see _assert_within_token_limit().
        """
        if not texts:
            return []

        src_code = LANG_TO_NLLB[source_lang]
        tgt_code = LANG_TO_NLLB[target_lang]

        self._assert_within_token_limit(texts, source_lang)

        last_exc: Optional[Exception] = None
        for attempt in range(LOCAL_MAX_RETRIES + 1):
            try:
                self.tokenizer.src_lang = src_code
                inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self._forced_bos_token_id(tgt_code),
                        **GENERATION_PARAMS,
                    )
                return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            except Exception as e:
                last_exc = e
                if self.device != "cpu":
                    # A device-related failure gets exactly one sticky
                    # fallback, not a fallback-per-attempt.
                    self._fallback_to_cpu(reason=str(e))
                    continue
                if attempt < LOCAL_MAX_RETRIES:
                    logger.warning(
                        f"[NLLBTranslator] batch translate attempt {attempt + 1}/"
                        f"{LOCAL_MAX_RETRIES + 1} failed: {e}. Retrying in "
                        f"{LOCAL_RETRY_DELAY_SECONDS:.0f}s."
                    )
                    time.sleep(LOCAL_RETRY_DELAY_SECONDS)

        raise NLLBTranslationError(f"batch translation failed after retries: {last_exc}")

    def model_info(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "actual_device": self.device,
            "device_fallback": self.device_fallback,
            "fallback_reason": self.fallback_reason,
            "generation_params": dict(GENERATION_PARAMS),
            "generation_version": GENERATION_VERSION,
            "batch_size": self.batch_size,
        }


def resolve_default_batch_size(device: Optional[str] = None) -> int:
    """Returns the batch size to use when the caller/CLI did not explicitly
    request one, based on the resolved device.

    Added after the fifth external review: MPS (Apple Silicon's GPU
    backend) shares unified memory with the OS rather than having its own
    VRAM, so DEFAULT_BATCH_SIZE (tuned assuming a discrete GPU or a CPU
    run) can push memory usage high enough to cause swapping on a
    low-memory Mac. `device` defaults to whatever
    NLLBTranslator._auto_detect_device() would resolve -- that call is
    cheap (no model load), so this can safely run before a translator is
    constructed, to pick its batch_size up front. An explicit
    `--batch-size` (or an explicit `batch_size=` argument anywhere in this
    module) always overrides this -- this function is only ever consulted
    when nothing was explicitly requested.
    """
    resolved_device = device if device is not None else NLLBTranslator._auto_detect_device()
    return LOW_MEMORY_DEVICE_BATCH_SIZE if resolved_device in LOW_MEMORY_DEVICES else DEFAULT_BATCH_SIZE


class ConsecutiveBatchFailureTracker:
    """Counts consecutive fully-failed translation batches (i.e. every
    retry inside NLLBTranslator.translate_batch also failed). This is the
    local-model equivalent of the old network consecutive-failure tracker,
    at a much lighter threshold, since a local failure that repeats is
    almost certainly systemic (e.g. persistent OOM), not transient.
    """

    def __init__(self, max_consecutive: int = MAX_CONSECUTIVE_BATCH_FAILURES):
        self.max_consecutive = max_consecutive
        self.consecutive_failures = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive:
            raise NLLBTranslationError(
                f"{self.consecutive_failures} consecutive translation batches failed "
                "-- aborting the run early rather than continuing against what looks "
                "like a systemic local problem (e.g. out-of-memory)."
            )


# ---------------------------------------------------------------------------
# Response-level language detection (unchanged from the previous revision)
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _is_sufficiently_long(text: str) -> bool:
    text = text.strip()
    return len(text) >= MIN_CHARS_FOR_DIRECT_DETECTION and _word_count(text) >= MIN_WORDS_FOR_DIRECT_DETECTION


def detect_raw_language(text: str) -> Tuple[Optional[str], str, float]:
    """Returns (lang_or_None, detail, confidence).
    detail in {"direct_detected_expected", "direct_detected_unexpected_english",
               "direct_detection_other_language", "direct_detection_failed"}
    """
    try:
        candidates = detect_langs(text[:500])
        if not candidates:
            return None, "direct_detection_failed", 0.0
        best = candidates[0]
        if best.lang in EXPECTED_INTERVIEW_LANGUAGES:
            return best.lang, "direct_detected_expected", float(best.prob)
        if best.lang == "en":
            return None, "direct_detected_unexpected_english", float(best.prob)
        return None, "direct_detection_other_language", float(best.prob)
    except LangDetectException:
        return None, "direct_detection_failed", 0.0


def compute_interview_primary_language(responses: Dict[str, str]) -> Tuple[str, dict]:
    votes = {"ca": 0, "es": 0}
    considered = 0
    unexpected_english = []
    for question_id, text in responses.items():
        if not _is_sufficiently_long(text):
            continue
        lang, detail, conf = detect_raw_language(text)
        if detail == "direct_detected_expected":
            considered += 1
            votes[lang] += 1
        elif detail == "direct_detected_unexpected_english":
            unexpected_english.append({"question_id": question_id, "confidence": conf})

    if considered == 0 or sum(votes.values()) == 0:
        return "ca", {
            "votes": votes,
            "considered_responses": considered,
            "unexpected_english_responses": unexpected_english,
            "fallback_used": True,
            "fallback_reason": "no_long_response_yielded_expected_language",
        }

    max_votes = max(votes.values())
    winners = [lang for lang in LANGUAGE_TIE_BREAK_ORDER if votes[lang] == max_votes]
    primary = winners[0]
    return primary, {
        "votes": votes,
        "considered_responses": considered,
        "unexpected_english_responses": unexpected_english,
        "fallback_used": False,
        "tie_broken": len([l for l in votes if votes[l] == max_votes]) > 1,
    }


def determine_response_language(text: str, interview_primary_language: str) -> Tuple[str, str, Optional[float]]:
    if not _is_sufficiently_long(text):
        return interview_primary_language, "interview_fallback_short", None

    lang, detail, conf = detect_raw_language(text)
    if detail == "direct_detected_expected":
        return lang, "direct", None
    if detail == "direct_detected_unexpected_english":
        return interview_primary_language, "interview_fallback_unexpected_english", conf
    if detail == "direct_detection_other_language":
        return interview_primary_language, "interview_fallback_unsupported_language", None
    return interview_primary_language, "interview_fallback_detection_failed", None


# ---------------------------------------------------------------------------
# Translation quality diagnostics (all informational -- see STEP2_VALID)
# ---------------------------------------------------------------------------

def check_target_language(translated_text: str, expected_lang: str = "es") -> dict:
    """langdetect on the OUTPUT of translation. Catches the most
    catastrophic local-model failure mode: echoing the source language
    back, or drifting into a third language.

    Very short output is NOT assessed (status="NOT_ASSESSED_SHORT_TEXT",
    passed=None) rather than run through langdetect at all -- reusing
    _is_sufficiently_long / MIN_CHARS_FOR_DIRECT_DETECTION /
    MIN_WORDS_FOR_DIRECT_DETECTION, the same threshold this module already
    uses for response-level source-language detection. This corpus's real
    responses include 46 of 3 words or fewer ("Sí.", "No.", "PC.", "2009",
    "-No, no."); langdetect on a single word or short exclamation is
    unreliable-to-meaningless, and a correct NLLB translation of one of
    these could otherwise be marked a language mismatch purely because the
    detector guessed wrong on too little evidence -- caught in the second
    external review. passed is None (never False) when not assessed, so a
    caller that naively does `if not passed` on this dict without checking
    status would need to notice -- see how language_mismatch_ids below is
    computed (`passed is False`, not just falsy) for the safe pattern.
    """
    if not _is_sufficiently_long(translated_text):
        return {
            "detected_language": None,
            "confidence": None,
            "passed": None,
            "status": "NOT_ASSESSED_SHORT_TEXT",
        }
    try:
        candidates = detect_langs(translated_text[:500])
        if not candidates:
            return {"detected_language": None, "confidence": None, "passed": False, "status": "ASSESSED"}
        best = candidates[0]
        return {
            "detected_language": best.lang,
            "confidence": float(best.prob),
            "passed": best.lang == expected_lang,
            "status": "ASSESSED",
        }
    except LangDetectException:
        return {"detected_language": None, "confidence": None, "passed": False, "status": "ASSESSED"}


def check_degenerate_output(source_text: str, translated_text: str) -> dict:
    """Flags: empty output, output identical to the (different-language)
    source (a translation that silently no-op'd), and n-gram repetition
    loops (a known small/mid-size NMT failure mode).

    is_identical_to_source is only treated as FATAL
    (is_identical_to_source_fatal) when the text is long enough (see
    _is_sufficiently_long) to make identity meaningfully suspicious. Short
    Catalan/Spanish text -- a single word, acronym, number, or proper noun
    ("Sí." -> "Sí.", "PC." -> "PC.", "2009" -> "2009") is frequently and
    CORRECTLY identical across both languages; this corpus's real
    responses include 46 of 3 words or fewer, so treating every short
    identical translation as degenerate would invalidate correct
    translations -- caught in the second external review. is_empty and
    repetition_flag remain fatal at every length: an empty translation or
    an n-gram repetition loop is never a legitimate short-text outcome the
    way source==target can be.
    """
    is_empty = not translated_text or not translated_text.strip()
    is_identical_to_source = (
        not is_empty and translated_text.strip() == source_text.strip()
    )
    is_identical_to_source_fatal = is_identical_to_source and _is_sufficiently_long(translated_text)

    repetition_flag = False
    repetition_detail = None
    if not is_empty:
        tokens = translated_text.split()
        n = REPETITION_NGRAM_SIZE
        if len(tokens) >= n * REPETITION_MIN_REPEATS:
            ngram_counts: Dict[Tuple[str, ...], int] = {}
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            worst_ngram, worst_count = max(ngram_counts.items(), key=lambda kv: kv[1])
            if worst_count >= REPETITION_MIN_REPEATS:
                repetition_flag = True
                repetition_detail = {"ngram": " ".join(worst_ngram), "count": worst_count}

    return {
        "is_empty": is_empty,
        "is_identical_to_source": is_identical_to_source,
        "is_identical_to_source_fatal": is_identical_to_source_fatal,
        "repetition_flag": repetition_flag,
        "repetition_detail": repetition_detail,
        "flagged": is_empty or is_identical_to_source_fatal or repetition_flag,
    }


def compute_length_diagnostics(source_text: str, translated_text: str) -> dict:
    source_chars = len(source_text)
    target_chars = len(translated_text) if translated_text else 0
    source_tokens = _word_count(source_text)
    target_tokens = _word_count(translated_text) if translated_text else 0

    char_ratio = (target_chars / source_chars) if source_chars else None
    token_ratio = (target_tokens / source_tokens) if source_tokens else None

    outlier = False
    if char_ratio is not None:
        outlier = char_ratio < LENGTH_RATIO_LOW or char_ratio > LENGTH_RATIO_HIGH

    return {
        "source_chars": source_chars,
        "target_chars": target_chars,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "char_ratio": char_ratio,
        "token_ratio": token_ratio,
        "length_ratio_outlier": outlier,
    }


def _distribution_summary(values: List[float]) -> dict:
    """mean / median / stdev / p05 / p25 / min, plus a count below the
    informational (non-gating) low bound. Returns a mostly-None shape when
    there is nothing to summarize, rather than raising, so a corpus with
    zero sampled values still produces a well-formed report.
    """
    if not values:
        return {
            "count": 0, "mean": None, "median": None, "stdev": None,
            "p05": None, "p25": None, "min": None,
            "count_below_informational_bound": 0,
            "informational_bound": SIMILARITY_INFORMATIONAL_LOW_BOUND,
        }
    sorted_values = sorted(values)

    def _percentile(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = p * (len(sorted_values) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
        frac = idx - lo
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p05": _percentile(0.05),
        "p25": _percentile(0.25),
        "min": sorted_values[0],
        "count_below_informational_bound": sum(1 for v in values if v < SIMILARITY_INFORMATIONAL_LOW_BOUND),
        "informational_bound": SIMILARITY_INFORMATIONAL_LOW_BOUND,
    }


class SemanticSimilarityScorer:
    """Thin wrapper around a multilingual sentence-embedding model, used
    for two reference-free translation-quality proxies:
      - semantic preservation: original Catalan vs. its Spanish translation
      - round-trip: original Catalan vs. Catalan-Spanish-Catalan round-trip

    Loaded lazily and only if actually needed (process() skips this
    entirely for a corpus with no Catalan-original responses), and degrades
    gracefully (reports why) if sentence-transformers isn't installed
    rather than crashing Step 2 over an optional quality signal.
    """

    def __init__(self, model_name: str = SIMILARITY_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.unavailable_reason: Optional[str] = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:  # ImportError or a download/load failure
            self.unavailable_reason = str(e)
            logger.warning(
                f"[SemanticSimilarityScorer] could not load '{model_name}': {e}. "
                "Semantic-preservation and round-trip similarity will be omitted "
                "from this run's report."
            )

    @property
    def available(self) -> bool:
        return self.model is not None

    def batch_cosine_similarity(self, texts_a: List[str], texts_b: List[str]) -> List[float]:
        if not self.available or not texts_a:
            return []
        import numpy as np
        emb_a = self.model.encode(texts_a, convert_to_numpy=True, normalize_embeddings=True)
        emb_b = self.model.encode(texts_b, convert_to_numpy=True, normalize_embeddings=True)
        return [float(np.dot(a, b)) for a, b in zip(emb_a, emb_b)]


# ---------------------------------------------------------------------------
# Cache-aware batched translation
# ---------------------------------------------------------------------------

def translate_texts_batched(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    translator: NLLBTranslator,
    cache: TranslationCache,
    batch_failure_tracker: ConsecutiveBatchFailureTracker,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> List[Tuple[Optional[str], str]]:
    """Translates a list of texts (same direction), serving cache hits
    without a model call and batching the remaining cache misses through
    translator.translate_batch(). Returns a list of (translated_text,
    status) pairs in the same order as `texts`. A translation that fails
    is never cached -- only a real, non-empty model output is written to
    the cache.

    `progress_callback(index, status)`, if given, is invoked once per
    element of `texts` as soon as its outcome is known (`index` is its
    position in `texts`; `status` is "CACHE_HIT", "TRANSLATED", or
    "WARNING_TRANSLATION_FAILED") -- cache hits fire immediately (no model
    call needed), fresh-batch results fire right after that batch
    completes. This is the low-level hook translate_responses_batched
    builds its response-level progress bar/checkpoint logging on top of
    (see below); it fires at CHUNK granularity here, since that's this
    function's own unit of work.

    Added after the fifth external review: the cache is now saved
    (atomically -- see TranslationCache.save()) after every batch that
    reaches a model call, successful or not, not just once at the very end
    of a multi-hour run. This is what makes an interrupted run resumable:
    whatever was translated before the interruption is already durably on
    disk, and the next run's cache.get() calls above will correctly serve
    those as CACHE_HIT instead of re-translating them. A save is skipped
    when there were no misses at all (nothing new to persist -- cache.save()
    is a no-op in that case anyway via its own dirty-flag check, but
    skipping the call entirely avoids even that redundant check on a
    fully-cached rerun).
    """
    results: List[Optional[Tuple[Optional[str], str]]] = [None] * len(texts)
    miss_indices: List[int] = []

    for i, text in enumerate(texts):
        cached = cache.get(text, source_lang, target_lang, NLLB_MODEL_NAME, GENERATION_VERSION)
        if cached is not None:
            results[i] = (cached["translated_text"], "CACHE_HIT")
            if progress_callback is not None:
                progress_callback(i, "CACHE_HIT")
        else:
            miss_indices.append(i)

    batch_size = translator.batch_size
    for start in range(0, len(miss_indices), batch_size):
        batch_idx = miss_indices[start:start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        try:
            translated_batch = translator.translate_batch(batch_texts, source_lang, target_lang)
            batch_failure_tracker.record_success()
        except NLLBTranslationError as e:
            logger.warning(f"[translate_texts_batched] batch failed: {e}")
            for i in batch_idx:
                results[i] = (None, "WARNING_TRANSLATION_FAILED")
                if progress_callback is not None:
                    progress_callback(i, "WARNING_TRANSLATION_FAILED")
            batch_failure_tracker.record_failure()  # may raise NLLBTranslationError to abort
            cache.save()
            continue

        for i, translated in zip(batch_idx, translated_batch):
            if translated is None or not str(translated).strip():
                results[i] = (None, "WARNING_TRANSLATION_FAILED")
                if progress_callback is not None:
                    progress_callback(i, "WARNING_TRANSLATION_FAILED")
                continue
            cache.set(
                texts[i], source_lang, target_lang, translated,
                model_name=NLLB_MODEL_NAME,
                generation_version=GENERATION_VERSION,
                engine="local_model",
                engine_version=_pkg_version("transformers"),
            )
            results[i] = (translated, "TRANSLATED")
            if progress_callback is not None:
                progress_callback(i, "TRANSLATED")

        # Incremental cache save (item 1 of the fifth review): after every
        # successful batch, not just once at the end of the whole run. See
        # this function's docstring and TranslationCache.save() for why.
        cache.save()

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Boundary-aware chunking for long responses
# ---------------------------------------------------------------------------
#
# NLLB generates with max_length=512 SUBWORD TOKENS (see GENERATION_PARAMS).
# This corpus is translated at the RESPONSE level (one call per response,
# not per sentence), and real response lengths in data/input/interviews.json
# go up to ~740 words -- comfortably enough to exceed 512 subword tokens for
# Catalan/Spanish. Handing a too-long response straight to
# NLLBTranslator.translate_batch() would not raise an error: the tokenizer
# silently truncates at max_length, and the model would translate only the
# beginning of the response, discarding the rest with no signal anywhere in
# the output that this happened. That failure mode does NOT change this
# design's "one Spanish-standardized response" methodology -- it just means
# a response that's too long for a single generate() call must be split into
# safe pieces, translated, and rejoined BEFORE it's treated as text_es.

MAX_TRANSLATE_CHUNK_WORDS = 150
# Conservative word budget per NLLB call. This is a word-count heuristic,
# not a live tokenizer measurement, so chunking behaves identically whether
# or not a real tokenizer/model is loaded -- needed so the offline test
# suite can exercise chunk boundaries without a real NLLB checkpoint.
# Subword tokenization for Catalan/Spanish typically expands a word into
# somewhat more than one token (accents, rarer words, punctuation
# splitting); 150 words would need an average expansion ratio above ~3.4x
# to reach the 512-token generation limit, which real interview text does
# not approach -- this leaves generous headroom rather than cutting it
# close.


def _enforce_real_token_budget(
    chunks: List[str],
    token_counter: Callable[[str], int],
    token_limit: int,
) -> Tuple[List[str], int]:
    """Second-pass, tokenizer-verified safety net: re-splits any chunk
    whose ACTUAL NLLB token count (not just its word count) exceeds
    `token_limit`, by repeatedly halving it at a word boundary until every
    resulting piece fits. Only called when a real token_counter is
    available (i.e. a real NLLBTranslator, via its count_tokens()) --
    _split_into_translation_chunks' word-count heuristic is what keeps
    this from ever firing in the overwhelming majority of cases, but a
    heuristic is still a heuristic (unusually token-dense text -- rare
    words, heavy accenting, numbers -- could in principle push a
    150-word chunk over the limit). This closes that gap with the real
    measurement instead of trusting the estimate, so no chunk ever reaches
    NLLBTranslator.translate_batch() relying on truncation=True to save it.
    """
    safe_chunks: List[str] = []
    extra_splits = 0
    stack = list(reversed(chunks))
    while stack:
        chunk = stack.pop()
        if token_counter(chunk) <= token_limit:
            safe_chunks.append(chunk)
            continue
        words = chunk.split()
        if len(words) <= 1:
            # A single "word" alone exceeds the token budget (pathological
            # input). Nothing left to split on -- handed through as-is;
            # NLLBTranslator._assert_within_token_limit() will refuse to
            # translate it rather than silently truncating.
            safe_chunks.append(chunk)
            continue
        midpoint = len(words) // 2
        second_half = " ".join(words[midpoint:])
        first_half = " ".join(words[:midpoint])
        extra_splits += 1
        stack.append(second_half)
        stack.append(first_half)
    return safe_chunks, extra_splits


def _split_into_translation_chunks(
    text: str,
    preprocessor: "Preprocessor",
    lang: str,
    token_counter: Optional[Callable[[str], int]] = None,
    token_limit: int = 0,
) -> Tuple[List[str], int]:
    """Splits `text` into pieces safe to hand to a single NLLB generate()
    call, at sentence boundaries wherever possible, so a long response is
    never silently truncated. Returns (chunks, hard_split_count).

    hard_split_count counts pieces produced by either last-resort split:
    a single SENTENCE that is, by itself, already over the per-chunk word
    budget (rare -- most real sentences are well under 150 words -- but
    the corpus should never lose text just because one sentence ran
    unusually long), or a chunk that passed the word-count heuristic but
    still measured over `token_limit` real NLLB tokens once `token_counter`
    is supplied (see _enforce_real_token_budget). This mirrors, at the
    response level, the same escape hatch the retired Google-Translate-era
    design had at the sentence level (see STEP2_RELIABILITY_CHANGES.md's
    `_boundary_aware_chunks()` / `hard_split_translation_chunk_count`).

    token_counter (text -> actual NLLB token count) and token_limit are
    optional specifically so this function stays testable offline, with
    only the word-count heuristic, when no real tokenizer is available
    (see StubNLLBTranslator in test_preprocess_v2_reliability.py) -- and
    tokenizer-verified, closing the gap the word-count heuristic alone
    cannot fully guarantee, when a real NLLBTranslator is passed in (see
    translate_responses_batched).
    """
    try:
        sentences = preprocessor.split_sentences_strict(text, lang)
    except Exception:
        sentences = [text]
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    hard_split_count = 0
    current: List[str] = []
    current_words = 0

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))

    for raw_sentence in sentences:
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        word_count = len(sentence.split())

        if word_count > MAX_TRANSLATE_CHUNK_WORDS:
            # A single sentence longer than the whole per-chunk budget --
            # flush whatever was pending, then hard-split THIS sentence by
            # words as a last resort so no text is silently dropped.
            flush()
            current.clear()
            current_words = 0
            words = sentence.split()
            for start in range(0, len(words), MAX_TRANSLATE_CHUNK_WORDS):
                chunks.append(" ".join(words[start:start + MAX_TRANSLATE_CHUNK_WORDS]))
                hard_split_count += 1
            continue

        if current_words + word_count > MAX_TRANSLATE_CHUNK_WORDS and current:
            flush()
            current.clear()
            current_words = 0

        current.append(sentence)
        current_words += word_count

    flush()
    if not chunks:
        chunks = [text]

    if token_counter is not None:
        chunks, extra_splits = _enforce_real_token_budget(chunks, token_counter, token_limit)
        hard_split_count += extra_splits

    return chunks, hard_split_count


def _evenly_spaced_sample(items: List[str], sample_size: Optional[int]) -> List[str]:
    """Deterministic evenly-spaced sample across `items`, in their existing
    order -- used for the round-trip diagnostic so it doesn't just look at
    the first N responses in corpus order (which, for this corpus, means
    the first N responses of the first few interviews only). No RNG
    involved, so the sample is identical across runs given the same input,
    consistent with this project's reproducibility goals.
    """
    n = len(items)
    if sample_size is None or sample_size < 0 or sample_size >= n:
        return list(items)
    if sample_size == 0:
        return []
    step = n / sample_size
    indices = sorted({int(i * step) for i in range(sample_size)})
    return [items[i] for i in indices]


def translate_responses_batched(
    texts: List[str],
    source_lang: str,
    target_lang: str,
    preprocessor: "Preprocessor",
    translator: NLLBTranslator,
    cache: TranslationCache,
    batch_failure_tracker: ConsecutiveBatchFailureTracker,
    show_progress: bool = False,
    progress_hook: Optional[Callable[[dict], None]] = None,
) -> List[Tuple[Optional[str], str, int, int]]:
    """Translates a list of (potentially long) response-level texts,
    chunking each one at sentence boundaries first (see
    _split_into_translation_chunks) so nothing is silently truncated by
    NLLB's max_length=512-subword-token generation limit. Chunks from ALL
    input texts are flattened into one batch-translation pass
    (translate_texts_batched) so batching still spans texts, not just
    chunks within one text.

    Returns one (joined_translation_or_None, status, chunk_count,
    hard_split_count) tuple per input text, in the same order as `texts`.
    status is "TRANSLATED" if at least one chunk was a fresh translation,
    "CACHE_HIT" if every chunk came from the cache, or
    "WARNING_TRANSLATION_FAILED" if ANY chunk failed -- a response is
    translated in full or not at all, never partially (a partially-
    translated response, silently mixing languages mid-text, would be
    worse than an honest failure).

    May raise NLLBTranslationError (propagated from translate_texts_batched)
    if too many consecutive chunk batches fail -- the caller handles this
    exactly as it did before chunking was introduced.

    show_progress (added after the fifth external review) turns on a live
    tqdm bar -- one tick per fully-completed RESPONSE (not per chunk; a
    multi-chunk response only advances the bar once every one of its
    chunks has resolved), captioned with running fresh/cache_hit/failed
    counts, the total chunk count, and the resolved device -- plus a
    persistent "[NLLB checkpoint]" log line every PROGRESS_CHECKPOINT_
    INTERVAL completed responses, so progress is visible even when stdout
    isn't a TTY. progress_hook, if given, is called with a plain dict
    snapshot (`{"fresh", "cache_hits", "failed", "responses_done",
    "total"}`) every time a response completes, independent of
    show_progress -- this is what makes progress counts assertable in
    tests without depending on tqdm's own rendering.

    fresh/cache_hits/failed are RESPONSE-level counts (fixed after the
    sixth external review -- they previously incremented once per CHUNK,
    so a response spanning several chunks could be counted more than once,
    silently inflating these numbers relative to the bar's own "n/total"
    reading, which was always response-level). A response's classification
    uses exactly the same rule this function's own final per-text status
    computation below uses: any failed chunk makes the whole response
    "failed"; otherwise all-chunks-cache-hit makes it a "cache hit";
    otherwise (at least one freshly-translated chunk, no failures) it's
    "fresh". `fresh + cache_hits + failed` therefore always equals
    `responses_done`, and both always reach exactly `len(texts)` once the
    whole call completes. `chunks=<total>` in the tqdm postfix is
    deliberately still the total CHUNK count for this call (context on how
    much chunking occurred), not a per-response figure -- only the
    fresh/cache_hits/failed labels were ever misleading.
    """
    per_text_chunks: List[List[str]] = []
    per_text_hard_splits: List[int] = []
    flat_chunk_texts: List[str] = []
    chunk_owner: List[Tuple[int, int]] = []  # (text_index, position_within_text)

    # When `translator` exposes a real tokenizer-backed count_tokens() (the
    # production NLLBTranslator does), wire it in as the second-pass,
    # tokenizer-verified safety net inside _split_into_translation_chunks --
    # so chunk safety is checked against actual NLLB subword tokens, not
    # just the word-count heuristic. Guarded with hasattr() rather than an
    # isinstance(translator, NLLBTranslator) check specifically so offline
    # test doubles that don't implement count_tokens() keep exercising only
    # the word-heuristic path unchanged.
    token_counter: Optional[Callable[[str], int]] = None
    if hasattr(translator, "count_tokens"):
        token_counter = functools.partial(translator.count_tokens, source_lang=source_lang)

    token_limit = GENERATION_PARAMS["max_length"]

    for text_index, text in enumerate(texts):
        chunks, hard_split_count = _split_into_translation_chunks(
            text, preprocessor, source_lang,
            token_counter=token_counter,
            token_limit=token_limit,
        )
        per_text_chunks.append(chunks)
        per_text_hard_splits.append(hard_split_count)
        for position, chunk in enumerate(chunks):
            flat_chunk_texts.append(chunk)
            chunk_owner.append((text_index, position))

    # --- Progress instrumentation (fifth external review; response-level
    # fresh/cache_hits/failed counts corrected after the sixth) -----------
    #
    # fresh/cache_hits/failed below are RESPONSE-level counts, matching what
    # the bar's own denominator (`len(texts)`, one tick per response) and
    # its "187/435" reading already promise -- NOT a count of individual
    # chunks. A response can span multiple chunks; classifying it only once
    # all of its chunks are in, using exactly the same rule
    # translate_responses_batched's own final per-text status computation
    # uses below (any failed chunk -> failed; else all-cache-hit -> cache
    # hit; else -> fresh), keeps this live count and the function's actual
    # return values in agreement. `chunks=<total>` in the postfix is a
    # separate, deliberately chunk-level number (the total chunk count for
    # this whole call) -- context on how much chunking is happening, not a
    # per-response count, so it does not need the same fix.
    total_chunks_per_text = [len(chunks) for chunks in per_text_chunks]
    completed_chunks_per_text = [0] * len(texts)
    live_chunk_statuses: List[List[Optional[str]]] = [[None] * n for n in total_chunks_per_text]
    progress_counts = {"fresh": 0, "cache_hits": 0, "failed": 0, "responses_done": 0}
    progress_start_time = time.time()
    device_label = getattr(translator, "device", "unknown")

    pbar = tqdm(total=len(texts), desc=f"NLLB {source_lang}->{target_lang}", unit="resp") if (show_progress and texts) else None

    def _on_chunk_done(flat_index: int, status: str) -> None:
        text_index, position = chunk_owner[flat_index]
        live_chunk_statuses[text_index][position] = status
        completed_chunks_per_text[text_index] += 1
        if completed_chunks_per_text[text_index] != total_chunks_per_text[text_index]:
            return  # this response still has chunks outstanding -- not done yet

        statuses = live_chunk_statuses[text_index]
        if any(s == "WARNING_TRANSLATION_FAILED" for s in statuses):
            progress_counts["failed"] += 1
        elif all(s == "CACHE_HIT" for s in statuses):
            progress_counts["cache_hits"] += 1
        else:
            progress_counts["fresh"] += 1
        progress_counts["responses_done"] += 1

        if pbar is not None:
            pbar.set_postfix_str(
                f"fresh={progress_counts['fresh']} cache_hits={progress_counts['cache_hits']} "
                f"failed={progress_counts['failed']} chunks={len(flat_chunk_texts)} device={device_label}"
            )
            pbar.update(1)
        if show_progress and progress_counts["responses_done"] % PROGRESS_CHECKPOINT_INTERVAL == 0:
            logger.info(
                "[NLLB checkpoint] Processed: %d/%d | Fresh translations: %d | "
                "Cache hits: %d | Failed: %d | Elapsed: %s",
                progress_counts["responses_done"], len(texts),
                progress_counts["fresh"], progress_counts["cache_hits"], progress_counts["failed"],
                _format_hms(time.time() - progress_start_time),
            )
        if progress_hook is not None:
            progress_hook(dict(progress_counts, total=len(texts)))

    progress_callback = _on_chunk_done if (show_progress or progress_hook is not None) else None

    chunk_results: List[Optional[Tuple[Optional[str], str]]] = [None] * len(flat_chunk_texts)
    if flat_chunk_texts:
        chunk_results = translate_texts_batched(
            flat_chunk_texts, source_lang, target_lang, translator, cache, batch_failure_tracker,
            progress_callback=progress_callback,
        )

    if pbar is not None:
        pbar.close()

    per_text_chunk_translations: List[List[Optional[str]]] = [[None] * len(chunks) for chunks in per_text_chunks]
    per_text_chunk_statuses: List[List[Optional[str]]] = [[None] * len(chunks) for chunks in per_text_chunks]

    for flat_index, (text_index, position) in enumerate(chunk_owner):
        result = chunk_results[flat_index] if flat_index < len(chunk_results) else None
        if result is None:
            per_text_chunk_statuses[text_index][position] = "WARNING_TRANSLATION_FAILED"
        else:
            translated_text, status = result
            per_text_chunk_translations[text_index][position] = translated_text
            per_text_chunk_statuses[text_index][position] = status

    results: List[Tuple[Optional[str], str, int, int]] = []
    for text_index in range(len(texts)):
        statuses = per_text_chunk_statuses[text_index]
        chunk_count = len(statuses)
        hard_split_count = per_text_hard_splits[text_index]
        if not statuses or any(status == "WARNING_TRANSLATION_FAILED" for status in statuses):
            results.append((None, "WARNING_TRANSLATION_FAILED", chunk_count, hard_split_count))
        else:
            joined = " ".join(t for t in per_text_chunk_translations[text_index] if t)
            status = "CACHE_HIT" if all(s == "CACHE_HIT" for s in statuses) else "TRANSLATED"
            results.append((joined, status, chunk_count, hard_split_count))

    return results


# ---------------------------------------------------------------------------
# Main per-corpus processing
# ---------------------------------------------------------------------------

def process(
    raw_data: Dict[str, Dict[str, str]],
    preprocessor: Preprocessor,
    cache: TranslationCache,
    translator: NLLBTranslator,
    similarity_scorer: Optional[SemanticSimilarityScorer] = None,
    roundtrip_sample_size: int = DEFAULT_ROUNDTRIP_SAMPLE_SIZE,
    show_progress: bool = False,
) -> Tuple[dict, dict, dict]:
    process_start_time = time.time()

    # Diagnostic-only runtime metadata (fifth external review): printed/
    # logged up front (before any translation work starts) so it's visible
    # regardless of how long the run then takes, and recorded in the report
    # below. Never affects STEP2_VALID -- see get_runtime_architecture_info().
    runtime_arch_info = get_runtime_architecture_info()
    if runtime_arch_info["warning"]:
        logger.warning(runtime_arch_info["warning"])
        print(f"\nWARNING: {runtime_arch_info['warning']}\n")

    responses_out: Dict[str, dict] = {}
    sentences_out: Dict[str, dict] = {}

    warnings: List[dict] = []
    fatal_errors: List[dict] = []
    excluded: List[dict] = []

    per_interview_language: Dict[str, dict] = {}
    detection_method_counts: Dict[str, int] = {}
    unexpected_english_response_ids: List[dict] = []

    seen_response_ids = set()
    seen_sentence_ids = set()

    batch_failure_tracker = ConsecutiveBatchFailureTracker()
    translation_aborted_early = False
    abort_reason: Optional[str] = None

    input_response_count = sum(len(qs) for qs in raw_data.values())

    # --- Pass 1: language detection + collect the Catalan responses that
    # actually require translation, so they can be sent through the local
    # model in batches instead of one call per response. ---
    pending: List[dict] = []  # one entry per non-excluded response, in order

    for interview_id, questions in raw_data.items():
        primary_lang, primary_detail = compute_interview_primary_language(questions)
        per_interview_language[interview_id] = {"primary_language": primary_lang, **primary_detail}

        for question_id, raw_text in questions.items():
            response_id = f"{interview_id}::{question_id}"

            if response_id in seen_response_ids:
                fatal_errors.append({
                    "id": response_id, "stage": "response_id_generation",
                    "status": "FATAL", "reason": "duplicate_response_id",
                })
                continue
            seen_response_ids.add(response_id)

            if not raw_text or not raw_text.strip():
                excluded.append({
                    "id": response_id, "stage": "response_ingest",
                    "status": "EXCLUDED", "reason": "empty_original_response",
                })
                responses_out[response_id] = {
                    "interview_id": interview_id,
                    "question_id": question_id,
                    "original_text": raw_text,
                    "status": "EXCLUDED",
                    "exclusion_reason": "empty_original_response",
                }
                continue

            source_language, method, flagged_conf = determine_response_language(raw_text, primary_lang)
            detection_method_counts[method] = detection_method_counts.get(method, 0) + 1
            if method == "interview_fallback_unexpected_english":
                unexpected_english_response_ids.append({"id": response_id, "confidence": flagged_conf})
                warnings.append({
                    "id": response_id, "stage": "response_language_detection",
                    "status": "WARNING", "reason": "unexpected_english_response_detected",
                })

            pending.append({
                "response_id": response_id,
                "interview_id": interview_id,
                "question_id": question_id,
                "source_language": source_language,
                "language_detection_method": method,
                "interview_primary_language": primary_lang,
                "original_text": raw_text,
                "translation_required": source_language == "ca",
            })

    # --- Batched ca->es translation for every response that needs it.
    # Chunked at sentence boundaries (translate_responses_batched) so a
    # long response is never silently truncated by NLLB's
    # max_length=512-subword-token generation limit. ---
    ca_indices = [i for i, r in enumerate(pending) if r["translation_required"]]
    ca_texts = [pending[i]["original_text"] for i in ca_indices]

    translation_results: List[Optional[Tuple[Optional[str], str, int, int]]] = [None] * len(ca_texts)
    if ca_texts:
        try:
            translation_results = translate_responses_batched(
                ca_texts, "ca", "es", preprocessor, translator, cache, batch_failure_tracker,
                show_progress=show_progress,
            )
        except NLLBTranslationError as e:
            translation_aborted_early = True
            abort_reason = str(e)
            logger.error(f"[process] {e}")
            # Whatever wasn't attempted yet stays WARNING_TRANSLATION_FAILED
            # below via the None entries already in translation_results.

    for list_idx, response_idx in enumerate(ca_indices):
        result = translation_results[list_idx] if list_idx < len(translation_results) else None
        if result is None:
            pending[response_idx]["text_es"] = None
            pending[response_idx]["text_es_status"] = "WARNING_TRANSLATION_FAILED"
            pending[response_idx]["translation_chunk_count"] = 0
            pending[response_idx]["translation_hard_split_count"] = 0
        else:
            text_es, status, chunk_count, hard_split_count = result
            pending[response_idx]["text_es"] = text_es
            pending[response_idx]["text_es_status"] = status
            pending[response_idx]["translation_chunk_count"] = chunk_count
            pending[response_idx]["translation_hard_split_count"] = hard_split_count

    for response_idx, r in enumerate(pending):
        if not r["translation_required"]:
            r["text_es"] = r["original_text"]
            r["text_es_status"] = "IDENTITY_COPY"
            r["translation_chunk_count"] = 0
            r["translation_hard_split_count"] = 0

    # --- Pass 2: per-response quality diagnostics, topic-text prep, and
    # sentence splitting (on the SPANISH text, once). ---
    for r in pending:
        response_id = r["response_id"]
        interview_id = r["interview_id"]
        question_id = r["question_id"]
        text_es = r["text_es"]
        text_es_status = r["text_es_status"]

        quality: Dict[str, Optional[dict]] = {
            "target_language_check": None,
            "degenerate_output": None,
            "length_diagnostics": None,
            "semantic_preservation_similarity": None,
        }

        if text_es_status == "WARNING_TRANSLATION_FAILED":
            warnings.append({
                "id": response_id, "stage": "response_translation_ca_es",
                "status": "WARNING", "reason": "translation_failed",
            })
            topic_text_es_raw = None
            topic_text_es_clean = None
            topic_clean_empty = None
        else:
            if r["translation_required"]:
                quality["target_language_check"] = check_target_language(text_es, "es")
                if not quality["target_language_check"]["passed"]:
                    warnings.append({
                        "id": response_id, "stage": "translation_quality",
                        "status": "WARNING", "reason": "target_language_mismatch",
                    })
                quality["degenerate_output"] = check_degenerate_output(r["original_text"], text_es)
                if quality["degenerate_output"]["flagged"]:
                    warnings.append({
                        "id": response_id, "stage": "translation_quality",
                        "status": "WARNING", "reason": "degenerate_output",
                    })
                quality["length_diagnostics"] = compute_length_diagnostics(r["original_text"], text_es)
                # semantic_preservation_similarity is filled in AFTER this
                # loop, in one batched embedding call across every
                # translated response (see below) -- not here, one pair at
                # a time, which would reload/run the embedding model once
                # per response for no benefit.

            topic_text_es_raw = text_es
            topic_text_es_clean = preprocessor.full_preprocess_v2(text_es, "es")
            topic_clean_empty = not topic_text_es_clean.strip()
            if topic_clean_empty:
                warnings.append({
                    "id": response_id, "stage": "topic_text_clean",
                    "status": "WARNING",
                    "reason": "topic_text_es_clean_empty_after_lemmatization_stopword_removal",
                })

        sentence_ids_for_response: List[str] = []
        if text_es_status != "WARNING_TRANSLATION_FAILED":
            raw_sentences = preprocessor.split_sentences_strict(text_es, "es")
            for idx, sent_text in enumerate(raw_sentences):
                sentence_id = f"{response_id}::s{idx:03d}"
                if sentence_id in seen_sentence_ids:
                    fatal_errors.append({
                        "id": sentence_id, "stage": "sentence_id_generation",
                        "status": "FATAL", "reason": "duplicate_sentence_id",
                    })
                    continue
                seen_sentence_ids.add(sentence_id)
                sentence_ids_for_response.append(sentence_id)
                sentences_out[sentence_id] = {
                    "sentence_id": sentence_id,
                    "response_id": response_id,
                    "interview_id": interview_id,
                    "question_id": question_id,
                    "sentence_index": idx,
                    "text_es": sent_text,
                    "source_language": r["source_language"],
                    "translation_required": r["translation_required"],
                }

        responses_out[response_id] = {
            "interview_id": interview_id,
            "question_id": question_id,
            "source_language": r["source_language"],
            "language_detection_method": r["language_detection_method"],
            "interview_primary_language": r["interview_primary_language"],
            "original_text": r["original_text"],
            "translation_required": r["translation_required"],
            "text_es": text_es,
            "text_es_status": text_es_status,
            "topic_text_es_raw": topic_text_es_raw,
            "topic_text_es_clean": topic_text_es_clean,
            "topic_text_es_clean_empty": topic_clean_empty,
            "quality": quality,
            "sentence_ids": sentence_ids_for_response,
            "translation_chunk_count": r.get("translation_chunk_count", 0),
            "translation_hard_split_count": r.get("translation_hard_split_count", 0),
            "status": "OK",
        }

    # --- Semantic-preservation similarity for every translated response,
    # in ONE batched embedding call rather than one call per response. ---
    similarity_pairs = [
        (rid, resp["original_text"], resp["text_es"])
        for rid, resp in responses_out.items()
        if resp.get("translation_required") and resp.get("text_es_status") in ("TRANSLATED", "CACHE_HIT")
    ]
    if similarity_pairs and similarity_scorer is not None and similarity_scorer.available:
        sims = similarity_scorer.batch_cosine_similarity(
            [p[1] for p in similarity_pairs], [p[2] for p in similarity_pairs],
        )
        for (rid, _, _), sim in zip(similarity_pairs, sims):
            responses_out[rid]["quality"]["semantic_preservation_similarity"] = sim

    # --- Round-trip diagnostic on a sample of the successfully-translated
    # Catalan responses: ca -> es (already have it) -> ca, compare to the
    # original Catalan via embeddings. Sampled by default because it
    # doubles translation cost; report the distribution, no threshold.
    # The sample is evenly spaced across the corpus (not just the first N
    # responses in corpus order, which for this corpus would mean the
    # first few interviews only) -- see _evenly_spaced_sample. ---
    roundtrip_summary = {"sample_size": 0, "sampled_response_ids": [], "similarity": _distribution_summary([])}
    translated_ok_ids = [
        rid for rid, resp in responses_out.items()
        if resp.get("translation_required") and resp.get("text_es_status") in ("TRANSLATED", "CACHE_HIT")
    ]
    if translated_ok_ids and similarity_scorer is not None and similarity_scorer.available and not translation_aborted_early:
        sample_ids = _evenly_spaced_sample(translated_ok_ids, roundtrip_sample_size)
        sample_texts_es = [responses_out[rid]["text_es"] for rid in sample_ids]
        try:
            roundtrip_results = translate_responses_batched(
                sample_texts_es, "es", "ca", preprocessor, translator, cache, ConsecutiveBatchFailureTracker(),
                show_progress=show_progress,
            )
            roundtrip_texts = [res[0] if res and res[0] else "" for res in roundtrip_results]
            original_texts = [responses_out[rid]["original_text"] for rid in sample_ids]
            valid_pairs = [(o, rt) for o, rt in zip(original_texts, roundtrip_texts) if rt]
            if valid_pairs:
                sims = similarity_scorer.batch_cosine_similarity(
                    [p[0] for p in valid_pairs], [p[1] for p in valid_pairs],
                )
                roundtrip_summary = {
                    "sample_size": len(valid_pairs),
                    "sampled_response_ids": sample_ids,
                    "similarity": _distribution_summary(sims),
                }
        except NLLBTranslationError as e:
            logger.warning(f"[process] round-trip diagnostic skipped after a translation failure: {e}")

    # --- cross-checks (no silent failures) ---
    unmapped_sentences = [sid for sid, s in sentences_out.items() if s["response_id"] not in responses_out]
    accounted_for = len(responses_out)
    silent_loss = (accounted_for != input_response_count) or bool(unmapped_sentences) or bool(fatal_errors)

    duplicate_response_id_count = sum(1 for f in fatal_errors if f["reason"] == "duplicate_response_id")
    duplicate_sentence_id_count = sum(1 for f in fatal_errors if f["reason"] == "duplicate_sentence_id")

    translation_required_ids = [rid for rid, resp in responses_out.items() if resp.get("translation_required")]
    translation_required_count = len(translation_required_ids)
    translation_successful_count = sum(
        1 for rid in translation_required_ids if responses_out[rid]["text_es_status"] in ("TRANSLATED", "CACHE_HIT")
    )
    translation_failed_count = sum(
        1 for rid in translation_required_ids if responses_out[rid]["text_es_status"] == "WARNING_TRANSLATION_FAILED"
    )
    # Response-level fresh-vs-cached split of translation_successful_count,
    # for the runtime-info reporting added after the fifth external review
    # (matches what the live progress bar/checkpoint counts during the
    # run -- see translate_responses_batched).
    translation_fresh_count = sum(
        1 for rid in translation_required_ids if responses_out[rid]["text_es_status"] == "TRANSLATED"
    )
    translation_cache_hit_count = sum(
        1 for rid in translation_required_ids if responses_out[rid]["text_es_status"] == "CACHE_HIT"
    )
    missing_text_es_count = sum(1 for resp in responses_out.values() if resp.get("status") == "OK" and resp.get("text_es") is None)
    missing_topic_text_es_raw_count = sum(
        1 for resp in responses_out.values() if resp.get("status") == "OK" and resp.get("topic_text_es_raw") is None
    )

    # NOTE: target_language_check["passed"] is now tri-state (True / False /
    # None -- None means "not assessed, text too short to judge", see
    # check_target_language). `not passed` would be True for None too,
    # silently double-counting short/unassessed responses as language
    # mismatches -- this must check `passed is False` explicitly (caught in
    # the second external review).
    language_mismatch_ids = [
        rid for rid, resp in responses_out.items()
        if resp["quality"]["target_language_check"] is not None
        and resp["quality"]["target_language_check"]["passed"] is False
    ]
    short_text_language_not_assessed_ids = [
        rid for rid, resp in responses_out.items()
        if resp["quality"]["target_language_check"] is not None
        and resp["quality"]["target_language_check"].get("status") == "NOT_ASSESSED_SHORT_TEXT"
    ]
    degenerate_output_ids = [
        rid for rid, resp in responses_out.items()
        if resp["quality"]["degenerate_output"] is not None and resp["quality"]["degenerate_output"]["flagged"]
    ]
    # Diagnostic-only companion to degenerate_output_ids: short text whose
    # translation is identical to its source (frequently correct -- "Sí."
    # -> "Sí.") but was NOT counted as fatal degenerate output because it
    # didn't meet _is_sufficiently_long. Reported so a human can still spot-
    # check these, never gating STEP2_VALID.
    short_text_identical_output_ids = [
        rid for rid, resp in responses_out.items()
        if resp["quality"]["degenerate_output"] is not None
        and resp["quality"]["degenerate_output"]["is_identical_to_source"]
        and not resp["quality"]["degenerate_output"]["is_identical_to_source_fatal"]
    ]
    char_ratios = [
        resp["quality"]["length_diagnostics"]["char_ratio"] for resp in responses_out.values()
        if resp["quality"]["length_diagnostics"] is not None and resp["quality"]["length_diagnostics"]["char_ratio"] is not None
    ]
    token_ratios = [
        resp["quality"]["length_diagnostics"]["token_ratio"] for resp in responses_out.values()
        if resp["quality"]["length_diagnostics"] is not None and resp["quality"]["length_diagnostics"]["token_ratio"] is not None
    ]
    length_ratio_outlier_ids = [
        rid for rid, resp in responses_out.items()
        if resp["quality"]["length_diagnostics"] is not None and resp["quality"]["length_diagnostics"]["length_ratio_outlier"]
    ]
    similarity_values = [
        resp["quality"]["semantic_preservation_similarity"] for resp in responses_out.values()
        if resp["quality"]["semantic_preservation_similarity"] is not None
    ]
    topic_text_es_clean_empty_ids = [
        rid for rid, resp in responses_out.items() if resp.get("topic_text_es_clean_empty") is True
    ]

    translation_quality_summary = {
        "similarity_model_available": similarity_scorer.available if similarity_scorer is not None else False,
        "similarity_model_unavailable_reason": (
            similarity_scorer.unavailable_reason if similarity_scorer is not None else "similarity_scorer_not_provided"
        ),
        "language_mismatch_count": len(language_mismatch_ids),
        "language_mismatch_ids": language_mismatch_ids,
        "short_text_language_not_assessed_count": len(short_text_language_not_assessed_ids),
        "short_text_language_not_assessed_ids": short_text_language_not_assessed_ids,
        "degenerate_output_count": len(degenerate_output_ids),
        "degenerate_output_ids": degenerate_output_ids,
        "short_text_identical_output_count": len(short_text_identical_output_ids),
        "short_text_identical_output_ids": short_text_identical_output_ids,
        "length_ratio_outlier_count": len(length_ratio_outlier_ids),
        "length_ratio_outlier_ids": length_ratio_outlier_ids,
        "length_diagnostics": {
            "char_ratio": _distribution_summary(char_ratios),
            "token_ratio": _distribution_summary(token_ratios),
        },
        "semantic_preservation": _distribution_summary(similarity_values),
        "roundtrip": roundtrip_summary,
    }

    structural_validity = not silent_loss
    translation_completeness_validity = (
        not translation_aborted_early
        and translation_failed_count == 0
        and translation_successful_count == translation_required_count
        and missing_text_es_count == 0
        and missing_topic_text_es_raw_count == 0
    )
    # Translation-output-QUALITY validity: a genuinely serious per-response
    # failure -- the output landed in the wrong language, or is degenerate/
    # repetitive/pathologically-identical-to-source text -- is not a "review
    # this later" situation, it means Step 2 did not actually produce a
    # usable Spanish-standardized response for that item. Any of these
    # anywhere in the corpus is fatal to STEP2_VALID, same tier as
    # structural_validity/translation_completeness_validity above.
    #
    # This is DELIBERATELY narrower than "every quality diagnostic". Length-
    # ratio outliers and somewhat-low semantic similarity are real signals
    # worth a human looking at, but on their own they do not mean the
    # translation is wrong -- interview responses vary a lot in how
    # Catalan/Spanish phrasing compresses or expands, and similarity scores
    # from a general-purpose embedding model are a noisy proxy, not ground
    # truth. Gating STEP2_VALID on those would produce false negatives that
    # block a perfectly good 950-response run over a handful of unusually
    # phrased (but correctly translated) responses. They stay informational
    # -- reported in translation_quality_summary and reflected in
    # translation_sanity_status below, but never fatal.
    translation_output_validity = (
        translation_quality_summary["language_mismatch_count"] == 0
        and translation_quality_summary["degenerate_output_count"] == 0
    )
    length_ratio_outlier_count = translation_quality_summary["length_ratio_outlier_count"]
    low_similarity_count = translation_quality_summary["semantic_preservation"]["count_below_informational_bound"]
    short_text_language_not_assessed_count = translation_quality_summary["short_text_language_not_assessed_count"]
    short_text_identical_output_count = translation_quality_summary["short_text_identical_output_count"]
    if not translation_output_validity:
        # A genuinely serious failure occurred -- this is fatal (see
        # translation_output_validity above), so "REVIEW" (which implies
        # "otherwise fine, just take a look") would understate it.
        translation_sanity_status = "FAIL"
    elif (
        length_ratio_outlier_count > 0
        or low_similarity_count > 0
        or short_text_language_not_assessed_count > 0
        or short_text_identical_output_count > 0
    ):
        # Diagnostic-only signals: worth a human review pass, but never a
        # reason to block STEP2_VALID on their own. The short-text signals
        # (added after the second external review) are exactly the "Sí." /
        # "No." / acronym / number case: too little text to run langdetect
        # meaningfully on, or a source==target translation that is probably
        # correct rather than degenerate -- surfaced for a human glance,
        # never fatal.
        translation_sanity_status = "REVIEW"
    else:
        translation_sanity_status = "PASS"
    step2_valid = structural_validity and translation_completeness_validity and translation_output_validity

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_response_count": input_response_count,
        "output_response_count": len(responses_out),
        "output_sentence_count": len(sentences_out),
        "per_interview_language": per_interview_language,
        "interview_primary_language_counts": {
            lang: sum(1 for v in per_interview_language.values() if v["primary_language"] == lang)
            for lang in LANGUAGE_TIE_BREAK_ORDER
        },
        "language_detection_method_counts": detection_method_counts,
        "unexpected_english_response_count": len(unexpected_english_response_ids),
        "unexpected_english_response_ids": unexpected_english_response_ids,
        "warnings": warnings,
        "warning_count": len(warnings),
        "fatal_errors": fatal_errors,
        "fatal_error_count": len(fatal_errors),
        "excluded": excluded,
        "excluded_count": len(excluded),
        "unmapped_sentences": unmapped_sentences,
        "duplicate_response_id_count": duplicate_response_id_count,
        "duplicate_sentence_id_count": duplicate_sentence_id_count,
        "cache_stats": cache.stats(),
        "silent_loss_check": {
            "input_response_count": input_response_count,
            "accounted_for_response_count": accounted_for,
            "unmapped_sentence_count": len(unmapped_sentences),
            "fatal_error_count": len(fatal_errors),
            "silent_loss_detected": silent_loss,
        },
        "translation_aborted_early": translation_aborted_early,
        "abort_reason": abort_reason,
        "translation_required_count": translation_required_count,
        "translation_successful_count": translation_successful_count,
        "translation_failed_count": translation_failed_count,
        "missing_text_es_count": missing_text_es_count,
        "missing_topic_text_es_raw_count": missing_topic_text_es_raw_count,
        # topic_text_es_clean can legitimately end up empty after stopword
        # removal/lemmatization (e.g. a response consisting only of
        # stopwords) -- this is not itself a Step 2 failure (the old
        # pipeline excluded these from topic modelling too), but it WAS
        # previously only visible buried in `warnings`, with no aggregate
        # count. Added after the second external review specifically so
        # `950 responses` vs. `documents actually entering BERTopic` can be
        # reconciled from this report alone, without re-deriving it by hand
        # -- JsonHandler.create_topic_modeling_input() drops exactly these
        # (documents where `text.strip()` is empty), so
        # expected_topic_model_document_count is what should actually reach
        # BERTopic, before running topic modelling, not after.
        "topic_text_es_clean_empty_count": len(topic_text_es_clean_empty_ids),
        "topic_text_es_clean_empty_ids": topic_text_es_clean_empty_ids,
        "expected_topic_model_document_count": len(responses_out) - len(topic_text_es_clean_empty_ids),
        "translation_chunking": {
            "total_chunks": sum(resp.get("translation_chunk_count", 0) for resp in responses_out.values()),
            "responses_requiring_multiple_chunks": sum(
                1 for resp in responses_out.values() if resp.get("translation_chunk_count", 0) > 1
            ),
            "hard_split_chunk_count": sum(resp.get("translation_hard_split_count", 0) for resp in responses_out.values()),
            "max_chunk_words": MAX_TRANSLATE_CHUNK_WORDS,
        },
        "translation_quality_summary": translation_quality_summary,
        "structural_validity": structural_validity,
        "translation_completeness_validity": translation_completeness_validity,
        "translation_output_validity": translation_output_validity,
        "translation_sanity_status": translation_sanity_status,
        "STEP2_VALID": step2_valid,
        "model_info": translator.model_info(),
        "software_versions": _software_versions(preprocessor),
        # Reproducibility metadata added after the fourth external review:
        # confirms, in the artifact a future reader actually opens
        # (preprocessing_language_report.json), that this run's NLLB/BART
        # stack executed on PyTorch and never touched TensorFlow -- the
        # backend isolation is what let the run reach this point at all on
        # a machine with an incompatible TensorFlow install (see
        # ml_backend.py). `ml_backend_info` carries the full detail
        # (installed-but-unused TensorFlow is still reported, since that's
        # useful provenance); the two flat keys are the exact shape
        # requested for quick reconciliation without unpacking a nested dict.
        "ml_backend": "pytorch",
        "tensorflow_used": False,
        "ml_backend_info": ml_backend.get_ml_backend_info(),
        # Runtime/performance metadata added after the fifth external
        # review: a real multi-hour run on a low-memory Apple Silicon Mac
        # running x86_64 Python under Rosetta appeared "stuck" (it was
        # actually thrashing under memory pressure -- see
        # get_runtime_architecture_info() and resolve_default_batch_size()
        # above). None of this affects STEP2_VALID -- it's reproducibility/
        # performance provenance only, same spirit as ml_backend_info
        # above. Flat keys match the exact shape requested; runtime_info
        # additionally carries the Rosetta detail and warning text.
        "device": translator.device,
        "batch_size": translator.batch_size,
        "runtime_architecture": runtime_arch_info["runtime_architecture"],
        "elapsed_seconds": round(time.time() - process_start_time, 3),
        "cache_hits": cache.stats()["hits_this_run"],
        "fresh_translations": translation_fresh_count,
        "failed_translations": translation_failed_count,
        "translation_chunks": sum(resp.get("translation_chunk_count", 0) for resp in responses_out.values()),
        "runtime_info": {
            "device": translator.device,
            "batch_size": translator.batch_size,
            "runtime_architecture": runtime_arch_info["runtime_architecture"],
            "running_under_rosetta": runtime_arch_info["running_under_rosetta"],
            "rosetta_warning": runtime_arch_info["warning"],
            "elapsed_seconds": round(time.time() - process_start_time, 3),
            "cache_hits": cache.stats()["hits_this_run"],
            "cache_hit_translations": translation_cache_hit_count,
            "fresh_translations": translation_fresh_count,
            "failed_translations": translation_failed_count,
            "translation_chunks": sum(resp.get("translation_chunk_count", 0) for resp in responses_out.values()),
        },
    }

    return responses_out, sentences_out, report


# ---------------------------------------------------------------------------
# Preflight: local environment/model readiness, not internet availability
# ---------------------------------------------------------------------------

PREFLIGHT_TEST_TEXT_CA = (
    "Aquesta és una frase de prova per comprovar que el model de traducció "
    "funciona correctament abans de processar tot el corpus."
)


def run_preflight(verbose: bool = True, batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[bool, dict]:
    """Loads the NLLB tokenizer/model, resolves the device, and runs one
    real cat_Latn->spa_Latn translation to confirm non-empty, Spanish-
    detected output. This checks LOCAL environment readiness (can the
    model load, does this machine's device work) rather than the health of
    a remote translation endpoint. Precisely: the FIRST time this runs on
    a machine, `AutoTokenizer`/`AutoModelForSeq2SeqLM.from_pretrained()`
    still needs network access to download and cache the checkpoint from
    Hugging Face -- that is a one-time cost, not a runtime dependency.
    Once the checkpoint is cached locally (the normal case for every run
    after the first), no network access is needed at all, and this
    preflight check is purely local from that point on.
    """
    if verbose:
        print("\nSTEP 2 NLLB PREFLIGHT\n")

    # Printed/recorded FIRST, before the tokenizer/model load below -- this
    # is exactly the step that previously reached an incompatible
    # TensorFlow build on at least one reviewer's machine and aborted the
    # whole process outright (a native-library abort, not a catchable
    # Python exception -- see ml_backend.py). Surfacing it up front means
    # a run that still aborts here at least leaves this line in the
    # terminal output, showing the guard was in effect and pointing
    # straight at the actual model-load step as the next thing that ran.
    ml_backend_info = ml_backend.get_ml_backend_info()
    if verbose:
        print("ML backend:")
        print(f"  PyTorch available       {'PASS' if ml_backend_info['pytorch_available'] else 'FAIL'}")
        print(f"  TensorFlow required     NO (USE_TF={ml_backend_info['env']['USE_TF']})")
        print(f"  Transformers backend    {ml_backend_info['ml_backend']}")
        print()

    # Runtime-architecture warning (fifth external review): printed here too,
    # before the model load, for the same "leave a visible trail even if
    # something later aborts" reason as the ML backend block above.
    runtime_arch_info = get_runtime_architecture_info()
    if verbose:
        print(f"Runtime architecture: {runtime_arch_info['runtime_architecture']}")
        print(f"Requested batch size: {batch_size}")
        if runtime_arch_info["warning"]:
            print(f"\nWARNING: {runtime_arch_info['warning']}\n")
        else:
            print()

    result: Dict[str, object] = {
        "steps": {}, "ml_backend": ml_backend_info, "runtime_architecture_info": runtime_arch_info,
    }
    try:
        translator = NLLBTranslator(batch_size=batch_size)
        result["steps"]["tokenizer_loads"] = True
        result["steps"]["model_loads"] = True
    except Exception as e:
        result["steps"]["tokenizer_loads"] = False
        result["steps"]["model_loads"] = False
        result["error"] = str(e)
        if verbose:
            print(f"Model: {NLLB_MODEL_NAME}")
            print(f"FAILED to load tokenizer/model: {e}\n")
            print("PREFLIGHT_VALID = FALSE")
        result["PREFLIGHT_VALID"] = False
        return False, result

    if verbose:
        print(f"Model: {NLLB_MODEL_NAME}")
        print(f"Device: {translator.device}" + (" (fell back from " + translator.requested_device + ")" if translator.device_fallback else ""))

    lang_codes_ok = all(code in LANG_TO_NLLB.values() for code in ("cat_Latn", "spa_Latn"))
    result["steps"]["cat_Latn_supported"] = "cat_Latn" in LANG_TO_NLLB.values()
    result["steps"]["spa_Latn_supported"] = "spa_Latn" in LANG_TO_NLLB.values()

    try:
        translated = translator.translate_batch([PREFLIGHT_TEST_TEXT_CA], "ca", "es")[0]
        non_empty = bool(translated and translated.strip())
        lang_check = check_target_language(translated, "es") if non_empty else {"detected_language": None, "passed": False}
        result["steps"]["cat_Latn_to_spa_Latn"] = non_empty
        result["steps"]["output_language_is_spanish"] = lang_check["passed"]
        result["sample_output"] = translated
        translation_ok = non_empty and lang_check["passed"]
    except Exception as e:
        result["steps"]["cat_Latn_to_spa_Latn"] = False
        result["steps"]["output_language_is_spanish"] = False
        result["error"] = str(e)
        translation_ok = False

    result["model_info"] = translator.model_info()
    result["software_versions"] = _software_versions()

    preflight_valid = bool(lang_codes_ok and translation_ok)
    result["PREFLIGHT_VALID"] = preflight_valid

    if verbose:
        print(f"cat_Latn -> spa_Latn: {'PASS' if result['steps'].get('cat_Latn_to_spa_Latn') else 'FAIL'}")
        print(f"Output language: {'es PASS' if result['steps'].get('output_language_is_spanish') else 'FAIL'}")
        print(f"Non-empty output: {'PASS' if result['steps'].get('cat_Latn_to_spa_Latn') else 'FAIL'}")
        print(f"\nPREFLIGHT_VALID = {'TRUE' if preflight_valid else 'FALSE'}")

    return preflight_valid, result


# ---------------------------------------------------------------------------
# Smoke test: small mixed-language sample through the full pipeline,
# writing to a dedicated smoke-test directory (never production outputs).
# ---------------------------------------------------------------------------

SMOKE_TEST_DATA = {
    "smoke_interview_ca": {
        "q1": (
            "Aquesta és una entrevista de prova que parla sobre la vida al poble "
            "i sobre com han canviat les coses en els últims anys."
        ),
    },
    "smoke_interview_es": {
        "q1": (
            "Esta es una entrevista de prueba que habla sobre la vida en el pueblo "
            "y sobre como han cambiado las cosas en los últimos años."
        ),
    },
}


def run_smoke_test(verbose: bool = True, batch_size: int = DEFAULT_BATCH_SIZE) -> bool:
    if verbose:
        print("\nStep 2 smoke test")
        print("Running a small mixed-language sample through the full pipeline...\n")

    preprocessor = Preprocessor()
    os.makedirs(SMOKE_TEST_DIR, exist_ok=True)
    smoke_cache_path = os.path.join(SMOKE_TEST_DIR, "_smoke_test_cache.json")
    cache = TranslationCache(smoke_cache_path)

    try:
        translator = NLLBTranslator(batch_size=batch_size)
        similarity_scorer = SemanticSimilarityScorer()
        responses_out, sentences_out, report = process(
            SMOKE_TEST_DATA, preprocessor, cache, translator, similarity_scorer,
        )
    finally:
        if os.path.exists(smoke_cache_path):
            os.remove(smoke_cache_path)

    JsonHandler.create_json(
        {"metadata": {"generated_at": report["generated_at"], "count": len(responses_out)}, "responses": responses_out},
        "smoke_test_responses.json", SMOKE_TEST_DIR,
    )
    JsonHandler.create_json(report, "smoke_test_report.json", SMOKE_TEST_DIR)

    if report.get("translation_aborted_early"):
        if verbose:
            print(f"Smoke test FAILED: run aborted early -- {report.get('abort_reason')}")
        return False

    all_ok = True
    for response_id in sorted(responses_out):
        response = responses_out[response_id]
        label = "Catalan response" if response["interview_id"].endswith("_ca") else (
            "Spanish response" if response["interview_id"].endswith("_es") else response_id
        )
        if verbose:
            print(f"{label}:")

        response_sentences = [s for s in sentences_out.values() if s["response_id"] == response_id]
        checks = {
            "original_text": bool(response.get("original_text")),
            "source_language": response.get("source_language"),
            "text_es": bool(response.get("text_es")) and (
                response["text_es"] != response["original_text"] if response["translation_required"] else True
            ),
            "topic_text_es_raw": bool(response.get("topic_text_es_raw")),
            "sentence_ids": bool(response_sentences),
            "translation_required": response.get("translation_required"),
        }
        all_ok = all_ok and bool(checks["original_text"]) and bool(checks["text_es"]) and bool(checks["topic_text_es_raw"]) and bool(checks["sentence_ids"])

        if verbose:
            print(f"  original_text       {'✓' if checks['original_text'] else '✗'}")
            print(f"  source_language     {checks['source_language']}")
            print(f"  text_es             {'✓' if checks['text_es'] else '✗' if response['translation_required'] else '✓ identity'}")
            print(f"  topic_text_es_raw   {'✓' if checks['topic_text_es_raw'] else '✗'}")
            print(f"  sentence IDs        {'✓' if checks['sentence_ids'] else '✗'}")
            print(f"  translation needed  {'Yes' if checks['translation_required'] else 'No'}")
            print()

    if verbose:
        print(f"Outputs written to {SMOKE_TEST_DIR}/ (not the production data/output/ files).")
        if all_ok:
            print("Smoke test PASSED: all required fields populated for both languages.")
        else:
            print("Smoke test FAILED: one or more required fields did not populate.")

    return all_ok


# ---------------------------------------------------------------------------
# Validation report printing + orchestration
# ---------------------------------------------------------------------------

def print_validation_report(report: dict) -> None:
    silent_loss_check = report["silent_loss_check"]
    tqs = report["translation_quality_summary"]
    sim = tqs["semantic_preservation"]
    rt = tqs["roundtrip"]["similarity"]

    print("\nStep 2 validation report")
    print(f"Input responses                      {report['input_response_count']}")
    print(f"Output responses                      {report['output_response_count']}")
    print()
    print(f"Duplicate response IDs                {report['duplicate_response_id_count']}")
    print(f"Duplicate sentence IDs                {report['duplicate_sentence_id_count']}")
    print(f"Unmapped sentences                    {silent_loss_check['unmapped_sentence_count']}")
    print()
    print(f"Catalan responses requiring NLLB      {report['translation_required_count']}")
    print(f"Successful ca->es translations        {report['translation_successful_count']}")
    print(f"Failed translations                   {report['translation_failed_count']}")
    print()
    print(f"Missing text_es                       {report['missing_text_es_count']}")
    print(f"Missing topic_text_es_raw             {report['missing_topic_text_es_raw_count']}")
    print()
    print(f"topic_text_es_clean empty (excluded)  {report['topic_text_es_clean_empty_count']}")
    print(f"Expected topic-model documents         {report['expected_topic_model_document_count']}")
    print()
    chunking = report["translation_chunking"]
    print(f"Total NLLB translation chunks         {chunking['total_chunks']}")
    print(f"Responses needing >1 chunk             {chunking['responses_requiring_multiple_chunks']}")
    print(f"Hard-split chunks (oversized sentence) {chunking['hard_split_chunk_count']}")
    print()
    print(f"Language mismatches                   {tqs['language_mismatch_count']}")
    print(f"  (short text, not assessed)           {tqs['short_text_language_not_assessed_count']}")
    print(f"Degenerate outputs                    {tqs['degenerate_output_count']}")
    print(f"  (short text, identical -- informational) {tqs['short_text_identical_output_count']}")
    print(f"Length-ratio outliers                 {tqs['length_ratio_outlier_count']}")
    print()
    print("Semantic similarity (original vs. translation):")
    print(f"  mean                                {sim['mean']}")
    print(f"  median                               {sim['median']}")
    print(f"  p05                                  {sim['p05']}")
    print(f"  minimum                              {sim['min']}")
    print()
    print(f"Round-trip diagnostic (sample_size={tqs['roundtrip']['sample_size']}):")
    print(f"  mean                                {rt['mean']}")
    print(f"  median                               {rt['median']}")
    print(f"  p05                                  {rt['p05']}")
    print(f"  minimum                              {rt['min']}")
    print()
    print(f"Structural validity                  {report['structural_validity']}")
    print(f"Translation completeness             {report['translation_completeness_validity']}")
    print(f"Translation output validity          {report['translation_output_validity']}")
    print(f"Translation sanity                   {report['translation_sanity_status']}")
    print()
    print(f"STEP2_VALID                          {report['STEP2_VALID']}")
    print()
    print(f"ML backend                           {report.get('ml_backend', 'unknown')}")
    print(f"TensorFlow used                      {report.get('tensorflow_used', 'unknown')}")
    print()
    print(f"Device                                {report.get('device', 'unknown')}")
    print(f"Batch size                            {report.get('batch_size', 'unknown')}")
    print(f"Runtime architecture                  {report.get('runtime_architecture', 'unknown')}")
    print(f"Fresh translations                    {report.get('fresh_translations', 'unknown')}")
    print(f"Cache hits                            {report.get('cache_hits', 'unknown')}")
    print(f"Elapsed                               {_format_hms(report.get('elapsed_seconds', 0))}")
    runtime_info = report.get("runtime_info") or {}
    if runtime_info.get("rosetta_warning"):
        print(f"\nWARNING: {runtime_info['rosetta_warning']}")
    if report.get("translation_aborted_early"):
        print(f"\nRun aborted early: {report.get('abort_reason')}")


def main(mode: str = "full", batch_size: Optional[int] = None) -> bool:
    # batch_size=None means "not explicitly requested" -- resolved here
    # (device-aware, after the fifth external review) rather than baked
    # into a fixed default, so an MPS/low-memory Mac gets a conservative
    # batch size automatically while an explicit --batch-size always wins.
    # This one auto-detect call is cheap (no model load); every downstream
    # call (run_preflight/run_smoke_test/NLLBTranslator) receives a
    # concrete int from here on.
    auto_selected = batch_size is None
    if auto_selected:
        batch_size = resolve_default_batch_size()
        if batch_size != DEFAULT_BATCH_SIZE:
            logger.info(
                f"No --batch-size given; auto-detected device suggests a low-memory "
                f"default of {batch_size} (override with --batch-size N)."
            )

    if mode == "preflight":
        passed, _ = run_preflight(batch_size=batch_size)
        return passed

    if mode == "smoke-test":
        return run_smoke_test(batch_size=batch_size)

    preflight_passed, _ = run_preflight(verbose=True, batch_size=batch_size)
    if not preflight_passed:
        logger.error(
            "Preflight failed -- aborting before processing the corpus. "
            "Run `python preprocess_v2.py --preflight` for full detail."
        )
        return False

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return False

    raw_data = JsonHandler.read_json(INPUT_FILE)
    if not raw_data:
        logger.error("No data loaded from input file")
        return False

    preprocessor = Preprocessor()
    cache = TranslationCache(CACHE_PATH)
    translator = NLLBTranslator(batch_size=batch_size)
    similarity_scorer = SemanticSimilarityScorer()

    responses_out, sentences_out, report = process(
        raw_data, preprocessor, cache, translator, similarity_scorer,
        show_progress=True,
    )

    # Final safety-net save: translate_texts_batched already saves the
    # cache atomically after every batch during process() (see the fifth
    # external review), so this is normally a no-op by the time we get
    # here -- kept as a defensive last write in case anything was left
    # dirty (e.g. the roundtrip diagnostic's own cache writes).
    cache.save()

    JsonHandler.create_json(
        {"metadata": {"generated_at": report["generated_at"], "count": len(responses_out)}, "responses": responses_out},
        "preprocessed_responses_v2.json", OUTPUT_DIR,
    )
    JsonHandler.create_json(
        {"metadata": {"generated_at": report["generated_at"], "count": len(sentences_out)}, "sentences": sentences_out},
        "preprocessed_sentences_v2.json", OUTPUT_DIR,
    )
    JsonHandler.create_json(report, "preprocessing_language_report.json", OUTPUT_DIR)

    print_validation_report(report)

    if report["STEP2_VALID"]:
        logger.info(
            f"Step 2 complete: {len(responses_out)} responses, "
            f"{len(sentences_out)} sentences, STEP2_VALID=True"
        )
    else:
        logger.error(
            "Step 2 FAILED: structural_validity="
            f"{report['structural_validity']}, translation_completeness_validity="
            f"{report['translation_completeness_validity']}, translation_output_validity="
            f"{report['translation_output_validity']} (translation_sanity_status="
            f"{report['translation_sanity_status']}). See "
            "preprocessing_language_report.json for full detail. Step 2 is NOT complete."
        )

    return report["STEP2_VALID"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2 preprocessing: structural parsing, sentence alignment, NLLB Catalan->Spanish translation."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--preflight", action="store_true",
        help="Only verify the local NLLB model loads and translates ca->es correctly, then exit.",
    )
    group.add_argument(
        "--smoke-test", action="store_true",
        help="Run a small mixed-language sample through the full pipeline (writes to data/output/smoke_test/ only), then exit.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Number of texts translated per model call. Default: {DEFAULT_BATCH_SIZE}, "
             f"or {LOW_MEMORY_DEVICE_BATCH_SIZE} automatically on MPS (Apple Silicon's "
             "GPU backend, which shares unified memory with the OS -- a large batch "
             "there can drive a low-memory Mac into swapping). Always overrides the "
             "automatic choice when given explicitly.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.preflight:
        ok = main(mode="preflight", batch_size=args.batch_size)
    elif args.smoke_test:
        ok = main(mode="smoke-test", batch_size=args.batch_size)
    else:
        ok = main(mode="full", batch_size=args.batch_size)
    sys.exit(0 if ok else 1)
