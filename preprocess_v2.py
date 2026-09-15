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
import re
import sys
import time
import hashlib
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
# A SEPARATE cache for the Round 7 one-sentence-per-translation redesign
# (segment_source_sentences / translate_frozen_sentences_batched), never
# the same file as CACHE_PATH above. The two caches are keyed on the exact
# same text-hashing scheme (TranslationCache._key), but the TEXTS being
# looked up are now individual frozen sentences, not multi-sentence
# ~150-word chunks -- almost none of CACHE_PATH's existing entries would
# ever hit against sentence-level lookups anyway, but the real reason for
# a separate file is explicit, not incidental: the six-hour real run that
# is CACHE_PATH's content stays byte-for-byte as it was, auditable
# (see evidence/round7_full_corpus_run_2026-09-07/), never silently
# overwritten by a later run against the new architecture.
SENTENCE_CACHE_PATH = "data/cache/nllb_sentence_translation_cache_v1.json"
FROZEN_SEGMENTATION_PATH = "data/frozen_source_sentence_segmentation_v1.json"
# Bumped from 1 -> 2 when input_sha256 became a required field (see
# validate_frozen_segmentation_against_input() and _hash_file() below) --
# a frozen file written before this addition is a real, structural gap
# (main() would have no way to detect a since-edited interviews.json), so
# it is treated as unusable, not silently accepted with the field missing.
FROZEN_SEGMENTATION_SCHEMA_VERSION = 2


def _hash_file(path: str) -> Optional[str]:
    """SHA-256 of a file's raw bytes, or None if it can't be read (missing,
    permission error, etc.) -- callers must treat None as "hash unknown",
    never as a value that could coincidentally match a recorded hash.
    Deliberately the same algorithm/chunking as terminal.py's own
    _hash_file() (kept as two small independent copies rather than an
    import, since terminal.py imports preprocess_v2, not the reverse) --
    both must produce the same digest for the same file.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANG_TO_NLLB = {"ca": "cat_Latn", "es": "spa_Latn"}
REQUIRED_TRANSLATION_DIRECTION = "ca->es"  # the only direction this corpus requires

# Deterministic decoding: no sampling, fixed beam count. Bump
# GENERATION_VERSION any time these change -- it is baked into the cache
# key (see translation_cache.py) specifically so a settings change can
# never silently serve a translation generated under the old settings.
#
# Round 12: this is the PRIMARY configuration -- the one every Catalan
# sentence is translated with FIRST, and the one the real corpus's
# existing ~5.8k-entry cache was built under. It deliberately does NOT
# carry repetition_penalty/no_repeat_ngram_size (see RETRY_GENERATION_
# PARAMS below for those) and GENERATION_VERSION is deliberately back to
# its pre-Round-10 value -- both reverted from Round 10's global
# antirep1 change. Round 10 added those two parameters HERE, applied to
# every sentence, and bumped this version, which would have forced the
# entire real cache to be discarded and the whole ~5,792-sentence corpus
# re-translated just to fix ~37 sentences. Round 11 confirmed (with the
# user, explicitly, twice) that this is unnecessary and undesirable:
# undesirable because applying an anti-repetition penalty to EVERY
# sentence risks subtly changing wording on the ~5,755 sentences that
# were never broken (this corpus's interview speech genuinely contains
# repetition, hesitation, and emphasis that a global penalty would
# suppress indiscriminately), and unnecessary because the fix only needs
# to reach the sentences that actually looped. See RETRY_GENERATION_
# PARAMS/RETRY_GENERATION_VERSION and attempt_repetition_fallback_retry()
# below for the targeted mechanism that replaced the global change.
# Reverting this version means the real corpus's existing cache (built
# entirely under this exact configuration) is fully valid again -- a run
# against unchanged raw input hits it for all ~5,792 primary translations,
# with zero NLLB calls needed for anything already known-good.
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

# Round 12: a SEPARATE, ADDITIONAL generation configuration and cache
# namespace used ONLY by attempt_repetition_fallback_retry() -- one
# targeted call for a sentence whose PRIMARY translation (under
# GENERATION_PARAMS/GENERATION_VERSION above) was independently confirmed
# degenerate by check_degenerate_output()'s repetition_flag, never applied
# speculatively or corpus-wide.
#   - repetition_penalty=1.3 is a soft, proportional penalty on the score
#     of any already-generated token, applied every step. It discourages
#     but never forbids repetition, so it does not distort short,
#     legitimately-repetitive translations ("No, no, no." -> "No, no,
#     no.").
#   - no_repeat_ngram_size=4 is a hard constraint: once any 4-token
#     sequence has been generated, that exact sequence can never recur in
#     the same output. This directly makes the observed failure mode --
#     the same short phrase repeating dozens of times -- structurally
#     impossible, while a 4-gram floor (rather than the 3-gram size
#     check_degenerate_output() uses for detection) still leaves room for
#     genuine short 3-word repeats ("el tema de ... el tema de") that show
#     up naturally in this corpus's rambling interview speech.
# Chosen as a standard, moderate combination for this exact NMT beam-
# search pathology rather than a more aggressive setting (e.g.
# no_repeat_ngram_size=2 or 3) that would force paraphrasing of ordinary
# short emphatic repeats. This has not been validated against the real
# corpus by an actual run yet -- that run is the user's next step after
# reviewing this fix, consistent with this project never running the real
# NLLB corpus on its own. RETRY_GENERATION_VERSION is a distinct cache
# key namespace from GENERATION_VERSION, not a successor to it -- a
# sentence can have entries under BOTH (its primary attempt, and a
# fallback retry), and the primary cache is never touched or invalidated
# by anything written under this version.
RETRY_GENERATION_PARAMS = {
    **GENERATION_PARAMS,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 4,
}
RETRY_GENERATION_VERSION = "nllb600M-beams4-maxlen512-chunked-v2-antirep-retry1"

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
# clears both bars.
#
# Round 10: raised from 30 chars / 5 words after the first real
# 950-response corpus run flagged 146 sentences, 105 of which (72%) were
# manually verified -- every one, full read, not sampled -- to be
# fluent, grammatically correct Spanish translations of short/informal
# transcribed speech that langdetect's statistical char-n-gram model
# simply misjudges at these lengths (misdetected as Portuguese, Catalan,
# Italian, French, English, Somali, Tagalog, German, often at >0.99
# confidence; see STEP2_NLLB_CHANGES.md's Round 10 section for the full
# diagnosis). 50 chars / 8 words was chosen from a data-driven tradeoff
# analysis against that real corpus's 4298 already-assessed sentences:
#   chars/words -> false positives resolved | correctly-passed sentences newly exempted
#   40 / 7  -> 51/105 (49%) resolved | 536/4157 (12.9%) exempted
#   45 / 7  -> 66/105 (63%) resolved | 731/4157 (17.6%) exempted
#   50 / 8  -> 78/105 (74%) resolved | 974/4157 (23.4%) exempted  <- chosen
#   55 / 9  -> 86/105 (82%) resolved | 1171/4157 (28.2%) exempted
#   60 / 10 -> 89/105 (85%) resolved | 1365/4157 (32.8%) exempted
# All five candidates still caught 100% of the 36 confirmed true-positive
# repetition-loop sentences from that same run (those have target_chars
# well over 300, far above any threshold considered). 50/8 was picked as
# the middle-ground point where the curve of newly-exempted correctly-
# passed sentences starts costing noticeably more coverage per additional
# point of false-positive resolution (each +5/+1 step past this one buys
# progressively less resolved-FP for progressively more exempted-from-
# assessment sentences), not because it is the only defensible value --
# see STEP2_NLLB_CHANGES.md's Round 10 section for the full table and
# reasoning if this needs revisiting.
#
# Round 11: target_language_check's result is no longer part of any
# FATAL gate at any length (see check_target_language()'s docstring and
# translation_output_validity's computation below) -- a manual, full read
# of every one of the 141 real language-mismatch flags from the Round 10
# run found ZERO real translation failures that were caught by language
# detection and NOT already independently caught by check_degenerate_
# output(); the 36 that were real failures were real because of
# repetition, not language. So this threshold no longer decides what's
# FATAL, only what's worth including in the REVIEW-tier diagnostic
# (NOT_ASSESSED_SHORT_TEXT text is excluded from language reporting
# entirely, same as before) -- it stays at 50/8 because that reasoning is
# unaffected by the fatality change, not because it needed re-justifying.
MIN_CHARS_FOR_DIRECT_DETECTION = 50
MIN_WORDS_FOR_DIRECT_DETECTION = 8

LANGUAGE_TIE_BREAK_ORDER = ["ca", "es"]

# Translation-sanity diagnostics (informational; see STEP2_VALID logic
# below for why these never gate validity on their own).
LENGTH_RATIO_LOW = 0.4
LENGTH_RATIO_HIGH = 3.0
REPETITION_NGRAM_SIZE = 3
# Round 11: raised from 4 to 6. A manual review of the Round 10 run's 41
# repetition flags found 4 false positives -- long, otherwise correctly-
# translated responses where a short, natural phrase ("se gasta el ...",
# "el plan de ...", "el tema de ...", "que hay que ...") legitimately
# repeats exactly 4 times in normal rambling interview speech. The 37
# genuine NLLB decoder repetition-loop failures in that same run all had
# their worst-repeated 3-gram appear at LEAST 15 times (most were
# 70-93+); 6 clears the observed false-positive count (4) with margin
# while staying well under the lowest real collapse case observed (15),
# so no known real failure is missed by this change.
REPETITION_MIN_REPEATS = 6  # same 3-gram appearing >=6x flags repetition
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

    def translate_batch(
        self, texts: List[str], source_lang: str, target_lang: str,
        generation_params: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        """Translates a batch of texts, all in the same direction. Retries
        once locally on failure; on a non-CPU device, a failure triggers a
        one-time, sticky fallback to CPU (recorded on the instance) before
        the retry. Raises NLLBTranslationError if the batch still fails
        after that, or immediately (no retry) if any input would exceed
        NLLB's token limit -- see _assert_within_token_limit().

        generation_params (Round 12): defaults to the module-level
        GENERATION_PARAMS (the PRIMARY configuration every normal call
        uses) when not given. Passing a different dict -- as attempt_
        repetition_fallback_retry() does, with RETRY_GENERATION_PARAMS --
        overrides decoding settings for just this one call, without
        touching the module-level default any other caller sees. This
        replaces a Round 10 design where the anti-repetition parameters
        were baked into GENERATION_PARAMS itself and applied to every
        call; see that constant's comment for why that was reverted.
        """
        if not texts:
            return []

        params = generation_params if generation_params is not None else GENERATION_PARAMS
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
                        **params,
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
    """langdetect on the OUTPUT of translation.

    Round 11: this result is diagnostic/REVIEW-only -- it is NEVER, at any
    length, part of the FATAL sentence-quality gate (see
    translation_output_validity's computation in process() for why: a
    full manual review of the 141 language-mismatch flags from the first
    real 950-response corpus run found every single one was either a
    langdetect false positive on genuinely correct Spanish (105 of them,
    often short/informal text misdetected as Portuguese/Catalan/Italian/
    etc. at >0.99 confidence), or co-occurred with a real NLLB repetition-
    loop failure that check_degenerate_output() independently, correctly
    flagged on its own (36 of them) -- i.e. language detection alone
    never contributed a genuine catch that degenerate-output detection
    didn't already make. See STEP2_NLLB_CHANGES.md's Round 11 section for
    the full breakdown.

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
    external review, and still worth excluding from the REVIEW-tier
    reporting even though it can no longer be fatal either way. passed is
    None (never False) when not assessed, so a caller that naively does
    `if not passed` on this dict without checking status would need to
    notice -- see how language_mismatch_ids below is computed (`passed is
    False`, not just falsy) for the safe pattern.
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

    Round 11: REPETITION_MIN_REPEATS raised 4 -> 6 after a manual review
    of the first real corpus run found the same 3-gram repeating exactly
    4 times legitimately in a handful of long, otherwise correctly-
    translated responses (a short natural phrase like "el tema de"
    recurring in normal rambling speech), while every genuine NLLB
    decoder repetition-loop failure in that run repeated its worst 3-gram
    at least 15 times (most 70+). is_empty and is_identical_to_source_
    fatal are unaffected by this change. Round 11 also made this function
    -- specifically repetition_flag/is_empty -- the ONLY sentence-level
    FATAL quality signal; check_target_language()'s result no longer
    contributes to `flagged` for a sentence at any length (see that
    function's docstring and translation_output_validity's computation in
    process()).
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


def attempt_repetition_fallback_retry(
    source_text: str,
    translator: "NLLBTranslator",
    cache: TranslationCache,
    source_lang: str = "ca",
    target_lang: str = "es",
) -> dict:
    """Round 12: one targeted, cache-aware retry for a sentence whose
    PRIMARY translation (under GENERATION_PARAMS/GENERATION_VERSION) the
    caller has already run through check_degenerate_output() and found
    repetition_flag=True. Never called speculatively -- process()'s Pass 2
    calls this only after independently confirming the primary result is
    a genuine repetition loop, and only once per sentence.

    Uses RETRY_GENERATION_PARAMS (anti-repetition decoding settings) and
    RETRY_GENERATION_VERSION (a cache namespace entirely separate from
    GENERATION_VERSION -- see those constants' comments). A cache hit
    here means an earlier sentence with the IDENTICAL source text already
    triggered this same retry (this corpus's cache reuse -- see
    STEP2_NLLB_CHANGES.md's Round 10 diagnosis for how much of it there
    is among the real repetition-loop cases, e.g. "No, no." alone behind
    14 flagged sentence IDs) -- so a source text pathological enough to
    need retrying once is never sent to NLLB a second time for it.

    Returns {"text_es": Optional[str], "cache_status": "CACHE_HIT" |
    "FRESH" | "FAILED", "generation_version": RETRY_GENERATION_VERSION}.
    Never raises: this runs late in a multi-thousand-sentence corpus
    pass, well after the primary translation and its own retry/abort
    logic already completed, so a failure HERE (model error, or the
    retried text still somehow exceeding the token limit) is caught and
    reported as "FAILED" (text_es=None) -- the caller keeps the sentence's
    original (still-degenerate) primary text_es in that case, never loses
    data, and the sentence correctly stays FLAGGED.

    Two fixes from external review, both applied here:

    1. A `None`/blank result is never cached and never reported as a
       success. `translate_batch()`'s underlying `tokenizer.batch_decode`
       can legitimately return an empty string for a genuinely empty
       generation (the same per-item empty-generation case the PRIMARY
       path already guards against in `translate_texts_batched` -- see
       its `if translated is None or not str(translated).strip()` check);
       without the identical guard here, a blank retry result would have
       been cached as if it were a real repaired translation and then
       served back as CACHE_HIT on every future lookup for that source
       text. A blank output is not a successful anti-repetition repair --
       it's treated exactly like any other retry failure: reported as
       "FAILED", never cached, and the caller keeps the original primary
       text_es.
    2. A successful fresh retry is saved to disk immediately
       (`cache.save()`), not left for some later batch's save to pick up.
       The PRIMARY path already treats every successful translation this
       way (see translate_texts_batched's own `cache.save()` after each
       batch, and its docstring on why) specifically so a crash later in
       the same run -- e.g. during the embeddings/round-trip diagnostic
       pass, which runs after all translation work -- cannot lose
       already-completed work. A retry-repaired sentence is exactly the
       kind of expensive, hard-won result (one extra real NLLB call, only
       ever made once per unique bad source text) that must not be
       silently redone on the next run because it was still only sitting
       in memory when the process died.
    """
    cached = cache.get(source_text, source_lang, target_lang, translator.model_name, RETRY_GENERATION_VERSION)
    if cached is not None:
        return {
            "text_es": cached["translated_text"], "cache_status": "CACHE_HIT",
            "generation_version": RETRY_GENERATION_VERSION,
        }
    try:
        translated = translator.translate_batch(
            [source_text], source_lang, target_lang, generation_params=RETRY_GENERATION_PARAMS,
        )[0]
    except Exception as e:
        logger.warning(
            f"[attempt_repetition_fallback_retry] fallback retry failed for a repetition-flagged "
            f"sentence -- keeping its original (still-degenerate) primary translation: {e}"
        )
        return {"text_es": None, "cache_status": "FAILED", "generation_version": RETRY_GENERATION_VERSION}

    if translated is None or not str(translated).strip():
        logger.warning(
            "[attempt_repetition_fallback_retry] fallback retry produced a blank/empty result for "
            "a repetition-flagged sentence -- a blank output is not a successful repair, so it is "
            "NOT cached; keeping the original (still-degenerate) primary translation."
        )
        return {"text_es": None, "cache_status": "FAILED", "generation_version": RETRY_GENERATION_VERSION}

    cache.set(
        source_text, source_lang, target_lang, translated,
        model_name=translator.model_name, generation_version=RETRY_GENERATION_VERSION,
    )
    # Immediate, crash-safe persistence -- see point 2 in the docstring
    # above. cache.save() is atomic (write-temp -> fsync -> os.replace(),
    # see TranslationCache.save()) and a no-op-safe warning-only failure,
    # exactly like every other cache.save() call site in this module.
    cache.save()
    return {"text_es": translated, "cache_status": "FRESH", "generation_version": RETRY_GENERATION_VERSION}


def compute_primary_cache_coverage(
    pending: List[dict],
    cache: TranslationCache,
    translator: "NLLBTranslator",
    source_lang: str = "ca",
    target_lang: str = "es",
) -> dict:
    """Round 12, added from external review: a cheap, read-only pre-flight
    check of how much of this run's Catalan sentence-translation work is
    already sitting in the PRIMARY cache under the current GENERATION_
    VERSION, computed BEFORE any NLLB call is made.

    Why this exists: every code delivery for this project is, deliberately,
    code-only -- it never bundles the real corpus's actual data/cache/
    file (see STEP2_NLLB_CHANGES.md's "Files in this delivery"). If an
    operator replaces their project folder with a new delivery and forgets
    to copy their existing warm cache back into place, `process()` would
    otherwise silently proceed to re-translate the entire corpus from
    scratch -- turning what should be a cheap, targeted repetition-repair
    run (a handful of new NLLB calls) into another multi-hour full run,
    with no warning until it's already too late to stop. This function
    (and process()'s `require_warm_primary_cache` gate that uses it) is
    that warning, given up front.

    `pending` is process()'s own Pass-1 list of per-response dicts (each
    with "translation_required" and "sentences": [{"sentence_id",
    "sentence_index", "text_source"}, ...]) -- this function does not
    reload or resegment anything, it only reads what Pass 1 already built.

    Coverage is counted by attempting the exact cache lookup for EVERY one
    of the corpus's Catalan sentence UNITS individually (one `contains()`
    call per sentence, for every response where translation_required is
    True) -- deliberately not by counting distinct cache entries or
    distinct source texts first. Two different sentence IDs can share one
    source text (documented cache-reuse behavior throughout this project;
    see attempt_repetition_fallback_retry()'s docstring for a real
    example, "No, no." behind 14 sentence IDs in the real corpus) and
    therefore one cache key -- but every sentence UNIT still individually
    needs a translation to exist for the corpus to be considered fully
    covered, however many distinct keys that maps to underneath. Counting
    distinct entries instead would silently under-count what "coverage"
    actually needs to mean here.

    Uses TranslationCache.contains(), not get() -- a read-only existence
    check that does NOT increment hits_this_run/misses_this_run. Using
    get() here would inflate this run's real cache-hit statistics by
    however many sentences this check walks, ahead of process()'s own
    genuine lookups for those same sentences moments later.

    Returns {"expected_catalan_sentence_units": int, "covered": int,
    "missing": int, "missing_sentence_ids": List[str] (capped -- see
    below), "missing_sentence_id_count": int, "safe_to_reuse_primary_
    cache": bool (True exactly when missing == 0)}. `missing_sentence_ids`
    is capped at 50 entries (with missing_sentence_id_count always the
    true total) purely so a catastrophically-cold-cache run doesn't dump
    thousands of IDs into a printed report or the JSON report file.
    """
    expected = 0
    covered = 0
    missing_sentence_ids: List[str] = []
    missing_cap = 50
    for r in pending:
        if not r["translation_required"]:
            continue
        for sr in r["sentences"]:
            expected += 1
            if cache.contains(sr["text_source"], source_lang, target_lang, translator.model_name, GENERATION_VERSION):
                covered += 1
            else:
                if len(missing_sentence_ids) < missing_cap:
                    missing_sentence_ids.append(sr["sentence_id"])
    missing = expected - covered
    return {
        "expected_catalan_sentence_units": expected,
        "covered": covered,
        "missing": missing,
        "missing_sentence_ids": missing_sentence_ids,
        "missing_sentence_id_count": missing,
        "safe_to_reuse_primary_cache": missing == 0,
    }


def print_cache_coverage_report(coverage: dict) -> None:
    """Prints compute_primary_cache_coverage()'s result in the exact shape
    an operator needs to see, unmissably, before a multi-hour translation
    pass either does or doesn't start -- printed by process() itself
    (not deferred to print_validation_report(), which only runs at the
    very end) specifically so this is the LAST thing visible before the
    expensive part of a real run begins.
    """
    print("\nPrimary cache coverage before run:")
    print(f"  expected Catalan sentence units: {coverage['expected_catalan_sentence_units']}")
    print(f"  covered:                         {coverage['covered']}")
    print(f"  missing:                         {coverage['missing']}")
    print(f"  SAFE TO REUSE PRIMARY CACHE = {coverage['safe_to_reuse_primary_cache']}")
    if coverage["missing"] > 0:
        shown = coverage["missing_sentence_ids"]
        more = coverage["missing_sentence_id_count"] - len(shown)
        print(f"  missing sentence IDs (first {len(shown)}): {shown}" + (f" (+{more} more)" if more > 0 else ""))


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

    Genuinely lazy: __init__ does no model construction at all -- the
    (potentially large) SentenceTransformer download/load only happens the
    first time `available` or `batch_cosine_similarity` is actually called,
    via `_ensure_loaded()`, and at most once (`_load_attempted` makes every
    later call a no-op). This matters on memory-constrained machines: a run
    with no similarity pairs to score (e.g. an all-Spanish-original corpus)
    never pays the load cost, and callers that only want to know whether a
    load was already attempted (rather than forcing one) should check
    `load_attempted` instead of reading `available` directly.
    """

    def __init__(self, model_name: str = SIMILARITY_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.unavailable_reason: Optional[str] = None
        self._load_attempted = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:  # ImportError or a download/load failure
            self.unavailable_reason = str(e)
            logger.warning(
                f"[SemanticSimilarityScorer] could not load '{self.model_name}': {e}. "
                "Semantic-preservation and round-trip similarity will be omitted "
                "from this run's report."
            )

    @property
    def load_attempted(self) -> bool:
        """True once a load has been attempted (success or failure), without
        forcing one. Use this to inspect state without triggering the lazy
        load as a side effect."""
        return self._load_attempted

    @property
    def available(self) -> bool:
        """Whether the model loaded successfully. Triggers the lazy load on
        first call -- this property IS the 'first actual use' trigger, so
        only call it at a real use site (guarded by an actual need, as the
        existing `similarity_pairs and ... .available` call sites already
        are). Reporting code that must not force a load should check
        `load_attempted` first."""
        self._ensure_loaded()
        return self.model is not None

    def batch_cosine_similarity(self, texts_a: List[str], texts_b: List[str]) -> List[float]:
        if not texts_a:
            return []
        if not self.available:
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
# Segmentation repair: ellipsis-continuation merge (Round 7 segmentation
# audit, source-side sentence redesign)
# ---------------------------------------------------------------------------
#
# preprocessor.split_sentences_strict() (spaCy sentence boundaries) is
# generally reliable on this corpus -- the historical "-Des" / "que vam
# posar el Fibra, un 10." bad split from the old target-side-split file does
# NOT reproduce with it (verified directly against the raw response). But
# a source-side segmentation audit across all 950 raw responses (see
# evidence/round7_full_corpus_run_2026-09-07/) found one real, narrow,
# recurring artifact: this is transcribed spoken interview speech, and a
# speaker trailing off mid-clause ("..."/"…") is routinely treated by
# spaCy's parser-based sentence boundary detector as a full sentence end,
# splitting the continuation off as its own "sentence" even though it's
# clearly the same clause resuming. 38 confirmed cases across the corpus,
# all reading as same-speaker disfluency continuations on manual review,
# none as a topic or speaker shift -- including one utterance split into
# THREE fragments by two separate ellipsis breaks:
#   "-I allà van acabar ficant sensors..."
#   "-...de camions com a tal, per saber que realment arriba a granja el
#    que s'havia..."
#   "-...enviat."
#
# This matters more once sentence boundaries become permanent, auditable
# sentence IDs (the whole point of the source-side redesign) rather than
# an intermediate detail re-derived after translation -- a segmentation
# artifact here would otherwise get baked into the corpus permanently.

_ELLIPSIS_TRAILING_RE = re.compile(r"(\.\.\.|…)\s*$")
_LEADING_TURN_MARKER_RE = re.compile(r"^[-–—]\s*")
_LEADING_ELLIPSIS_RE = re.compile(r"^(\.\.\.|…)\s*")
_OPENING_PUNCT_CHARS = "\"'‘’“”(¿¡"


def _looks_like_ellipsis_continuation(sentence: str) -> bool:
    """True if `sentence`, after stripping a leading speaker-turn dash and
    any leading ellipsis remnant, starts with a lowercase letter -- the
    signature of a spoken-transcript disfluency continuation rather than a
    genuinely new sentence. A digit, quote, or opening-punctuation start is
    never treated as suspicious (not what this pattern looks like).

    The leading dash is stripped before the check -- deliberately -- even
    though a leading "-" is this corpus's speaker-turn marker. The audit
    found this corpus reuses "-" for two different things: a genuine new
    speaker turn, AND resuming after an ellipsis-marked pause by the SAME
    speaker (e.g. "-I allà van acabar ficant sensors..." / "-...de
    camions..." -- the dash reappears on the resumed fragment even though
    it's the same utterance). So the dash alone is not a safe never-merge
    signal here. What IS safe is requiring the PRECEDING sentence to
    independently end in an ellipsis (see merge_ellipsis_continuations) --
    a genuine new speaker turn being immediately preceded by a trailing-off
    AND itself happening to start with a lowercase word was not observed in
    any of the 38 audited cases.
    """
    s = _LEADING_TURN_MARKER_RE.sub("", sentence)
    s = _LEADING_ELLIPSIS_RE.sub("", s)
    for ch in s:
        if ch.isalpha():
            return ch.islower()
        if ch.isdigit() or ch in _OPENING_PUNCT_CHARS:
            return False
    return False


def merge_ellipsis_continuations(sentences: List[str]) -> Tuple[List[str], List[dict]]:
    """Repairs the ellipsis-continuation segmentation artifact described
    above. Only ever merges sentence i into sentence i+1, and only when
    BOTH (a) sentence i ends in an ellipsis and (b)
    _looks_like_ellipsis_continuation(sentence i+1) is True. Applied
    iteratively (a `while`, not a single pass) so a multi-fragment chain
    (the three-fragment "sensors.../...de camions.../...enviat." example)
    collapses fully into one sentence, not just one link at a time: after
    a merge, the newly-merged sentence is re-checked against whatever
    follows it before moving on.

    Only ever operates on the single list of sentences passed in -- i.e.
    only ever merges within one response's own sentences. There is no
    cross-response merging: this function has no notion of "response" at
    all, and preprocess_v2.process() calls split_sentences_strict() (and
    would call this) once per response, on that response's own sentence
    list only.

    The merged text strips the transcription artifacts (the trailing
    ellipsis on the first fragment, the leading dash/ellipsis on the
    fragment(s) being folded in) rather than concatenating them verbatim,
    since the result is meant to read as one clean sentence, not a
    concatenation of fragments with stray punctuation in the middle. The
    first fragment's own leading marker (if it has one) is left untouched
    -- it's the true start of the merged sentence.

    Returns (merged_sentences, merge_log). merge_log has one entry per
    merge actually performed -- {"before": [text_i, text_i+1], "after":
    merged_text} -- specifically so every merge can be reviewed against
    the original text before sentence IDs are frozen, rather than trusted
    blindly.
    """
    result = list(sentences)
    merge_log: List[dict] = []
    i = 0
    while i < len(result) - 1:
        current = result[i]
        nxt = result[i + 1]
        if _ELLIPSIS_TRAILING_RE.search(current) and _looks_like_ellipsis_continuation(nxt):
            head = _ELLIPSIS_TRAILING_RE.sub("", current).rstrip()
            tail = _LEADING_TURN_MARKER_RE.sub("", nxt)
            tail = _LEADING_ELLIPSIS_RE.sub("", tail).lstrip()
            merged = f"{head} {tail}".strip()
            merge_log.append({"before": [current, nxt], "after": merged})
            result[i] = merged
            del result[i + 1]
            continue  # re-check the merged sentence against its new neighbor
        i += 1
    return result, merge_log


def segment_source_sentences(
    text: str,
    lang: str,
    preprocessor: "Preprocessor",
) -> Tuple[List[str], List[dict]]:
    """The single canonical source-side segmentation path: real spaCy
    sentence boundaries (split_sentences_strict) followed by the
    ellipsis-continuation repair (merge_ellipsis_continuations). This is
    deliberately the ONE place that combines the two steps, so the frozen
    sentence-ID structure and the (future) one-sentence-per-translation
    redesign are guaranteed to segment text identically -- neither can
    drift from the other by calling the two steps in a different order or
    forgetting the merge step.

    Returns (sentences, merge_log) -- merge_log is whatever
    merge_ellipsis_continuations() logged, passed through unchanged, so a
    caller that wants an audit trail (e.g. the sentence-ID-freezing script)
    doesn't have to re-derive it.
    """
    try:
        raw_sentences = preprocessor.split_sentences_strict(text, lang)
    except Exception:
        raw_sentences = [text]
    if not raw_sentences:
        raw_sentences = [text]
    return merge_ellipsis_continuations(raw_sentences)


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
# One-sentence-per-translation (Round 7, step 6): the primary ca->es
# translation path, built on the frozen source-side sentence structure.
# translate_responses_batched/_split_into_translation_chunks above are
# DELIBERATELY left in place, unchanged, and still used -- but only for
# the secondary round-trip diagnostic sample (translating a handful of
# already-reconstructed whole responses back to Catalan for a similarity
# check), never for primary corpus translation any more.
# ---------------------------------------------------------------------------

def translate_frozen_sentences_batched(
    segmented_responses: Dict[str, dict],
    translator: NLLBTranslator,
    cache: TranslationCache,
    batch_failure_tracker: ConsecutiveBatchFailureTracker,
    show_progress: bool = False,
    progress_hook: Optional[Callable[[dict], None]] = None,
) -> Dict[str, dict]:
    """Translates every frozen source sentence independently -- one NLLB
    call per sentence (batched across ALL responses, not chunked within
    one), never several sentences bundled into a shared translation unit.
    This is what makes NLLB structurally unable to merge or drop content
    across a sentence boundary any more: there is no boundary inside any
    single translation unit for it to lose. See the Round 7 segmentation
    audit and evidence/round7_full_corpus_run_2026-09-07/ for the failure
    mode this replaces.

    `segmented_responses` is response_id -> {"source_language": "ca"|"es",
    "sentences": [{"sentence_id", "sentence_index", "text_source"}, ...]}
    -- the shape process() assembles from segment_source_sentences() (or
    from the frozen data/frozen_source_sentence_segmentation_v1.json
    file's own "responses" dict, when process() is given one explicitly).

    A Spanish-source sentence is never sent to NLLB -- reproduced
    unchanged with translation_provenance "SOURCE_ES". A Catalan sentence
    is one translation unit unless it is, by itself, too long for one NLLB
    call (rare -- the audit found a real 179-word single sentence); in
    that one case it is hard-split the same tokenizer-verified way
    _enforce_real_token_budget already does elsewhere in this module,
    translated as pieces, and rejoined into that ONE sentence's text_es.
    A failure on any piece fails the whole sentence -- never a silently
    partial sentence. This is the old code's all-or-nothing-per-unit
    principle, now applied at the correct grain: per sentence, not per
    response, so one bad sentence no longer takes its neighbors down with
    it (see process() for how a response is reconstructed from whatever
    its sentences actually produced).

    Failed sentences get exactly one automatic retry, as a second, smaller
    flat batch, after every sentence has had its first attempt -- cheap at
    sentence grain, and safe: everything that already succeeded is already
    durably cached (translate_texts_batched saves incrementally) before
    this second pass even starts, so a retry-pass failure can never lose
    first-pass progress. The retry pass uses its OWN
    ConsecutiveBatchFailureTracker, deliberately not the caller's -- a bad
    retry pass should not spend the same abort budget the main pass may
    have already partially used, and a retry-pass abort is logged and
    treated as "retry didn't help" (those sentences stay FAILED) rather
    than propagated as a run-aborting error.

    translation_provenance per sentence is one of: SOURCE_ES, CACHE_HIT,
    FRESH_OK, RETRY_OK, FAILED. Quality-diagnostic flagging (FLAGGED) is
    deliberately NOT decided here -- exactly like the response-level
    design this replaces, that judgment belongs to the caller (process()),
    using check_target_language/check_degenerate_output at sentence grain
    now -- keeping "did NLLB produce something" (this function's job)
    separate from "is what it produced trustworthy" (the caller's job).

    show_progress/progress_hook mirror translate_responses_batched's own,
    and are genuinely per-SENTENCE -- there is no "wait for every chunk of
    a response before ticking the bar" bookkeeping to do, because the
    sentence IS the unit now. Progress is also genuinely INCREMENTAL: a
    cache hit ticks essentially immediately (translate_texts_batched's own
    progress_callback fires for cache hits before any model call), and a
    fresh translation ticks the moment the batch containing its piece(s)
    returns -- not after every batch in the whole corpus has finished. On
    a real multi-thousand-sentence corpus this is the difference between a
    progress bar that visibly advances throughout the run and one that
    sits at 0/N for the entire first pass and then jumps to N/N in a
    single burst right before returning (a real symptom this fixes -- see
    the incremental-progress-reporting revision below this docstring for
    the mechanism). This is done via the same technique translate_
    responses_batched already used for chunk-level progress: translate_
    texts_batched's own progress_callback fires per PIECE as soon as its
    outcome is known, and a small piece-completion counter per sentence
    ticks the outer bar/counters/progress_hook the moment ALL of a given
    sentence's piece(s) -- almost always exactly one -- have reported in.
    Building the actual `results` entries (the joined text) still happens
    from the full returned list after each translate_texts_batched call,
    exactly as before -- only the progress *signaling* moved earlier; the
    translation data itself is unchanged and no less correct. A defensive
    end-of-function reconciliation pass guarantees every Catalan sentence
    is ticked exactly once no matter what: idempotent per-sentence ticking
    means a sentence already accounted for live is never double-counted,
    and any sentence that somehow never got a live tick (the only way that
    can happen is the retry pass raising NLLBTranslationError partway
    through, after some of its own callbacks already fired) is ticked at
    the end using its REAL final outcome from `results` -- so the live
    counters can never end up disagreeing with the data they describe, and
    the bar always finishes at exactly total_ca_sentences.

    Returns sentence_id -> {"text_es": Optional[str],
    "translation_provenance": str, "hard_split_count": int}. May raise
    NLLBTranslationError, propagated from the FIRST-pass translate_texts_
    batched call only (see above for why the retry pass never propagates
    one) -- the caller handles this exactly as the old response-level path
    did (translation_aborted_early).
    """
    token_counter: Optional[Callable[[str], int]] = None
    if hasattr(translator, "count_tokens"):
        token_counter = functools.partial(translator.count_tokens, source_lang="ca")
    token_limit = GENERATION_PARAMS["max_length"]

    results: Dict[str, dict] = {}
    sentence_source_text: Dict[str, str] = {}  # ca sentence_id -> its own text_source (for retry)
    per_sentence_hard_splits: Dict[str, int] = {}

    flat_pieces: List[str] = []
    piece_owner: List[str] = []  # sentence_id per flat piece (repeats for a rare hard-split sentence)
    piece_position_in_sentence: List[int] = []  # this piece's 0-based position within its own sentence
    sentence_piece_total: Dict[str, int] = {}  # sentence_id -> how many pieces it has, total

    for response_id, r in segmented_responses.items():
        source_language = r["source_language"]
        for s in r["sentences"]:
            sentence_id = s["sentence_id"]
            text = s["text_source"]

            if source_language != "ca":
                results[sentence_id] = {
                    "text_es": text, "translation_provenance": "SOURCE_ES", "hard_split_count": 0,
                }
                continue

            sentence_source_text[sentence_id] = text
            if token_counter is not None:
                pieces, hard_splits = _enforce_real_token_budget([text], token_counter, token_limit)
            else:
                pieces, hard_splits = [text], 0
            per_sentence_hard_splits[sentence_id] = hard_splits
            for piece in pieces:
                piece_position_in_sentence.append(sentence_piece_total.get(sentence_id, 0))
                sentence_piece_total[sentence_id] = sentence_piece_total.get(sentence_id, 0) + 1
                flat_pieces.append(piece)
                piece_owner.append(sentence_id)

    total_ca_sentences = len(sentence_source_text)
    progress_counts = {"cache_hits": 0, "fresh": 0, "retried_ok": 0, "failed": 0, "sentences_done": 0}
    progress_start_time = time.time()
    device_label = getattr(translator, "device", "unknown")
    pbar = tqdm(total=total_ca_sentences, desc="NLLB ca->es (sentence)", unit="sent") if (show_progress and total_ca_sentences) else None

    _ticked_sentences: set = set()

    def _tick_progress(sentence_id: str, provenance: str) -> None:
        # Idempotent: a sentence is counted exactly once no matter how many
        # times something tries to tick it -- the live per-piece callbacks
        # below, and the end-of-function reconciliation pass, can both
        # reach the same sentence_id in the rare retry-pass-abort case, and
        # only the first tick (whichever happens first) should count.
        if sentence_id in _ticked_sentences:
            return
        _ticked_sentences.add(sentence_id)
        progress_counts["sentences_done"] += 1
        if provenance == "CACHE_HIT":
            progress_counts["cache_hits"] += 1
        elif provenance == "RETRY_OK":
            progress_counts["retried_ok"] += 1
        elif provenance == "FAILED":
            progress_counts["failed"] += 1
        else:  # FRESH_OK
            progress_counts["fresh"] += 1
        if pbar is not None:
            pbar.set_postfix_str(
                f"cache_hits={progress_counts['cache_hits']} fresh={progress_counts['fresh']} "
                f"retried_ok={progress_counts['retried_ok']} failed={progress_counts['failed']} "
                f"device={device_label}"
            )
            pbar.update(1)
        if show_progress and progress_counts["sentences_done"] % PROGRESS_CHECKPOINT_INTERVAL == 0:
            logger.info(
                "[NLLB sentence checkpoint] Processed: %d/%d | Cache hits: %d | Fresh: %d | "
                "Retried OK: %d | Failed: %d | Elapsed: %s",
                progress_counts["sentences_done"], total_ca_sentences,
                progress_counts["cache_hits"], progress_counts["fresh"],
                progress_counts["retried_ok"], progress_counts["failed"],
                _format_hms(time.time() - progress_start_time),
            )
        if progress_hook is not None:
            progress_hook(dict(progress_counts, total=total_ca_sentences))

    def _record(sentence_id: str, text_es: Optional[str], provenance: str) -> None:
        results[sentence_id] = {
            "text_es": text_es, "translation_provenance": provenance,
            "hard_split_count": per_sentence_hard_splits.get(sentence_id, 0),
        }

    # --- First pass: every Catalan sentence's piece(s), cache-aware,
    # batched across ALL responses at once. Progress ticks incrementally,
    # per sentence, as translate_texts_batched's own progress_callback
    # reports each piece's outcome -- see the docstring above. May raise
    # NLLBTranslationError (propagated -- see docstring). ---
    first_pass_piece_status: Dict[str, List[Optional[str]]] = {
        sid: [None] * n for sid, n in sentence_piece_total.items()
    }
    first_pass_completed_pieces: Dict[str, int] = {sid: 0 for sid in sentence_piece_total}
    first_pass_failed: List[str] = []

    def _on_first_pass_piece_done(flat_index: int, status: str) -> None:
        sentence_id = piece_owner[flat_index]
        position = piece_position_in_sentence[flat_index]
        first_pass_piece_status[sentence_id][position] = status
        first_pass_completed_pieces[sentence_id] += 1
        if first_pass_completed_pieces[sentence_id] != sentence_piece_total[sentence_id]:
            return  # this sentence still has piece(s) outstanding
        statuses = first_pass_piece_status[sentence_id]
        if any(s == "WARNING_TRANSLATION_FAILED" for s in statuses):
            # Deferred to the retry pass below -- not ticked yet, exactly
            # like the original bulk-finalize code deferred it (a failed
            # sentence isn't "done" until retry has had its shot).
            first_pass_failed.append(sentence_id)
            return
        provenance = "CACHE_HIT" if all(s == "CACHE_HIT" for s in statuses) else "FRESH_OK"
        _tick_progress(sentence_id, provenance)

    piece_results = translate_texts_batched(
        flat_pieces, "ca", "es", translator, cache, batch_failure_tracker,
        progress_callback=_on_first_pass_piece_done,
    ) if flat_pieces else []

    per_sentence_pieces: Dict[str, List[Tuple[Optional[str], str]]] = {}
    for flat_index, sentence_id in enumerate(piece_owner):
        per_sentence_pieces.setdefault(sentence_id, []).append(piece_results[flat_index])

    first_pass_failed_set = set(first_pass_failed)
    for sentence_id, piece_list in per_sentence_pieces.items():
        if sentence_id in first_pass_failed_set:
            continue  # handled after the retry pass below
        statuses = [p[1] for p in piece_list]
        joined = " ".join(p[0] for p in piece_list if p[0])
        provenance = "CACHE_HIT" if all(s == "CACHE_HIT" for s in statuses) else "FRESH_OK"
        _record(sentence_id, joined, provenance)

    # --- Second pass: one automatic retry for whatever failed on the
    # first pass. Deliberately does not re-derive hard-split pieces --
    # retries the SAME piece texts the first pass attempted (a piece
    # already measured safely within the token budget failing is a
    # transient/model-call failure, not a size problem -- see
    # _assert_within_token_limit's own docstring on why an oversized input
    # is never retried the same way). Progress ticks incrementally here
    # too, via its own per-piece callback, so a long retry pass is visible
    # as it runs rather than only once it (or its abort-recovery fallback)
    # finishes. ---
    if first_pass_failed:
        retry_pieces: List[str] = []
        retry_owner: List[str] = []
        retry_position: List[int] = []
        retry_piece_total: Dict[str, int] = {}
        for i, sentence_id in enumerate(piece_owner):
            if sentence_id not in first_pass_failed_set:
                continue
            retry_position.append(retry_piece_total.get(sentence_id, 0))
            retry_piece_total[sentence_id] = retry_piece_total.get(sentence_id, 0) + 1
            retry_pieces.append(flat_pieces[i])
            retry_owner.append(sentence_id)

        retry_piece_status: Dict[str, List[Optional[str]]] = {sid: [None] * n for sid, n in retry_piece_total.items()}
        retry_completed_pieces: Dict[str, int] = {sid: 0 for sid in retry_piece_total}

        def _on_retry_piece_done(flat_index: int, status: str) -> None:
            sentence_id = retry_owner[flat_index]
            position = retry_position[flat_index]
            retry_piece_status[sentence_id][position] = status
            retry_completed_pieces[sentence_id] += 1
            if retry_completed_pieces[sentence_id] != retry_piece_total[sentence_id]:
                return
            statuses = retry_piece_status[sentence_id]
            provenance = "FAILED" if any(s == "WARNING_TRANSLATION_FAILED" for s in statuses) else "RETRY_OK"
            _tick_progress(sentence_id, provenance)

        try:
            retry_results = translate_texts_batched(
                retry_pieces, "ca", "es", translator, cache, ConsecutiveBatchFailureTracker(),
                progress_callback=_on_retry_piece_done,
            )
        except NLLBTranslationError as e:
            logger.warning(f"[translate_frozen_sentences_batched] retry pass aborted, retried sentences stay FAILED: {e}")
            retry_results = [(None, "WARNING_TRANSLATION_FAILED")] * len(retry_pieces)

        per_sentence_retry_pieces: Dict[str, List[Tuple[Optional[str], str]]] = {}
        for flat_index, sentence_id in enumerate(retry_owner):
            per_sentence_retry_pieces.setdefault(sentence_id, []).append(retry_results[flat_index])

        for sentence_id in first_pass_failed:
            piece_list = per_sentence_retry_pieces.get(sentence_id, [])
            statuses = [p[1] for p in piece_list]
            if piece_list and not any(s == "WARNING_TRANSLATION_FAILED" for s in statuses):
                joined = " ".join(p[0] for p in piece_list if p[0])
                _record(sentence_id, joined, "RETRY_OK")
            else:
                _record(sentence_id, None, "FAILED")

    # Safety-net reconciliation (see the docstring's incrementality
    # paragraph): guarantees every Catalan sentence is ticked exactly once
    # no matter how the retry pass above actually played out, including
    # the rare case where it raises partway through after some of its own
    # callbacks already fired. Ticks here always use the sentence's REAL
    # final outcome from `results` (already fully populated by this
    # point), so the live counters can never disagree with the data they
    # describe, and the bar always reaches exactly total_ca_sentences.
    for sentence_id in sentence_source_text:
        if sentence_id not in _ticked_sentences:
            _tick_progress(sentence_id, results[sentence_id]["translation_provenance"])

    if pbar is not None:
        pbar.close()

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
    frozen_segmentation: Optional[Dict[str, dict]] = None,
    require_warm_primary_cache: bool = False,
) -> Tuple[dict, dict, dict]:
    """`frozen_segmentation`, when given, is the "responses" dict from
    data/frozen_source_sentence_segmentation_v1.json -- response_id ->
    {"source_language": ..., "sentences": [{"sentence_id",
    "sentence_index", "text_source"}, ...]}. This is the real corpus's
    sentence IDs, frozen and archived per the Round 7 segmentation audit
    (evidence/round7_full_corpus_run_2026-09-07/), never recomputed live
    for a production run. main() always passes this. When it's None
    (the default -- used by every test in this suite, which builds its own
    small synthetic `raw_data` with no matching frozen file), this
    function computes the same segmentation live via segment_source_
    sentences() -- byte-identical to what freeze_sentence_segmentation.py
    would produce for the same input and code version, just not tied to an
    on-disk artifact. Either way, sentence IDs are `response_id::sNNN`
    from the same segmentation function -- process() never re-splits
    translated Spanish text into sentences any more (see the removed
    "Pass 2" target-side split this replaces).

    `require_warm_primary_cache` (Round 12, added from external review):
    default False, so this parameter changes nothing for any existing
    caller (every test in this suite, and terminal.py's menu-driven path,
    which is intentionally NOT tied to the frozen production corpus -- see
    _create_processed_versions()'s own comment on that). When True,
    process() computes primary-cache coverage across every Catalan
    sentence this run's `pending` set would need to translate (see
    compute_primary_cache_coverage()) BEFORE making any NLLB call, prints
    that report unconditionally either way, and -- if even one sentence is
    missing from the primary cache under the CURRENT GENERATION_VERSION --
    refuses to call translate_frozen_sentences_batched() at all. The run
    is not silently allowed to fall through into a full (re-)translation;
    instead it fails exactly the way any other aborted translation pass
    does (translation_aborted_early=True, abort_reason set, every pending
    Catalan sentence recorded as FAILED, STEP2_VALID=False), with the
    coverage numbers themselves recorded in the returned report under
    "primary_cache_coverage" for the JSON audit trail. main()'s
    `--require-warm-cache` CLI flag is what sets this True for the real
    corpus run -- see there for why the default here stays False (a
    genuinely first-ever run has an empty cache by definition, and must
    not be blocked by this same gate).
    """
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
    sentence_length_ratio_outlier_ids: List[str] = []
    sentence_language_mismatch_ids: List[str] = []
    repetition_fallback_retry_resolved_ids: List[str] = []
    repetition_fallback_retry_unresolved_ids: List[str] = []
    # Round 13 (external review): a response whose LIVE per-response
    # language detection disagrees with what the FROZEN segmentation file
    # recorded for it -- diagnostic/REVIEW-only, never fatal, never allowed
    # to discard frozen sentences or flip translation_required. See the
    # frozen-entry lookup below and STEP2_NLLB_CHANGES.md's Round 13
    # section for why the frozen language is authoritative here.
    source_language_mismatch_ids: List[str] = []

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

            # Source-side sentence segmentation (Round 7 redesign): the
            # frozen structure when the caller supplied one (main() always
            # does, for a real corpus run), otherwise computed live via the
            # exact same canonical function the freeze script uses --
            # byte-identical for the same code version, just not tied to
            # an on-disk artifact (see this function's docstring).
            frozen_entry = frozen_segmentation.get(response_id) if frozen_segmentation is not None else None
            if frozen_segmentation is not None and frozen_entry is None:
                # This response has no frozen record at all -- still fatal.
                # In the real main() CLI path this should never actually
                # fire: validate_frozen_segmentation_against_input() already
                # confirms, before process() is ever called, that the
                # frozen file's response ID set exactly matches the current
                # corpus's. It stays here as defense-in-depth for any other
                # caller (tests, or a future caller) that hands process() a
                # frozen_segmentation dict without going through that outer
                # check first -- a response process() cannot find ANY
                # frozen sentence structure for is not something the frozen
                # LANGUAGE authority below can paper over.
                fatal_errors.append({
                    "id": response_id, "stage": "frozen_segmentation_lookup",
                    "status": "FATAL", "reason": "missing_from_frozen_segmentation",
                })
                sentence_records = []
            elif frozen_entry is not None:
                # Round 13 (external review): frozen segmentation AND the
                # frozen file's recorded source_language are BOTH
                # authoritative once a frozen file is in play -- never the
                # live per-response detector above. validate_frozen_
                # segmentation_against_input() (called once, in main(),
                # before process() ever runs) already confirms this exact
                # response's original_text is byte-identical between the
                # frozen file and the current corpus, so the language the
                # freeze script recorded for this unchanged text is exactly
                # as trustworthy as a fresh detection call over the same
                # text would be -- there is no scientific reason to prefer
                # a NEW live guess over the frozen one for the SAME bytes.
                # A disagreement here is real and worth a look (langdetect
                # is a known source of false positives throughout this
                # project -- see Round 10/11), but it must never discard
                # this response's frozen sentences, never flip
                # translation_required, and never be fatal: doing so was a
                # real bug (see STEP2_NLLB_CHANGES.md's Round 13 section)
                # that silently dropped real frozen sentences (6396 ->
                # 6393 in the real corpus) and mis-routed two genuinely
                # Spanish responses into "requires NLLB translation".
                if frozen_entry.get("source_language") != source_language:
                    source_language_mismatch_ids.append(response_id)
                    warnings.append({
                        "id": response_id, "stage": "response_language_detection",
                        "status": "REVIEW", "reason": "live_detector_disagrees_with_frozen_language",
                        "live_detected_language": source_language,
                        "frozen_language": frozen_entry["source_language"],
                    })
                    source_language = frozen_entry["source_language"]
                sentence_records = [
                    {"sentence_id": s["sentence_id"], "sentence_index": s["sentence_index"], "text_source": s["text_source"]}
                    for s in frozen_entry["sentences"]
                ]
            else:
                live_sentences, _merge_log = segment_source_sentences(raw_text, source_language, preprocessor)
                sentence_records = [
                    {"sentence_id": f"{response_id}::s{idx:03d}", "sentence_index": idx, "text_source": sent}
                    for idx, sent in enumerate(live_sentences)
                ]

            for sr in sentence_records:
                if sr["sentence_id"] in seen_sentence_ids:
                    fatal_errors.append({
                        "id": sr["sentence_id"], "stage": "sentence_id_generation",
                        "status": "FATAL", "reason": "duplicate_sentence_id",
                    })
                seen_sentence_ids.add(sr["sentence_id"])

            pending.append({
                "response_id": response_id,
                "interview_id": interview_id,
                "question_id": question_id,
                "source_language": source_language,
                "language_detection_method": method,
                "interview_primary_language": primary_lang,
                "original_text": raw_text,
                "translation_required": source_language == "ca",
                "sentences": sentence_records,
            })

    # --- Batched ca->es translation, one frozen source sentence per
    # translation unit (translate_frozen_sentences_batched) -- see that
    # function's docstring for why this replaces the old whole-response/
    # chunk design entirely for primary translation. ---
    segmented_for_translation = {
        r["response_id"]: {"source_language": r["source_language"], "sentences": r["sentences"]}
        for r in pending
    }
    sentence_translation_results: Dict[str, dict] = {}
    primary_cache_coverage: Optional[dict] = None
    if any(r["translation_required"] for r in pending):
        # Round 12 (external review): a cheap, read-only pre-flight check,
        # computed and printed BEFORE any NLLB call -- see
        # compute_primary_cache_coverage()'s docstring for exactly why
        # this exists. Always computed and printed when there's Catalan
        # work to do, regardless of require_warm_primary_cache, so the
        # numbers are visible either way; only the GATE (refusing to
        # proceed) is conditional on that flag.
        primary_cache_coverage = compute_primary_cache_coverage(pending, cache, translator)
        print_cache_coverage_report(primary_cache_coverage)
        if require_warm_primary_cache and not primary_cache_coverage["safe_to_reuse_primary_cache"]:
            translation_aborted_early = True
            abort_reason = (
                f"primary cache coverage check failed: require_warm_primary_cache=True but "
                f"{primary_cache_coverage['missing']} of "
                f"{primary_cache_coverage['expected_catalan_sentence_units']} Catalan sentence "
                "units are missing from the primary cache under the current GENERATION_VERSION "
                f"({GENERATION_VERSION}). Refusing to silently start translating them. Either "
                "restore the expected warm cache at the configured cache path before re-running, "
                "or omit require_warm_primary_cache / --require-warm-cache if this is genuinely "
                "meant to be a first-time (or partial) translation run."
            )
            logger.error(f"[process] {abort_reason}")
            # Every ca sentence stays unattempted -- filled in as FAILED
            # below via sentence_translation_results.get(..., None). No
            # NLLB call was made.
        else:
            try:
                sentence_translation_results = translate_frozen_sentences_batched(
                    segmented_for_translation, translator, cache, batch_failure_tracker,
                    show_progress=show_progress,
                )
            except NLLBTranslationError as e:
                translation_aborted_early = True
                abort_reason = str(e)
                logger.error(f"[process] {e}")
                # Every ca sentence stays unattempted -- filled in as
                # FAILED below via sentence_translation_results.get(...,
                # None).

    # --- Pass 2: assemble each response's sentences_out entries directly
    # from the per-sentence translation results (NOT by re-splitting
    # translated Spanish text -- that target-side step no longer exists),
    # apply sentence-level quality diagnostics to decide FLAGGED, rebuild
    # each response's text_es from its OWN sentences in order, and run the
    # existing response-level quality diagnostics on that reconstruction
    # for corpus-level summaries. ---
    for r in pending:
        response_id = r["response_id"]
        interview_id = r["interview_id"]
        question_id = r["question_id"]
        translation_required = r["translation_required"]

        response_sentence_ids: List[str] = []
        sentence_text_es_parts: List[str] = []
        sentence_status_counts: Dict[str, int] = {}
        sentence_provenance_counts: Dict[str, int] = {}
        failed_sentence_ids: List[str] = []
        flagged_sentence_ids: List[str] = []

        for sr in r["sentences"]:
            sentence_id = sr["sentence_id"]
            response_sentence_ids.append(sentence_id)

            translation = sentence_translation_results.get(sentence_id)
            if translation is None:
                # Not attempted at all -- either this response doesn't
                # need translation (identity-copy path never populates
                # sentence_translation_results, handled just below) or the
                # whole run aborted before reaching this sentence.
                if translation_required:
                    text_es_sent: Optional[str] = None
                    provenance = "FAILED"
                else:
                    text_es_sent = sr["text_source"]
                    provenance = "SOURCE_ES"
            else:
                text_es_sent = translation["text_es"]
                provenance = translation["translation_provenance"]

            sentence_quality: Dict[str, Optional[dict]] = {
                "target_language_check": None, "degenerate_output": None, "length_diagnostics": None,
                "repetition_fallback_retry": None,
            }
            sentence_status = provenance
            if text_es_sent is not None and provenance != "SOURCE_ES":
                original_primary_degenerate_output = check_degenerate_output(sr["text_source"], text_es_sent)
                final_degenerate_output = original_primary_degenerate_output

                # Round 12: a PRIMARY translation that is itself a genuine
                # repetition loop gets exactly one targeted, cache-aware
                # retry under RETRY_GENERATION_PARAMS -- never speculative,
                # only after the primary result is independently confirmed
                # degenerate right here. If the retry resolves it, its
                # output REPLACES text_es_sent for every use below (the
                # response reconstruction, the sentence record, and all
                # remaining quality diagnostics in this block are computed
                # on the FINAL text, never the discarded primary one) and
                # the ORIGINAL primary diagnostic is preserved under
                # repetition_fallback_retry for provenance/audit. If the
                # retry does NOT resolve it (or itself fails), text_es_sent
                # is left exactly as the primary translation produced it,
                # and the sentence correctly stays FLAGGED on that basis --
                # see attempt_repetition_fallback_retry()'s docstring.
                if original_primary_degenerate_output["repetition_flag"]:
                    # Fix from external review: `provenance` (CACHE_HIT /
                    # FRESH_OK / RETRY_OK[pass-1 transient-retry] / etc.)
                    # describes the PRIMARY translation attempt only, and
                    # was previously left unchanged even when the retry
                    # below succeeded and text_es_sent was replaced --
                    # so a repaired sentence could be written to disk as
                    # e.g. "translation_provenance": "CACHE_HIT" even
                    # though its actual final text came from the
                    # anti-repetition retry, and every corpus-wide
                    # provenance rollup (by_translation_provenance,
                    # retried_ok_translations) inherited the same lie.
                    # primary_provenance_for_audit preserves the ORIGINAL
                    # value for the record; `provenance` itself is
                    # reassigned below, on resolution, to a NEW, distinct
                    # value -- deliberately NOT "RETRY_OK", which already
                    # means something different in this codebase (a
                    # PRIMARY-pass piece that needed a transient-failure
                    # retry inside translate_texts_batched, still entirely
                    # under GENERATION_PARAMS -- see _on_retry_piece_done
                    # above). Reusing that label here would silently merge
                    # two unrelated kinds of retry under one name, exactly
                    # the ambiguity this project has been eliminating
                    # round over round.
                    primary_provenance_for_audit = provenance
                    retry_result = attempt_repetition_fallback_retry(sr["text_source"], translator, cache)
                    retry_text = retry_result["text_es"]
                    retry_degenerate_output = (
                        check_degenerate_output(sr["text_source"], retry_text) if retry_text is not None else None
                    )
                    resolved = retry_degenerate_output is not None and not retry_degenerate_output["flagged"]
                    if resolved:
                        text_es_sent = retry_text
                        final_degenerate_output = retry_degenerate_output
                        provenance = "REPETITION_REPAIRED"
                        sentence_status = provenance
                        repetition_fallback_retry_resolved_ids.append(sentence_id)
                    else:
                        repetition_fallback_retry_unresolved_ids.append(sentence_id)
                    sentence_quality["repetition_fallback_retry"] = {
                        "attempted": True,
                        "reason": "pathological_repetition",
                        "cache_status": retry_result["cache_status"],
                        "primary_generation_version": GENERATION_VERSION,
                        "retry_generation_version": RETRY_GENERATION_VERSION,
                        "primary_degenerate_output": original_primary_degenerate_output,
                        "resolved": resolved,
                        "selected_generation_version": RETRY_GENERATION_VERSION if resolved else GENERATION_VERSION,
                        # New fields from external review: the two-tier
                        # provenance the top-level translation_provenance
                        # field alone can't carry (it must stay one flat
                        # value per sentence). primary_translation_
                        # provenance is what the PRIMARY pass alone would
                        # have recorded; selected_translation_provenance
                        # mirrors whatever sentences_out[...]
                        # ["translation_provenance"] actually ends up as
                        # for this sentence (either unchanged from primary,
                        # when unresolved, or "REPETITION_REPAIRED").
                        "primary_translation_provenance": primary_provenance_for_audit,
                        "selected_translation_provenance": provenance,
                    }

                sentence_quality["target_language_check"] = check_target_language(text_es_sent, "es")
                sentence_quality["degenerate_output"] = final_degenerate_output
                # Per-sentence length ratio (added after an external review
                # noted response-level length/similarity diagnostics alone
                # can't catch one sentence losing most of its content while
                # the rest of the response statistically compensates --
                # e.g. an 8-word source sentence collapsing to 1 word would
                # barely move a whole response's aggregate ratio). This is
                # diagnostic/REVIEW-only, deliberately NOT part of the
                # `flagged` fatal-quality condition below -- source<->target
                # length legitimately varies a lot sentence-to-sentence
                # (a short acknowledgement, an elided clause), so an
                # arbitrary per-sentence threshold would produce false
                # positives the same way an early response-level length
                # gate would have. See sentence_length_ratio_outlier_ids
                # in the corpus-wide rollup below.
                sentence_quality["length_diagnostics"] = compute_length_diagnostics(sr["text_source"], text_es_sent)
                if sentence_quality["length_diagnostics"]["length_ratio_outlier"]:
                    sentence_length_ratio_outlier_ids.append(sentence_id)
                # Round 11: target_language_check is diagnostic/REVIEW-only
                # here too, tracked separately below (sentence_language_
                # mismatch_ids) rather than folded into `flagged` -- see
                # check_target_language()'s docstring for why. degenerate_
                # output (empty / long-identical / repetition) is now the
                # ONLY sentence-level FATAL signal.
                if sentence_quality["target_language_check"]["passed"] is False:
                    sentence_language_mismatch_ids.append(sentence_id)
                flagged = sentence_quality["degenerate_output"]["flagged"]
                if flagged:
                    sentence_status = "FLAGGED"
                    flagged_sentence_ids.append(sentence_id)

            if sentence_status == "FAILED":
                failed_sentence_ids.append(sentence_id)
            else:
                if text_es_sent:
                    sentence_text_es_parts.append(text_es_sent)

            sentence_status_counts[sentence_status] = sentence_status_counts.get(sentence_status, 0) + 1
            sentence_provenance_counts[provenance] = sentence_provenance_counts.get(provenance, 0) + 1

            sentences_out[sentence_id] = {
                "sentence_id": sentence_id,
                "response_id": response_id,
                "interview_id": interview_id,
                "question_id": question_id,
                "sentence_index": sr["sentence_index"],
                "text_source": sr["text_source"],
                "text_es": text_es_sent,
                "sentence_status": sentence_status,
                "translation_provenance": provenance,
                "quality": sentence_quality,
                "source_language": r["source_language"],
                "translation_required": translation_required,
            }

        total_sentence_count = len(r["sentences"])
        response_translation_complete = len(failed_sentence_ids) == 0
        # Response-level text_es_status is an AGGREGATE derived from its
        # sentences (per the redesign's completeness requirement), using
        # translation_provenance (the mechanical outcome) rather than
        # sentence_status (which collapses to FLAGGED once a quality
        # diagnostic fires) -- a response where every sentence came back
        # FLAGGED-but-translated is still "TRANSLATED", not treated as
        # some fourth thing, exactly as a response with zero flagged
        # sentences would be.
        if not translation_required:
            text_es_status = "IDENTITY_COPY"
        elif total_sentence_count == 0 or len(failed_sentence_ids) == total_sentence_count:
            # No sentences at all, or every single one failed -- no usable
            # text_es for this response (matches the old code's meaning of
            # WARNING_TRANSLATION_FAILED: nothing to reconstruct).
            text_es_status = "WARNING_TRANSLATION_FAILED"
        elif failed_sentence_ids:
            # Some, but not all, sentences failed -- text_es IS still
            # reconstructed from whatever succeeded (see below): a failed
            # sentence no longer erases the rest of its response.
            text_es_status = "PARTIAL_TRANSLATION_FAILURE"
        elif sentence_provenance_counts.get("CACHE_HIT", 0) == total_sentence_count:
            # Every sentence was a pure cache hit -- nothing fresh at all.
            text_es_status = "CACHE_HIT"
        else:
            text_es_status = "TRANSLATED"

        text_es = " ".join(sentence_text_es_parts) if sentence_text_es_parts else None

        r["text_es"] = text_es
        r["text_es_status"] = text_es_status
        r["sentence_ids"] = response_sentence_ids
        r["sentence_status_counts"] = sentence_status_counts
        r["failed_sentence_ids"] = failed_sentence_ids
        r["flagged_sentence_ids"] = flagged_sentence_ids
        r["response_translation_complete"] = response_translation_complete

    # --- Pass 3: per-response quality diagnostics (on the reconstructed
    # text_es) and topic-text prep -- unchanged in spirit from before,
    # just no longer the place sentence IDs are assigned. ---
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

        if text_es is None:
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

        # Sentence IDs, statuses, and sentences_out entries were already
        # assembled in Pass 2, directly from the frozen source sentences
        # and their translation results -- there is no re-splitting of
        # text_es here any more (see this function's docstring: no
        # target-side re-splitting is allowed once sentence IDs are
        # frozen). This is just pulling those already-computed values onto
        # the response record.
        sentence_hard_split_count = sum(
            sentence_translation_results.get(sid, {}).get("hard_split_count", 0) for sid in r["sentence_ids"]
        )

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
            "sentence_ids": r["sentence_ids"],
            "sentence_count": len(r["sentence_ids"]),
            "sentence_status_counts": r["sentence_status_counts"],
            "failed_sentence_ids": r["failed_sentence_ids"],
            "flagged_sentence_ids": r["flagged_sentence_ids"],
            "response_translation_complete": r["response_translation_complete"],
            "sentence_hard_split_count": sentence_hard_split_count,
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
    # The validation target the Round 7 redesign explicitly asks for: the
    # frozen source sentence count must equal the number of downstream
    # sentence records produced, exactly -- not "produced approximately
    # that many Spanish sentences by re-splitting Spanish text" (the old,
    # now-removed target-side check this replaces). expected_sentence_
    # count is the sum of each pending (non-excluded) response's OWN
    # frozen sentence list length -- every one of those sentence IDs must
    # show up in sentences_out exactly once, whether its translation
    # succeeded, was flagged, or failed (a FAILED sentence still gets a
    # sentences_out record -- see Pass 2 -- it just carries text_es=None).
    expected_sentence_count = sum(len(r["sentences"]) for r in pending)
    # The actual validation target is NOT just cardinality (expected count
    # == actual count) -- two numbers matching by coincidence would still
    # pass a pure count check even if, say, one frozen sentence ID were
    # silently dropped from sentences_out while a duplicate of another one
    # were produced. The real invariant this redesign promises is EXACT
    # ID-SET equality: every frozen sentence ID that went into this run
    # produces exactly one downstream sentence record, and no downstream
    # record exists for anything else. missing_sentence_ids/
    # unexpected_sentence_ids make a mismatch immediately actionable
    # (which specific IDs, not just "off by N") rather than only visible as
    # a bare count discrepancy.
    expected_sentence_ids = {s["sentence_id"] for r in pending for s in r["sentences"]}
    actual_sentence_ids = set(sentences_out.keys())
    missing_sentence_ids = sorted(expected_sentence_ids - actual_sentence_ids)
    unexpected_sentence_ids = sorted(actual_sentence_ids - expected_sentence_ids)
    sentence_id_set_match = not missing_sentence_ids and not unexpected_sentence_ids
    sentence_count_invariant = {
        "expected": expected_sentence_count,
        "actual": len(sentences_out),
        "match": expected_sentence_count == len(sentences_out) and sentence_id_set_match,
        "sentence_id_set_match": sentence_id_set_match,
        "missing_sentence_id_count": len(missing_sentence_ids),
        "missing_sentence_ids": missing_sentence_ids[:20],
        "unexpected_sentence_id_count": len(unexpected_sentence_ids),
        "unexpected_sentence_ids": unexpected_sentence_ids[:20],
    }
    silent_loss = (
        accounted_for != input_response_count
        or bool(unmapped_sentences)
        or bool(fatal_errors)
        or not sentence_count_invariant["match"]
    )

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
    # New in the Round 7 redesign: a response where SOME but not all of
    # its sentences failed -- distinct from translation_failed_count
    # (every sentence failed, no usable text_es at all). Per-sentence
    # atomicity means this response still has a (partial) text_es and a
    # full sentence-level record of exactly which sentence(s) failed (see
    # responses_out[rid]["failed_sentence_ids"]) -- it is not silently
    # folded into either the successful or the fully-failed bucket.
    translation_partial_failure_count = sum(
        1 for rid in translation_required_ids if responses_out[rid]["text_es_status"] == "PARTIAL_TRANSLATION_FAILURE"
    )
    # Response-level fresh-vs-cached split of translation_successful_count.
    # Kept for reference (a response where every sentence was a cache hit
    # vs. one where at least one sentence was freshly translated), but the
    # runtime-info reporting below now uses the SENTENCE-level totals
    # (sentence_translation_summary) instead, since the sentence is the
    # actual live-progress-bar unit in the Round 7 redesign -- see
    # translate_frozen_sentences_batched.
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

    # NOTE: deliberately read `load_attempted` here, not `available` --
    # `available` is the lazy-load trigger itself, and this is end-of-run
    # report assembly, not a real use site. A corpus with no similarity
    # pairs to score never reaches the `.available` check at the actual use
    # sites above, so it must not be forced to load the model here just to
    # fill in this summary field (that would silently defeat the laziness
    # fix for exactly the runs it matters most for -- ones with nothing to
    # score).
    _similarity_scorer_used = similarity_scorer is not None and similarity_scorer.load_attempted
    translation_quality_summary = {
        "similarity_model_available": similarity_scorer.available if _similarity_scorer_used else False,
        "similarity_model_unavailable_reason": (
            similarity_scorer.unavailable_reason if _similarity_scorer_used
            else (
                "similarity_scorer_not_provided" if similarity_scorer is None
                else "not_needed_for_this_run_no_similarity_pairs_computed"
            )
        ),
        "language_mismatch_count": len(language_mismatch_ids),
        "language_mismatch_ids": language_mismatch_ids,
        # Round 13 (external review): a PRE-translation, INPUT-side signal
        # -- the live per-response language detector disagreed with what
        # the frozen segmentation file recorded for this response's source
        # language. Deliberately named distinctly from language_mismatch_
        # count above, which is the OUTPUT-side check (translated text vs.
        # expected target language, post-translation) -- these are
        # different checks at different pipeline stages and must never be
        # conflated. Diagnostic/REVIEW-only: see the frozen-entry lookup in
        # Pass 1 for why the frozen language is authoritative and this
        # never discards sentences or changes translation_required.
        "source_language_mismatch_count": len(source_language_mismatch_ids),
        "source_language_mismatch_ids": source_language_mismatch_ids,
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

    # Corpus-wide sentence-level rollup (Round 7 redesign) -- the sentence
    # is the actual translation unit now, so this is what the live
    # progress bar during a run and this report's runtime-info fields
    # should agree with, the same way the old response-level fresh/cache_
    # hit/failed counts were kept in agreement with translate_responses_
    # batched's response-level progress bar (fifth/sixth external review).
    sentence_status_totals: Dict[str, int] = {}
    sentence_provenance_totals: Dict[str, int] = {}
    sentence_hard_split_total = 0
    for s in sentences_out.values():
        sentence_status_totals[s["sentence_status"]] = sentence_status_totals.get(s["sentence_status"], 0) + 1
        sentence_provenance_totals[s["translation_provenance"]] = sentence_provenance_totals.get(s["translation_provenance"], 0) + 1
    for resp in responses_out.values():
        sentence_hard_split_total += resp.get("sentence_hard_split_count", 0)
    flagged_sentence_ids_corpus = [sid for sid, s in sentences_out.items() if s["sentence_status"] == "FLAGGED"]
    sentence_translation_summary = {
        "total_sentences": len(sentences_out),
        "by_sentence_status": sentence_status_totals,
        "by_translation_provenance": sentence_provenance_totals,
        "hard_split_sentence_pieces": sentence_hard_split_total,
        "responses_with_partial_translation_failure_count": translation_partial_failure_count,
        "responses_with_partial_translation_failure_ids": [
            rid for rid, resp in responses_out.items() if resp.get("text_es_status") == "PARTIAL_TRANSLATION_FAILURE"
        ],
        # flagged_sentence_count/ids is the corpus-wide view of exactly the
        # signal translation_output_validity now gates on below (sentence_
        # status_totals.get("FLAGGED", 0) is the same number, this is just
        # named for what it actually means at the call site) -- also
        # available per-response as responses_out[rid]["flagged_sentence_ids"].
        "flagged_sentence_count": sentence_status_totals.get("FLAGGED", 0),
        "flagged_sentence_ids": flagged_sentence_ids_corpus,
        # Per-sentence length-ratio outliers (diagnostic/REVIEW-only -- see
        # the length_diagnostics comment in Pass 2 above for why this is
        # deliberately not part of the FLAGGED/fatal-gate condition).
        "sentence_length_ratio_outlier_count": len(sentence_length_ratio_outlier_ids),
        "sentence_length_ratio_outlier_ids": sentence_length_ratio_outlier_ids,
        # Per-sentence target-language mismatches (Round 11, diagnostic/
        # REVIEW-only -- see check_target_language()'s docstring for why
        # this is deliberately not part of the FLAGGED/fatal-gate
        # condition above, even though it was through Round 10).
        "sentence_language_mismatch_count": len(sentence_language_mismatch_ids),
        "sentence_language_mismatch_ids": sentence_language_mismatch_ids,
        # Round 12: how many repetition-flagged PRIMARY translations got a
        # targeted fallback retry, and how it went. "attempted" is the sum
        # of resolved+unresolved (every sentence whose primary result
        # tripped repetition_flag gets exactly one retry attempt).
        # "resolved" sentences are no longer FLAGGED (their retry output
        # replaced the primary text_es); "unresolved" sentences kept their
        # original primary text_es and correctly remain FLAGGED -- see
        # attempt_repetition_fallback_retry() and the repetition_fallback_
        # retry field on each affected sentence (in sentences_out) for the
        # full per-sentence detail (cache_status, both generation
        # versions, the original primary diagnostic).
        "repetition_fallback_retry_attempted_count": (
            len(repetition_fallback_retry_resolved_ids) + len(repetition_fallback_retry_unresolved_ids)
        ),
        "repetition_fallback_retry_resolved_count": len(repetition_fallback_retry_resolved_ids),
        "repetition_fallback_retry_resolved_ids": repetition_fallback_retry_resolved_ids,
        "repetition_fallback_retry_unresolved_count": len(repetition_fallback_retry_unresolved_ids),
        "repetition_fallback_retry_unresolved_ids": repetition_fallback_retry_unresolved_ids,
    }

    structural_validity = not silent_loss
    translation_completeness_validity = (
        not translation_aborted_early
        and translation_failed_count == 0
        and translation_partial_failure_count == 0
        and translation_successful_count == translation_required_count
        and missing_text_es_count == 0
        and missing_topic_text_es_raw_count == 0
    )
    # Translation-output-QUALITY validity is gated at SENTENCE grain, not
    # response grain (patched after an external review of the Round 7/
    # step 6 redesign found the response-level gate below was a real hole:
    # check_degenerate_output() already runs once per sentence in Pass 2,
    # correctly flagging a genuinely degenerate sentence as sentence_
    # status="FLAGGED" -- but nothing downstream actually GATED on that. A
    # response's own response-level checks run separately, on the JOIN of
    # ALL its sentences' text, so a single bad sentence sitting among
    # several good ones could easily still pass at response grain even
    # though the sentence-level check had already correctly caught it.
    # STEP2_VALID could end up True with a frozen analytical sentence
    # explicitly marked bad -- exactly the failure mode the one-sentence-
    # per-translation architecture exists to make detectable, going
    # undetected by the gate itself.
    #
    # The fix: the fatal gate is "no sentence anywhere in the corpus was
    # flagged degenerate" -- the per-UNIT signal this architecture is
    # built around, at the same grain as the frozen sentence IDs
    # themselves. sentence_status="FLAGGED" is set exactly when a
    # sentence's own check_degenerate_output fails (see Pass 2) --
    # nothing sentence-level is excluded from this gate the way length-
    # ratio/similarity are excluded below.
    #
    # Round 11: check_target_language's result was ALSO part of this gate
    # through Round 10 (a sentence with passed=False was FLAGGED on that
    # alone). Removed after a manual review of the first real 950-
    # response corpus run's 141 language-mismatch flags found every one
    # was either a langdetect false positive on genuinely correct Spanish
    # (105) or already independently, correctly caught by check_
    # degenerate_output (36) -- language detection alone never
    # contributed a real catch check_degenerate_output didn't already
    # make. It remains fully computed and reported at both response grain
    # (language_mismatch_count/ids below) and sentence grain (sentence_
    # language_mismatch_count/ids in sentence_translation_summary, added
    # this round) -- still worth a human's attention -- but is REVIEW-
    # tier only now, never FAIL, at any length. See STEP2_NLLB_CHANGES.md's
    # Round 11 section for the full breakdown.
    sentence_flagged_count = sentence_translation_summary["flagged_sentence_count"]
    translation_output_validity = sentence_flagged_count == 0

    # The RESPONSE-level equivalents (check_target_language/check_
    # degenerate_output run once more, on the reconstructed whole-response
    # text_es) are deliberately NOT part of the fatal gate any more. This
    # is what the sixth external review's "many of the 25 degenerate flags
    # were ordinary repeated domain phrasing, not NLLB collapse" finding
    # was already pointing at: several individually-correct sentences that
    # happen to repeat a common phrase (routine in transcribed interview
    # speech) can still look repetitive once joined into one response-
    # length block, even though NOT ONE of those sentences was itself
    # flagged. Gating STEP2_VALID on that recreates exactly the false-
    # positive problem the sixth review already fixed once, now at
    # response instead of chunk grain. They stay fully computed and
    # reported (language_mismatch_count/degenerate_output_count below,
    # still worth a human's attention) but move to the REVIEW tier,
    # alongside length-ratio/similarity, rather than FAIL.
    response_level_language_mismatch_count = translation_quality_summary["language_mismatch_count"]
    response_level_degenerate_output_count = translation_quality_summary["degenerate_output_count"]
    length_ratio_outlier_count = translation_quality_summary["length_ratio_outlier_count"]
    low_similarity_count = translation_quality_summary["semantic_preservation"]["count_below_informational_bound"]
    short_text_language_not_assessed_count = translation_quality_summary["short_text_language_not_assessed_count"]
    short_text_identical_output_count = translation_quality_summary["short_text_identical_output_count"]
    sentence_length_ratio_outlier_count = sentence_translation_summary.get("sentence_length_ratio_outlier_count", 0)
    # Round 11: the sentence-grain mirror of response_level_language_
    # mismatch_count -- a sentence whose OWN target-language check failed
    # is no longer fatal (see above), but should still surface at REVIEW
    # tier just like its response-level equivalent already did.
    sentence_language_mismatch_count = sentence_translation_summary.get("sentence_language_mismatch_count", 0)
    # Round 13: the frozen-vs-live SOURCE-language disagreement, read back
    # off translation_quality_summary (already reflects source_language_
    # mismatch_ids computed in Pass 1) rather than the raw list directly,
    # so this stays in the same "read the summary dict" style as every
    # other REVIEW-tier signal below.
    source_language_mismatch_count = translation_quality_summary.get("source_language_mismatch_count", 0)
    if not translation_output_validity:
        # A genuinely serious failure occurred at sentence grain -- this is
        # fatal (see translation_output_validity above), so "REVIEW" (which
        # implies "otherwise fine, just take a look") would understate it.
        translation_sanity_status = "FAIL"
    elif (
        length_ratio_outlier_count > 0
        or low_similarity_count > 0
        or short_text_language_not_assessed_count > 0
        or short_text_identical_output_count > 0
        or response_level_language_mismatch_count > 0
        or response_level_degenerate_output_count > 0
        or sentence_length_ratio_outlier_count > 0
        or sentence_language_mismatch_count > 0
        or source_language_mismatch_count > 0
    ):
        # Diagnostic-only signals: worth a human review pass, but never a
        # reason to block STEP2_VALID on their own. The short-text signals
        # (added after the second external review) are exactly the "Sí." /
        # "No." / acronym / number case: too little text to run langdetect
        # meaningfully on, or a source==target translation that is probably
        # correct rather than degenerate -- surfaced for a human glance,
        # never fatal. Sentence-level length-ratio outliers (added
        # alongside the fatal-gate fix above) are the same kind of signal,
        # just measured per sentence instead of per response -- see Pass 2.
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
        # Round 12 (external review): None when there was no Catalan
        # translation work at all this run (translation_required was
        # False for every response); otherwise compute_primary_cache_
        # coverage()'s result, for the JSON audit trail -- see that
        # function's docstring and require_warm_primary_cache above.
        "primary_cache_coverage": primary_cache_coverage,
        "silent_loss_check": {
            "input_response_count": input_response_count,
            "accounted_for_response_count": accounted_for,
            "unmapped_sentence_count": len(unmapped_sentences),
            "fatal_error_count": len(fatal_errors),
            "sentence_count_invariant": sentence_count_invariant,
            "silent_loss_detected": silent_loss,
        },
        # The Round 7 redesign's validation target, stated directly (not
        # just buried inside silent_loss_check): every frozen source
        # sentence ID produced exactly one downstream sentence record.
        "sentence_count_invariant": sentence_count_invariant,
        "translation_aborted_early": translation_aborted_early,
        "abort_reason": abort_reason,
        "translation_required_count": translation_required_count,
        "translation_successful_count": translation_successful_count,
        "translation_failed_count": translation_failed_count,
        "translation_partial_failure_count": translation_partial_failure_count,
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
        "sentence_translation_summary": sentence_translation_summary,
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
        # cache_hits is the raw cache.get() hit count (unchanged meaning);
        # the four *_translations fields below are SENTENCE-level now (see
        # sentence_translation_summary above) -- the sentence is the
        # actual translation unit and live-progress-bar unit in the
        # Round 7 redesign, so these agree with what a person watched
        # scroll by during the run, not a response-level rollup over what
        # used to be multi-sentence chunks.
        "cache_hits": cache.stats()["hits_this_run"],
        "fresh_translations": sentence_provenance_totals.get("FRESH_OK", 0),
        "retried_ok_translations": sentence_provenance_totals.get("RETRY_OK", 0),
        # Round 12 (external review, provenance fix): sentences whose
        # FINAL text_es came from a resolved anti-repetition fallback
        # retry, distinct from retried_ok_translations above (which is
        # the PRIMARY pass's own transient-failure retry, unrelated --
        # see the comment at the Pass 2 provenance fix for why these are
        # deliberately never merged under one label).
        "repetition_repaired_translations": sentence_provenance_totals.get("REPETITION_REPAIRED", 0),
        "failed_translations": sentence_provenance_totals.get("FAILED", 0),
        "translation_chunks": sentence_hard_split_total,
        "runtime_info": {
            "device": translator.device,
            "batch_size": translator.batch_size,
            "runtime_architecture": runtime_arch_info["runtime_architecture"],
            "running_under_rosetta": runtime_arch_info["running_under_rosetta"],
            "rosetta_warning": runtime_arch_info["warning"],
            "elapsed_seconds": round(time.time() - process_start_time, 3),
            "cache_hits": cache.stats()["hits_this_run"],
            "cache_hit_translations": sentence_provenance_totals.get("CACHE_HIT", 0),
            "fresh_translations": sentence_provenance_totals.get("FRESH_OK", 0),
            "retried_ok_translations": sentence_provenance_totals.get("RETRY_OK", 0),
            "repetition_repaired_translations": sentence_provenance_totals.get("REPETITION_REPAIRED", 0),
            "failed_translations": sentence_provenance_totals.get("FAILED", 0),
            "translation_chunks": sentence_hard_split_total,
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
        # Stored so a caller running a full corpus run right after preflight
        # (see main()) can reuse THIS already-loaded translator instead of
        # constructing (and loading the ~600M-parameter model) a second
        # time -- only set on success; a caller must never try to reuse a
        # translator from a failed/partial preflight.
        result["translator"] = translator
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
    pcc = report.get("primary_cache_coverage")
    if pcc is not None:
        print("Primary cache coverage (before this run's translation pass):")
        print(f"  expected Catalan sentence units    {pcc['expected_catalan_sentence_units']}")
        print(f"  covered                            {pcc['covered']}")
        print(f"  missing                            {pcc['missing']}")
        print(f"  SAFE TO REUSE PRIMARY CACHE         {pcc['safe_to_reuse_primary_cache']}")
        print()
    print(f"Catalan responses requiring NLLB      {report['translation_required_count']}")
    print(f"Successful ca->es translations        {report['translation_successful_count']}")
    print(f"Partially failed (some sentences)     {report.get('translation_partial_failure_count', 0)}")
    print(f"Failed translations                   {report['translation_failed_count']}")
    print()
    print(f"Missing text_es                       {report['missing_text_es_count']}")
    print(f"Missing topic_text_es_raw             {report['missing_topic_text_es_raw_count']}")
    print()
    print(f"topic_text_es_clean empty (excluded)  {report['topic_text_es_clean_empty_count']}")
    print(f"Expected topic-model documents         {report['expected_topic_model_document_count']}")
    print()
    sci = report.get("sentence_count_invariant", {})
    print(f"Frozen sentence IDs expected           {sci.get('expected', 'unknown')}")
    print(f"Downstream sentence records produced   {sci.get('actual', 'unknown')}")
    print(f"Sentence-count invariant holds         {sci.get('match', 'unknown')}")
    print(f"  (exact frozen-ID-set match)           {sci.get('sentence_id_set_match', 'unknown')}")
    if not sci.get("sentence_id_set_match", True):
        print(f"  missing sentence IDs (frozen, not produced): {sci.get('missing_sentence_id_count', 0)}")
        print(f"  unexpected sentence IDs (produced, not frozen): {sci.get('unexpected_sentence_id_count', 0)}")
    sts = report.get("sentence_translation_summary", {})
    by_status = sts.get("by_sentence_status", {})
    print(f"Sentences: source_es={by_status.get('SOURCE_ES', 0)} cache_hit={by_status.get('CACHE_HIT', 0)} "
          f"fresh={by_status.get('FRESH_OK', 0)} retry_ok={by_status.get('RETRY_OK', 0)} "
          f"repetition_repaired={by_status.get('REPETITION_REPAIRED', 0)} "
          f"flagged={by_status.get('FLAGGED', 0)} failed={by_status.get('FAILED', 0)}")
    print(f"Hard-split sentence pieces (oversized sentence) {sts.get('hard_split_sentence_pieces', 0)}")
    print()
    print(f"FLAGGED sentences (fatal -- gates STEP2_VALID)  {sts.get('flagged_sentence_count', 0)}")
    print(f"Sentence length-ratio outliers (review-only)    {sts.get('sentence_length_ratio_outlier_count', 0)}")
    print(f"Sentence language mismatches (review-only)      {sts.get('sentence_language_mismatch_count', 0)}")
    print(f"Repetition fallback retries attempted           {sts.get('repetition_fallback_retry_attempted_count', 0)}")
    print(f"  resolved (no longer flagged)                  {sts.get('repetition_fallback_retry_resolved_count', 0)}")
    print(f"  unresolved (still FLAGGED)                     {sts.get('repetition_fallback_retry_unresolved_count', 0)}")
    print()
    print("Response-level diagnostics (review-only -- see FLAGGED sentences above for the fatal gate):")
    print(f"  Source-language mismatches (frozen vs. live, Round 13) {tqs.get('source_language_mismatch_count', 0)}")
    print(f"  Language mismatches                 {tqs['language_mismatch_count']}")
    print(f"    (short text, not assessed)         {tqs['short_text_language_not_assessed_count']}")
    print(f"  Degenerate outputs                  {tqs['degenerate_output_count']}")
    print(f"    (short text, identical -- informational) {tqs['short_text_identical_output_count']}")
    print(f"  Length-ratio outliers               {tqs['length_ratio_outlier_count']}")
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
    print(f"Retried-OK translations               {report.get('retried_ok_translations', 'unknown')}")
    print(f"Repetition-repaired translations      {report.get('repetition_repaired_translations', 'unknown')}")
    print(f"Cache hits                            {report.get('cache_hits', 'unknown')}")
    print(f"Elapsed                               {_format_hms(report.get('elapsed_seconds', 0))}")
    runtime_info = report.get("runtime_info") or {}
    if runtime_info.get("rosetta_warning"):
        print(f"\nWARNING: {runtime_info['rosetta_warning']}")
    if report.get("translation_aborted_early"):
        print(f"\nRun aborted early: {report.get('abort_reason')}")


def validate_frozen_segmentation_against_input(
    frozen_file: dict,
    raw_data: Dict[str, Dict[str, str]],
    input_file_path: str,
) -> List[str]:
    """Hard pre-run check that a loaded frozen segmentation file is still
    the frozen file FOR this exact corpus, not a stale one left over from
    a previous version of interviews.json that happens to still parse and
    still have matching response IDs and languages. process()'s own
    per-response frozen lookup (missing_from_frozen_segmentation /
    frozen_segmentation_source_language_mismatch) cannot catch this on its
    own -- it only ever iterates the CURRENT corpus's responses, so a
    frozen response that quietly disappeared from the current input, or
    one whose original text quietly changed while its ID and detected
    language happened to stay the same, would never surface there.

    Checks, in order:
      1. schema_version is current -- an older frozen file predates
         input_sha256 entirely and cannot be verified at all.
      2. The frozen file's own declared response_count/total_sentence_count
         agree with what is actually inside it (self-consistency -- catches
         a hand-edited or corrupted frozen file).
      3. input_sha256, recomputed from the actual on-disk input file right
         now, matches what the frozen file recorded when it was generated
         -- the cheapest, strongest single check: a match already proves
         the input is byte-for-byte identical to what the freeze was built
         from.
      4. Response ID sets match exactly, both directions -- nothing in the
         current corpus is missing from the frozen file, AND nothing in
         the frozen file is stale (no longer present in the current
         corpus).
      5. Every response's original_text is identical between the frozen
         file and the current corpus.
    (4) and (5) are logically redundant with (3) whenever the whole-file
    hash matches -- they are kept anyway, both because they give a far
    more actionable error message when something DOES drift (which
    specific response, not just "the file changed somehow"), and as an
    independent way of catching the same class of problem.

    Returns a list of human-readable problem descriptions; empty means
    safe to proceed. Never raises -- always describes everything wrong,
    not just the first thing found.
    """
    problems: List[str] = []

    schema_version = frozen_file.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < FROZEN_SEGMENTATION_SCHEMA_VERSION:
        problems.append(
            f"frozen file schema_version is {schema_version!r}, expected >= "
            f"{FROZEN_SEGMENTATION_SCHEMA_VERSION} -- re-run freeze_sentence_segmentation.py "
            "to regenerate it with the current schema (an older file predates the "
            "input-hash safety check and cannot be verified at all)."
        )
        return problems  # nothing else below is trustworthy without a current schema

    frozen_responses = frozen_file.get("responses", {})
    declared_response_count = frozen_file.get("response_count")
    declared_sentence_count = frozen_file.get("total_sentence_count")
    actual_response_count = len(frozen_responses)
    actual_sentence_count = sum(len(r.get("sentences", [])) for r in frozen_responses.values())
    if declared_response_count != actual_response_count:
        problems.append(
            f"frozen file's declared response_count ({declared_response_count!r}) does not "
            f"match its actual number of response entries ({actual_response_count})."
        )
    if declared_sentence_count != actual_sentence_count:
        problems.append(
            f"frozen file's declared total_sentence_count ({declared_sentence_count!r}) does "
            f"not match the actual sentence count inside it ({actual_sentence_count})."
        )

    recorded_hash = frozen_file.get("input_sha256")
    current_hash = _hash_file(input_file_path)
    if not recorded_hash:
        problems.append("frozen file has no recorded input_sha256.")
    elif current_hash is None:
        problems.append(f"could not hash the current input file to verify against the frozen one: {input_file_path}")
    elif recorded_hash != current_hash:
        problems.append(
            f"input_sha256 mismatch -- {input_file_path} has changed since the frozen "
            f"segmentation was generated (frozen={recorded_hash}, current={current_hash}). "
            "Re-run freeze_sentence_segmentation.py against the current corpus before "
            "running Step 2 against it."
        )

    current_response_ids = set()
    current_original_text: Dict[str, str] = {}
    for interview_id, questions in raw_data.items():
        for question_id, raw_text in questions.items():
            if not raw_text or not raw_text.strip():
                continue
            response_id = f"{interview_id}::{question_id}"
            current_response_ids.add(response_id)
            current_original_text[response_id] = raw_text

    frozen_response_ids = set(frozen_responses.keys())
    missing_from_frozen = sorted(current_response_ids - frozen_response_ids)
    stale_in_frozen = sorted(frozen_response_ids - current_response_ids)
    if missing_from_frozen:
        problems.append(
            f"{len(missing_from_frozen)} response(s) in the current corpus are not in the "
            f"frozen segmentation at all (e.g. {missing_from_frozen[:5]})."
        )
    if stale_in_frozen:
        problems.append(
            f"{len(stale_in_frozen)} response(s) in the frozen segmentation no longer exist "
            f"in the current corpus (e.g. {stale_in_frozen[:5]}) -- this is exactly the case "
            "process()'s own per-response frozen lookup can never detect by itself, since it "
            "only ever iterates the CURRENT corpus's responses."
        )

    text_mismatches = sorted(
        rid for rid in (current_response_ids & frozen_response_ids)
        if frozen_responses[rid].get("original_text") != current_original_text[rid]
    )
    if text_mismatches:
        problems.append(
            f"{len(text_mismatches)} response(s) exist in both under the same ID, but their "
            f"text differs from what the frozen segmentation was built from (e.g. "
            f"{text_mismatches[:5]})."
        )

    return problems


def main(mode: str = "full", batch_size: Optional[int] = None, require_warm_cache: bool = False) -> bool:
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

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return False

    raw_data = JsonHandler.read_json(INPUT_FILE)
    if not raw_data:
        logger.error("No data loaded from input file")
        return False

    # The Round 7 redesign's frozen sentence structure is REQUIRED for a
    # real corpus run (not for --smoke-test, which uses its own small
    # synthetic data and computes segmentation live -- see run_smoke_test)
    # -- "frozen" means tied to this actual on-disk artifact, not silently
    # recomputed from whatever segment_source_sentences() happens to
    # produce on a given day. If this is missing, that's a setup problem
    # to fix (run freeze_sentence_segmentation.py), not something to work
    # around by falling back to live computation for a full run.
    if not os.path.exists(FROZEN_SEGMENTATION_PATH):
        logger.error(
            f"Frozen sentence segmentation not found: {FROZEN_SEGMENTATION_PATH}. "
            "Run `python freeze_sentence_segmentation.py` first -- see "
            "evidence/round7_full_corpus_run_2026-09-07/README.md."
        )
        return False
    frozen_file = JsonHandler.read_json(FROZEN_SEGMENTATION_PATH)
    frozen_segmentation = frozen_file["responses"]

    # Hard pre-run gate (checked BEFORE the expensive model load below, not
    # after): the frozen file must still be the frozen file FOR this exact
    # corpus, not a stale one that happens to still have matching response
    # IDs and languages. process()'s own per-response frozen lookup cannot
    # catch a response that quietly disappeared from the current input, or
    # one whose text quietly changed -- see validate_frozen_segmentation_
    # against_input()'s own docstring for the full list of what this
    # checks and why each check is there.
    frozen_validation_problems = validate_frozen_segmentation_against_input(frozen_file, raw_data, INPUT_FILE)
    if frozen_validation_problems:
        logger.error(
            "Frozen segmentation no longer matches the current input corpus -- refusing to "
            "run Step 2 against it. Re-run freeze_sentence_segmentation.py if this corpus "
            "change is intentional. Problems found:"
        )
        for problem in frozen_validation_problems:
            logger.error(f"  - {problem}")
        return False

    preprocessor = Preprocessor()
    # SENTENCE_CACHE_PATH, not CACHE_PATH -- the old whole-response/chunk
    # cache from the six-hour real run stays untouched and auditable (see
    # evidence/round7_full_corpus_run_2026-09-07/). Sentence-level
    # translations get their own cache file.
    cache = TranslationCache(SENTENCE_CACHE_PATH)

    # A single NLLBTranslator load, not two: run_preflight() below already
    # constructs one (a real tokenizer/model load, confirmed by an actual
    # translation call) purely to verify this environment can run NLLB at
    # all -- it used to be thrown away immediately afterward, and a SECOND
    # NLLBTranslator (a second full model load) was constructed here for
    # the actual corpus run. On a low-memory machine (the exact Mac that
    # previously crashed under memory pressure -- see get_runtime_
    # architecture_info()/resolve_default_batch_size()), loading a ~600M-
    # parameter model twice back-to-back before doing any real work is
    # wasteful and risky for no benefit: preflight already used the same
    # batch_size this run will use, so its translator is exactly the one
    # this run needs. run_preflight() now returns it in its result dict
    # (only on success) specifically so it can be reused here instead.
    preflight_passed, preflight_result = run_preflight(verbose=True, batch_size=batch_size)
    if not preflight_passed:
        logger.error(
            "Preflight failed -- aborting before processing the corpus. "
            "Run `python preprocess_v2.py --preflight` for full detail."
        )
        return False
    translator = preflight_result["translator"]
    similarity_scorer = SemanticSimilarityScorer()

    if require_warm_cache:
        logger.info(
            "--require-warm-cache is set: this run will refuse to proceed if the primary "
            "cache is missing any of this corpus's Catalan sentence units under the current "
            "GENERATION_VERSION, rather than silently re-translating them."
        )

    responses_out, sentences_out, report = process(
        raw_data, preprocessor, cache, translator, similarity_scorer,
        show_progress=True,
        frozen_segmentation=frozen_segmentation,
        require_warm_primary_cache=require_warm_cache,
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
    parser.add_argument(
        "--require-warm-cache", action="store_true",
        help=(
            "Round 12 safety gate: before translating anything, verify every Catalan "
            "sentence in the frozen corpus is already in the primary cache under the "
            "current GENERATION_VERSION, and refuse to proceed (no NLLB calls made) if "
            "any are missing, instead of silently re-translating them. Pass this on every "
            "run EXCEPT a genuinely first-ever run against an empty cache -- see "
            "STEP2_NLLB_CHANGES.md's 'Execution sequence for the next real run'. Ignored "
            "with --preflight/--smoke-test (neither calls process())."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.preflight:
        ok = main(mode="preflight", batch_size=args.batch_size)
    elif args.smoke_test:
        ok = main(mode="smoke-test", batch_size=args.batch_size)
    else:
        ok = main(mode="full", batch_size=args.batch_size, require_warm_cache=args.require_warm_cache)
    sys.exit(0 if ok else 1)
