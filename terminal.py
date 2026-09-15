import os
import nltk
import hashlib
from typing import Dict, List, Optional, TYPE_CHECKING
from json_handler import JsonHandler
from preprocessing import Preprocessor
import preprocess_v2
# NOTE: TopicModeler / SentimentAnalyzer / Visualizer are deliberately NOT
# imported at module level. Each drags in a heavy ML stack transitively
# (topic_modeling.py and visualization.py both import bertopic ->
# sentence_transformers -> transformers; sentiment_analysis.py imports
# transformers directly for BART/nlptown) -- so importing THIS module
# (which `python terminal.py` and this project's test suite both do)
# used to trigger all of that immediately, even for a session that only
# ever runs Step 2. On at least one real machine, that eager import chain
# reached an incompatible TensorFlow build and aborted the whole process
# before Step 2 ever got a chance to run (see ml_backend.py). Each is now
# imported lazily, inside its own _get_*() method below, only when the
# corresponding menu option is actually used -- matching the existing
# lazy pattern already used for the NLLB components and SentimentAnalyzer
# construction (see _get_nllb_translator / _get_analyzer).
#
# TYPE_CHECKING-guarded so return-type annotations below can still name
# these classes for readability/IDE support without importing them (or
# their heavy dependency chains) at actual runtime.
if TYPE_CHECKING:
    from sentiment_analysis import SentimentAnalyzer
    from visualization import Visualizer
import logging
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sidecar written alongside the Step 2 bridge files (fully_processed_ca.json /
# sentiment_input_ca.json / topic_modeling_input.json) so _analyze_topics()
# and _analyze_sentiment() can tell a bridge file that was actually produced
# by a STEP2_VALID=True run of THIS pipeline apart from one that merely
# exists on disk (a historical file left over from a previous pipeline
# version, or copied in from elsewhere). A bare os.path.exists() check on
# the bridge file alone cannot make that distinction -- see
# _step2_bridge_is_fresh() below and STEP2_NLLB_CHANGES.md.
STEP2_BRIDGE_METADATA_FILENAME = "step2_bridge_metadata.json"
# v2 (bumped after the second external review): the sidecar now also
# records a SHA-256 of data/input/interviews.json at the time Step 2 wrote
# it, and _step2_bridge_is_fresh() recomputes that hash and compares --
# closing a reproducibility gap where a sidecar from a genuinely valid
# Step 2 run could otherwise still read as "fresh" after the corpus itself
# was edited/replaced underneath it. Bumping the schema version means an
# old v1 sidecar (written before this hash existed) is correctly treated
# as NOT fresh -- there is no hash to check it against, so Step 2 simply
# runs again once, the same deliberate cache/version-break pattern already
# used for GENERATION_VERSION in preprocess_v2.py.
STEP2_BRIDGE_SCHEMA_VERSION = "nllb_step2_bridge_v2"

# The canonical Step 2 input file. Hashed at bridge-write time and
# re-hashed at freshness-check time (see _step2_bridge_is_fresh) so a
# corpus that changed after the last validated Step 2 run is detected,
# even though _create_processed_versions() may have been called against
# only a SUBSET of this file (via _select_questions()) -- the concern this
# guards against is the FILE on disk changing, not which subset of it was
# processed.
STEP2_INTERVIEWS_INPUT_FILE = "./data/input/interviews.json"


def _hash_file(path: str) -> Optional[str]:
    """SHA-256 of a file's bytes, or None if it can't be read (missing,
    permission error, etc.) -- callers must treat None as "hash unknown",
    never as a value that could coincidentally match a recorded hash.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class TerminalInterface:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.preprocessor = Preprocessor()
        # Lazy -- SentimentAnalyzer() loads the nlptown sentiment model AND
        # facebook/bart-large-mnli (the confusion signal) immediately in
        # its own __init__, and importing sentiment_analysis.py at all
        # pulls in transformers. Constructing/importing it here
        # unconditionally meant `python terminal.py` started
        # downloading/loading BART (and importing transformers) even for a
        # session that only ever runs Step 2 (menu option 1). See
        # _get_analyzer() below.
        self.analyzer = None
        # Lazy for the same reason -- visualization.py imports bertopic at
        # module level (-> sentence_transformers -> transformers). See
        # _get_visualizer() below.
        self.visualizer = None
        # Lazy -- topic_modeling.py imports bertopic/sentence_transformers/
        # umap/hdbscan at module level. TopicModeler itself is never
        # instantiated (extract_topics is a @staticmethod), so this caches
        # the CLASS, not an instance. See _get_topic_modeler() below.
        self._topic_modeler_cls = None
        self.stopwords = nltk.corpus.stopwords.words("spanish")
        self.file_path = './data/output'
        self.current_topic_algorithm = None
        # Step 2 (NLLB) components -- lazily constructed on first use and
        # reused for the rest of this session (the model is ~600M
        # parameters; reloading it on every menu selection would be slow
        # on a laptop for no benefit, since decoding is deterministic and
        # the model doesn't change mid-session).
        self._nllb_translator = None
        self._nllb_cache = None
        self._nllb_similarity_scorer = None

    def _get_nllb_translator(self) -> "preprocess_v2.NLLBTranslator":
        if self._nllb_translator is None:
            # resolve_default_batch_size() (added after the fifth external
            # review) picks a conservative batch size automatically on
            # MPS/low-memory Macs -- see preprocess_v2.py -- rather than
            # always using the flat DEFAULT_BATCH_SIZE, which was tuned
            # assuming a discrete GPU or CPU and could push a low-memory
            # Mac into memory-pressure swapping on a full corpus run.
            self._nllb_translator = preprocess_v2.NLLBTranslator(
                batch_size=preprocess_v2.resolve_default_batch_size(),
            )
        return self._nllb_translator

    def _get_nllb_cache(self) -> "preprocess_v2.TranslationCache":
        if self._nllb_cache is None:
            # SENTENCE_CACHE_PATH (Round 7 redesign), not CACHE_PATH -- the
            # old whole-response/chunk cache from the six-hour real run
            # stays untouched and auditable (see
            # evidence/round7_full_corpus_run_2026-09-07/). See
            # preprocess_v2.SENTENCE_CACHE_PATH's own comment.
            self._nllb_cache = preprocess_v2.TranslationCache(preprocess_v2.SENTENCE_CACHE_PATH)
        return self._nllb_cache

    def _get_nllb_similarity_scorer(self) -> "preprocess_v2.SemanticSimilarityScorer":
        if self._nllb_similarity_scorer is None:
            self._nllb_similarity_scorer = preprocess_v2.SemanticSimilarityScorer()
        return self._nllb_similarity_scorer

    def _get_analyzer(self) -> "SentimentAnalyzer":
        """Lazily IMPORTS sentiment_analysis (which pulls in transformers)
        and constructs SentimentAnalyzer (nlptown sentiment +
        facebook/bart-large-mnli for the confusion signal) on first actual
        use, not at TerminalInterface() construction time or module import
        time.

        NOTE: the confusion signal here is still English BART-large-MNLI,
        run against Spanish-standardized sentences -- that mismatch is a
        known, tracked gap (the frozen plan calls for replacing it with
        multilingual mDeBERTa-v3-base-mnli-xnli), but that replacement is
        explicitly a LATER Stage 5 change, not part of this Step 2
        delivery. See STEP2_NLLB_CHANGES.md and the warning printed in
        _analyze_sentiment() below.
        """
        if self.analyzer is None:
            from sentiment_analysis import SentimentAnalyzer
            logger.info("Loading sentiment/confusion models (nlptown sentiment + facebook/bart-large-mnli)...")
            self.analyzer = SentimentAnalyzer()
        return self.analyzer

    def _get_visualizer(self) -> "Visualizer":
        """Lazily imports visualization.py (which imports bertopic at
        module level, pulling in transformers transitively) and constructs
        Visualizer() on first actual use (menu option 4), not at
        TerminalInterface() construction time or module import time.
        """
        if self.visualizer is None:
            from visualization import Visualizer
            self.visualizer = Visualizer()
        return self.visualizer

    def _get_topic_modeler(self):
        """Lazily imports topic_modeling.py (bertopic/sentence_transformers/
        umap/hdbscan, pulling in transformers transitively) on first actual
        use (menu option 2), not at module import time. Returns the
        TopicModeler CLASS itself, not an instance -- TopicModeler.
        extract_topics is a @staticmethod, so TopicModeler is never
        instantiated anywhere in this project; this just caches the import.
        """
        if self._topic_modeler_cls is None:
            from topic_modeling import TopicModeler
            self._topic_modeler_cls = TopicModeler
        return self._topic_modeler_cls

    def _step2_bridge_is_fresh(self) -> bool:
        """True only when the Step 2 bridge files (topic_modeling_input.json,
        sentiment_input_ca.json, fully_processed_ca.json) were actually
        produced by a STEP2_VALID=True run of THIS pipeline, against the
        CURRENT data/input/interviews.json -- not merely present on disk,
        and not merely from some earlier version of the corpus.

        Without the schema/step2_valid check, _analyze_topics()/
        _analyze_sentiment()'s original "if the file exists, skip
        regenerating it" gate would happily accept a stale file left over
        from a previous pipeline version (in this delivery, that risk is
        concrete: this project's data/output/ has, at various points, held
        Catalan-era bridge files from before the NLLB redesign) without
        ever running Step 2 or checking STEP2_VALID. A bare filename check
        cannot tell "fresh, validated Step 2 output" apart from "some file
        happens to be at this path" -- this sidecar can.

        Without the input-hash check (added after the second external
        review), a genuinely valid Step 2 run's sidecar would still read as
        fresh even after someone edited or replaced interviews.json
        underneath it -- e.g. Step 2 succeeds, the corpus file is later
        changed, and topic modelling/sentiment analysis then unknowingly
        run against bridge files that no longer correspond to the current
        corpus. Recomputing the hash here and comparing against what was
        recorded at Step 2 write-time catches that case: a changed corpus
        correctly invalidates the bridge, forcing a fresh Step 2 run.
        """
        metadata_path = os.path.join(self.file_path, STEP2_BRIDGE_METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            return False
        try:
            metadata = JsonHandler.read_json(metadata_path)
        except Exception:
            return False
        if not metadata:
            return False
        if not (
            metadata.get("schema_version") == STEP2_BRIDGE_SCHEMA_VERSION
            and metadata.get("step2_valid") is True
        ):
            return False

        recorded_input_hash = metadata.get("input_sha256")
        if not recorded_input_hash:
            return False
        current_input_hash = _hash_file(STEP2_INTERVIEWS_INPUT_FILE)
        if current_input_hash is None or current_input_hash != recorded_input_hash:
            return False

        return True

    def _select_questions(self) -> Dict:
        """Display all questions and let user select which to analyze"""
        input_file = "./data/input/interviews.json"
        if not os.path.exists(input_file):
            logger.error(f"Input file not found at {input_file}")
            return {}

        raw_data = JsonHandler.read_json(input_file)
        if not raw_data:
            logger.error("No data found in input file")
            return {}

        # Collect all unique questions across interviews
        all_questions = set()
        for interview in raw_data.values():
            all_questions.update(interview.keys())

        # Sort questions numerically
        sorted_questions = sorted(all_questions, key=lambda x: [int(i) for i in x.split('.') if i.isdigit()])

        print("\n📋 Available Questions:")
        for i, question in enumerate(sorted_questions, 1):
            print(f"{i}. {question}")

        print("\nSelect questions to analyze (comma-separated numbers, or 'all'):")
        selection = input("Your selection: ").strip().lower()

        selected_questions = []
        if selection == 'all':
            selected_questions = sorted_questions
            print("✅ Selected all questions")
        else:
            try:
                selected_indices = [int(i.strip()) for i in selection.split(',')]
                selected_questions = [sorted_questions[i - 1] for i in selected_indices]
                print(f"✅ Selected questions: {', '.join(selected_questions)}")
            except Exception as e:
                logger.error(f"Invalid selection: {str(e)}")
                return {}

        # Filter the data to only include selected questions
        filtered_data = {}
        for interview_id, questions in raw_data.items():
            filtered_questions = {q: text for q, text in questions.items() if q in selected_questions}
            if filtered_questions:
                filtered_data[interview_id] = filtered_questions

        return filtered_data

    def _create_processed_versions(self, source_data: Optional[Dict] = None) -> bool:
        """Step 2 entry point: NLLB Catalan->Spanish translation + Spanish
        preprocessing (preprocess_v2.process()), gated by STEP2_VALID.

        This is the frozen Step 2 redesign's integration into the normal
        `python terminal.py` workflow. `terminal.py` remains the everyday
        entry point (per the agreed design) -- it just calls the new
        Step 2 module internally instead of the old
        translate_to_catalan()/full_preprocess() path.

        source_data=None (the path taken when _analyze_topics()/
        _analyze_sentiment() fall back into this method because their
        input file is missing) loads the full corpus from
        data/input/interviews.json, the same file _select_questions()
        reads -- so that fallback no longer crashes with a missing
        required argument as it would have against the old signature.

        Critical gate: if the Step 2 report's STEP2_VALID is False, this
        method logs a clear failure message and returns False WITHOUT
        writing fully_processed_ca.json / sentiment_input_ca.json /
        topic_modeling_input.json. Because those are exactly the files
        _analyze_topics() and _analyze_sentiment() check for before
        running, an invalid Step 2 run leaves them absent (or stale from
        a previous valid run only if one ever succeeded) and both
        methods' existing "file missing -> call _create_processed_versions
        again" fallback simply gets False again -- there is no separate
        flag to maintain, and no path from an invalid Step 2 run into
        topic modelling, sentiment analysis, or (later) confusion
        analysis.
        """
        if source_data is None:
            input_file = "./data/input/interviews.json"
            if not os.path.exists(input_file):
                logger.error(f"Input file not found at {input_file}")
                return False
            source_data = JsonHandler.read_json(input_file)
            if not source_data:
                logger.error("No data found in input file")
                return False

        logger.info("Running Step 2 (NLLB Catalan->Spanish translation + preprocessing)...")
        translator = self._get_nllb_translator()
        cache = self._get_nllb_cache()
        similarity_scorer = self._get_nllb_similarity_scorer()

        # Deliberately NOT passing frozen_segmentation here: source_data
        # can legitimately be a filtered subset (_select_questions()) or
        # synthetic data that would never match a frozen file keyed to the
        # full 950-response corpus, and this method is the interactive/
        # menu-driven entry point, not the one true full-corpus production
        # run. process() computes segmentation live in that case -- via
        # the exact same segment_source_sentences() function the frozen
        # file itself was built from, so this is byte-identical to the
        # frozen structure whenever source_data IS the full corpus (see
        # the Round 7 segmentation audit -- 0 reconstruction mismatches).
        # preprocess_v2.main() (the `python preprocess_v2.py` CLI path) is
        # what ties a run to the actual frozen artifact -- see there.
        responses_out, sentences_out, report = preprocess_v2.process(
            source_data, self.preprocessor, cache, translator, similarity_scorer,
            show_progress=True,
        )
        # Safety-net save: process() already saves the cache atomically
        # after every batch during translation (see the fifth external
        # review) -- this is normally a no-op by the time we get here.
        cache.save()

        preprocess_v2.print_validation_report(report)

        # These are Step 2's own audit-trail outputs (new filenames,
        # unrelated to the legacy bridge below) -- written whether or not
        # STEP2_VALID is True, so a failed run leaves a full diagnostic
        # record behind, not silence.
        JsonHandler.create_json(
            {"metadata": {"generated_at": report["generated_at"], "count": len(responses_out)}, "responses": responses_out},
            "preprocessed_responses_v2.json", self.file_path,
        )
        JsonHandler.create_json(
            {"metadata": {"generated_at": report["generated_at"], "count": len(sentences_out)}, "sentences": sentences_out},
            "preprocessed_sentences_v2.json", self.file_path,
        )
        JsonHandler.create_json(report, "preprocessing_language_report.json", self.file_path)

        if not report["STEP2_VALID"]:
            logger.error(
                "Step 2 FAILED validation (STEP2_VALID=False) -- stopping here. "
                "Topic modelling, sentiment analysis, and confusion analysis will "
                "NOT run against this output. structural_validity="
                f"{report['structural_validity']}, translation_completeness_validity="
                f"{report['translation_completeness_validity']}, translation_output_validity="
                f"{report['translation_output_validity']} (translation_sanity_status="
                f"{report['translation_sanity_status']}). See "
                "data/output/preprocessing_language_report.json for full detail."
            )
            return False

        # STEP2_VALID -- bridge the validated Spanish-standardized output
        # into the existing legacy filenames _analyze_topics() and
        # _analyze_sentiment() already read. Filenames keep their
        # historical "_ca" suffix for backward compatibility with those
        # unchanged methods; the *content* is now Spanish
        # (topic_text_es_clean / text_es), not Catalan -- see
        # STEP2_NLLB_CHANGES.md for why the suffix was left alone here.
        full_processed: Dict[str, Dict[str, str]] = {}
        sentiment_ready: Dict[str, Dict[str, List[str]]] = {}

        for response in responses_out.values():
            if response.get("status") != "OK":
                continue
            interview_id = response["interview_id"]
            question_id = response["question_id"]

            full_processed.setdefault(interview_id, {})[question_id] = response.get("topic_text_es_clean") or ""

            sentence_texts = [
                sentences_out[sid]["text_es"]
                for sid in response.get("sentence_ids", [])
                if sid in sentences_out
            ]
            sentiment_ready.setdefault(interview_id, {})[question_id] = sentence_texts

        JsonHandler.create_json(full_processed, "fully_processed_ca.json", self.file_path)
        JsonHandler.create_json(sentiment_ready, "sentiment_input_ca.json", self.file_path)
        JsonHandler.create_topic_modeling_input(full_processed, self.file_path)

        # Written LAST, only once every bridge file above has been written
        # successfully -- this is what _step2_bridge_is_fresh() checks
        # before _analyze_topics()/_analyze_sentiment() will trust the
        # bridge files as fresh, validated Step 2 output rather than a
        # stale file that merely happens to exist at that path.
        #
        # input_sha256 (added after the second external review): a hash of
        # data/input/interviews.json AS IT EXISTED for this run, so
        # _step2_bridge_is_fresh() can later detect a corpus that changed
        # since this validated run -- see that method and
        # STEP2_INTERVIEWS_INPUT_FILE above. nllb_model_name/
        # nllb_generation_version are recorded too, for the same
        # reproducibility reason (auditing which model/decoding settings a
        # given bridged output actually came from).
        JsonHandler.create_json(
            {
                "schema_version": STEP2_BRIDGE_SCHEMA_VERSION,
                "step2_valid": True,
                "generated_at": report["generated_at"],
                "source_interview_count": len(full_processed),
                "input_file": STEP2_INTERVIEWS_INPUT_FILE,
                "input_sha256": _hash_file(STEP2_INTERVIEWS_INPUT_FILE),
                "nllb_model_name": report.get("model_info", {}).get("model_name"),
                "nllb_generation_version": report.get("model_info", {}).get("generation_version"),
            },
            STEP2_BRIDGE_METADATA_FILENAME, self.file_path,
        )

        logger.info(
            f"Step 2 complete, STEP2_VALID=True: bridged {len(full_processed)} interview(s) "
            "into fully_processed_ca.json / sentiment_input_ca.json / topic_modeling_input.json"
        )
        return True

    def _analyze_topics(self) -> bool:
        """Run topic modeling using the dedicated input file"""
        input_file = os.path.join(self.file_path, "topic_modeling_input.json")

        # Checking mere existence is not enough: a file at this exact path
        # can be left over from a previous pipeline version (this project
        # has historically had Catalan-era topic_modeling_input.json files
        # at this path) rather than fresh output from a validated Step 2
        # run. _step2_bridge_is_fresh() requires the sidecar
        # _create_processed_versions() only writes after a STEP2_VALID=True
        # run -- see its docstring and STEP2_NLLB_CHANGES.md.
        if not os.path.exists(input_file) or not self._step2_bridge_is_fresh():
            logger.warning(
                "Topic modeling input is missing or was not produced by a validated "
                "Step 2 run. Running Step 2 now..."
            )
            if not self._create_processed_versions():
                return False

        try:
            documents = JsonHandler.load_topic_modeling_input(input_file)
            if not documents:
                logger.error("No valid documents found in topic modeling input")
                return False

            logger.info(f"Loaded {len(documents)} documents for topic modeling")
            if len(documents) < 10:
                logger.error(f"Insufficient documents ({len(documents)}). Need ≥10.")
                return False

            # Get algorithm choice from user
            print("\nAvailable topic modeling algorithms:")
            print("1. BERTopic (default)")
            print("2. LDA")
            print("3. NMF")
            print("4. SVD")
            choice = input("Select algorithm (1-4): ").strip() or "1"

            algorithm_map = {
                "1": "bertopic",
                "2": "lda",
                "3": "nmf",
                "4": "svd"
            }
            algorithm = algorithm_map.get(choice, "bertopic")
            self.current_topic_algorithm = algorithm

            # Run topic modeling
            topics = self._get_topic_modeler().extract_topics(
                documents,
                algorithm=algorithm,
                n_topics="auto" if algorithm == "bertopic" else 10,
                n_words=10,
                stopwords=self.stopwords
            )

            # Save results with algorithm-specific filename
            output_filename = f"topic_results_{algorithm}.json"  # This line already uses the algorithm name
            JsonHandler.create_json(
                {
                    "topics": topics.get("topics", {}),
                    "metadata": {
                        "source": "topic_modeling_input.json",
                        "document_count": len(documents),
                        "algorithm": algorithm,  # This includes the algorithm in metadata
                        "n_topics_requested": "auto" if algorithm == "bertopic" else 10,
                        "random_state": 42,
                        "generated_at": datetime.now().isoformat()
                    },
                    "diagnostics": topics.get("diagnostics", {})
                },
                output_filename,
                self.file_path
            )

            logger.info(f"Topic analysis using {algorithm.upper()} completed successfully")
            return True  # ✅ FIXED: Added missing return statement here

        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
            return False

    def _analyze_sentiment(self) -> bool:
        """Run sentiment analysis using sentiment_input_ca.json"""
        input_file = os.path.join(self.file_path, "sentiment_input_ca.json")
        # See _analyze_topics()'s identical comment: existence alone is not
        # a reliable signal that this file came from a validated Step 2
        # run rather than a historical/stale one.
        if not os.path.exists(input_file) or not self._step2_bridge_is_fresh():
            logger.warning(
                "Sentiment input is missing or was not produced by a validated Step 2 "
                "run. Running Step 2 now..."
            )
            if not self._create_processed_versions():
                return False

        print(
            "\nNote: the 'confusion' signal below still comes from English "
            "facebook/bart-large-mnli run against Spanish-standardized sentences "
            "(a pre-existing mismatch). This is a known, tracked gap -- the frozen "
            "plan replaces it with multilingual mDeBERTa-v3-base-mnli-xnli, but that "
            "replacement is a later Stage 5 change, not part of this Step 2 delivery. "
            "Treat 'confusion_percentage' below as legacy until that lands; the "
            "sentiment score itself is unaffected by this.\n"
        )

        try:
            data = JsonHandler.read_json(input_file)
            if not data:
                logger.error("No data available for sentiment analysis")
                return False

            analyzer = self._get_analyzer()
            results = {}
            total_sentences = 0

            # Analyze each interview and question
            for interview_id, questions in tqdm(data.items(), desc="Analyzing interviews"):
                results[interview_id] = {}
                for question_id, sentences in questions.items():
                    sentence_results = []
                    for sentence in sentences:
                        analysis = analyzer.analyze(sentence)
                        sentence_results.append({
                            "text": sentence,
                            "sentiment": analysis
                        })
                        total_sentences += 1

                    # Calculate overall sentiment for this question
                    if sentence_results:
                        avg_score = sum(s['sentiment']['score'] for s in sentence_results) / len(sentence_results)
                        confusion_count = sum(1 for s in sentence_results
                                          if s['sentiment']['confusion'] and
                                          s['sentiment']['confusion']['status'] == "CONFUSED")

                        results[interview_id][question_id] = {
                            "sentences": sentence_results,
                            "overall_sentiment": {
                                "score": avg_score,
                                "confusion_percentage": (confusion_count / len(sentence_results)) * 100
                            }
                        }

            # Save results
            JsonHandler.create_json(
                results,
                "sentiment_results_ca.json",
                self.file_path
            )

            logger.info(f"Analyzed {total_sentences} sentences across {len(data)} interviews")
            return True

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return False

    def _visualize_results(self):
        """Generate visualizations with algorithm-specific organization"""
        try:
            # Check for sentiment results
            sentiment_file = os.path.join(self.file_path, "sentiment_results_ca.json")
            if not os.path.exists(sentiment_file):
                logger.error("No sentiment analysis results found. Run sentiment analysis first.")
                return

            sentiment_results = JsonHandler.read_json(sentiment_file)
            if not sentiment_results:
                logger.error("Could not load sentiment results")
                return

            # Find all topic result files
            topic_files = []
            for filename in os.listdir(self.file_path):
                if filename.startswith("topic_results_") and filename.endswith(".json"):
                    topic_files.append(os.path.join(self.file_path, filename))

            if not topic_files:
                logger.error("No topic analysis results found. Run topic analysis first.")
                return

            # Process each topic result file
            for topic_file in topic_files:
                try:
                    topic_results = JsonHandler.read_json(topic_file)
                    if not topic_results:
                        continue

                    # Get algorithm from filename
                    algo = os.path.basename(topic_file).split('_')[2].split('.')[0].lower()

                    # Generate visualizations
                    self._get_visualizer().generate_all_visualizations(
                        topic_data=topic_results,
                        sentiment_results=sentiment_results
                    )

                except Exception as e:
                    logger.error(f"Failed to process {topic_file}: {str(e)}")

            logger.info("Visualizations generated successfully")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")

    def run(self):
        """Main interface menu"""
        while True:
            print("\n📋 Main Menu:")
            print("1. Select questions and create processed versions")
            print("2. Analyze topics")
            print("3. Analyze sentiment (confusion signal is legacy BART pending Stage 5 mDeBERTa)")
            print("4. Visualize results")
            print("0. Exit")

            try:
                choice = input("Enter choice (0-4): ").strip()
                if choice == "1":
                    selected_data = self._select_questions()
                    if selected_data:
                        self._create_processed_versions(selected_data)
                elif choice == "2":
                    self._analyze_topics()
                elif choice == "3":
                    self._analyze_sentiment()
                elif choice == "4":
                    self._visualize_results()
                elif choice == "0":
                    print("\n👋 Exiting program.")
                    break
                else:
                    print("⚠️ Invalid choice. Please try again.")
            except Exception as e:
                logger.error(f"Menu error: {str(e)}")


if __name__ == "__main__":
    # `python terminal.py` is the normal, everyday entry point for this
    # project (per the frozen Step 2 design) -- this was previously
    # missing, which meant running this file directly did nothing.
    # `python main.py --mode terminal` already reached the same
    # TerminalInterface().run() and still works unchanged.
    TerminalInterface().run()