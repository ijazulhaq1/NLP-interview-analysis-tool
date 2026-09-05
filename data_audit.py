"""
data_audit.py — R1 (mandatory, first script)

Reconciles interview/response/sentence counts across every stage of the
existing V8 pipeline, without re-running any model. Answers, concretely:

    - Why does the pipeline currently produce 921 / 930 / 950 responses
      depending on which file you look at?
    - Why does it produce 6,071 / 6,073 / 6,284 / 6,528 sentences depending
      on which file you look at?
    - Where exactly do records get dropped, and is every drop explained?
    - Are there TWO different copies of the same pipeline stage on disk
      (data/*.json vs data/output/*.json), and if so, which one does the
      code actually read, and do they even agree with each other?
    - Are there any duplicate (interview_id, question_id) records anywhere,
      including in structures where a set-based diff would silently hide
      them (the flat "documents" list in topic_modeling_input.json)?
    - Is there any sign that interviewer speech (as opposed to respondent
      speech) is present in the analysis inputs?

No result generation downstream should be trusted until every assertion
in this script passes, or the discrepancy it reports is understood and
explicitly accepted.

Run:
    python data_audit.py
Outputs:
    results/data_audit_report.json
    results/data_audit_report.md
"""
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from json_handler import JsonHandler

DATA_INPUT = "data/input"
DATA_OUTPUT = "data/output"
DATA_ROOT = "data"
RESULTS_DIR = "results"

# terminal.py hardcodes self.file_path = './data/output' — that is the copy
# the actual running pipeline reads and writes. Anything under data/*.json
# directly (no /output/) is a second, separate copy sitting next to it.
PIPELINE_WIRED_DIR = DATA_OUTPUT

DUPLICATED_FILENAMES = [
    "topic_modeling_input.json",
    "sentiment_input_ca.json",
    "sentiment_results_ca.json",
]

# Sentences at or below this token count are treated as "short" for the
# interviewer/boilerplate-contamination heuristic.
SHORT_SENTENCE_MAX_TOKENS = 4
# Flag a question_id if its most common short sentence appears in at least
# this fraction of the interviews that have that question.
CONTAMINATION_COVERAGE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Raw-parse helpers (list-based, so duplicates are NOT silently collapsed
# the way they would be if we built a dict straight away)
# ---------------------------------------------------------------------------

def parse_questions_list(content: str) -> List[Tuple[str, str]]:
    """Same regex as JsonHandler.parse_questions, but returns a list of
    (question_number, text) pairs instead of a dict, so a transcript that
    repeats a question marker twice shows up as two list entries instead
    of silently overwriting itself."""
    try:
        parts = re.split(r'-\.Pregunta ([\d\.a-zA-Z]+)\.-', content)[1:]
        pairs = []
        for i in range(0, len(parts), 2):
            question_number = parts[i].strip()
            question_text = re.sub(
                r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+',
                '',
                parts[i + 1].strip()
            )
            pairs.append((question_number, question_text.replace('\n', ' ').replace('  ', ' ')))
        return pairs
    except Exception:
        return []


def load_raw_txt_files_list() -> Dict[str, List[Tuple[str, str]]]:
    """interview_id -> list of (question_id, text), duplicates preserved."""
    parsed = {}
    if not os.path.isdir(DATA_INPUT):
        return parsed
    for fname in sorted(os.listdir(DATA_INPUT)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DATA_INPUT, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed[fname] = parse_questions_list(content)
    return parsed


def duplicate_keys_in_raw_parse(raw_list: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[str]]:
    """Detect any (interview_id, question_id) that appears more than once
    in a *fresh* regex parse, before it gets collapsed into a dict."""
    dupes = {}
    for interview_id, pairs in raw_list.items():
        counts = Counter(q for q, _ in pairs)
        repeated = [q for q, c in counts.items() if c > 1]
        if repeated:
            dupes[interview_id] = repeated
    return dupes


# ---------------------------------------------------------------------------
# Stage loaders (dict-based — this is what the pipeline itself consumes)
# ---------------------------------------------------------------------------

def load_interviews_json() -> Dict[str, Dict[str, str]]:
    return JsonHandler.read_json(os.path.join(DATA_INPUT, "interviews.json"))


def load_fully_processed() -> Dict[str, Dict[str, str]]:
    return JsonHandler.read_json(os.path.join(DATA_OUTPUT, "fully_processed_ca.json"))


def load_topic_modeling_input(base_dir: str = DATA_OUTPUT) -> Tuple[List[dict], dict]:
    path = os.path.join(base_dir, "topic_modeling_input.json")
    data = JsonHandler.read_json(path)
    return data.get("documents", []), data.get("stats", {})


def load_sentiment_input(base_dir: str = DATA_OUTPUT) -> Dict[str, Dict[str, List[str]]]:
    return JsonHandler.read_json(os.path.join(base_dir, "sentiment_input_ca.json"))


def load_sentiment_results(base_dir: str = DATA_OUTPUT) -> Dict[str, Dict[str, dict]]:
    return JsonHandler.read_json(os.path.join(base_dir, "sentiment_results_ca.json"))


def load_topic_results_bertopic() -> dict:
    return JsonHandler.read_json(os.path.join(DATA_OUTPUT, "topic_results_bertopic.json"))


# ---------------------------------------------------------------------------
# Stage summaries
# ---------------------------------------------------------------------------

def summarize_qa_dict(d: Dict[str, Dict[str, str]]) -> dict:
    """Summarize a {interview_id: {question_id: text}} structure.

    NOTE: a Python dict cannot itself contain a duplicate key — if the
    source ever produced the same question_id twice for one interview, the
    second write already silently overwrote the first by the time this
    structure exists. That is exactly why duplicate_keys_in_raw_parse()
    above checks the pre-dict list form instead. This function documents
    that limitation rather than pretending duplicates are impossible.
    """
    keys = set()
    empty_keys = set()
    for interview_id, questions in d.items():
        for question_id, text in questions.items():
            keys.add((interview_id, question_id))
            if not str(text).strip():
                empty_keys.add((interview_id, question_id))
    return {
        "n_interviews": len(d),
        "n_responses": len(keys),
        "n_empty_responses": len(empty_keys),
        "empty_keys": empty_keys,
        "keys": keys,
    }


def summarize_topic_modeling_input(documents: List[dict]) -> dict:
    """documents is a flat LIST, so duplicate (interview_id, question_id)
    metadata entries are structurally possible and must be checked
    explicitly with a Counter, not hidden behind a set()."""
    key_list = [
        (doc.get("metadata", {}).get("interview_id"), doc.get("metadata", {}).get("question_id"))
        for doc in documents
    ]
    counts = Counter(key_list)
    duplicate_keys = {k: c for k, c in counts.items() if c > 1}
    keys = set(key_list)
    return {
        "n_documents_total": len(documents),
        "n_interviews": len({k[0] for k in keys}),
        "n_responses_unique_keys": len(keys),
        "n_duplicate_keys": len(duplicate_keys),
        "duplicate_keys": duplicate_keys,
        "keys": keys,
    }


def summarize_sentiment_input(d: Dict[str, Dict[str, List[str]]]) -> dict:
    keys = set()
    n_sentences = 0
    empty_sentence_lists = 0
    for interview_id, questions in d.items():
        for question_id, sentences in questions.items():
            keys.add((interview_id, question_id))
            n_sentences += len(sentences)
            if len(sentences) == 0:
                empty_sentence_lists += 1
    return {
        "n_interviews": len(d),
        "n_responses": len(keys),
        "n_sentences": n_sentences,
        "n_empty_responses": empty_sentence_lists,
        "keys": keys,
    }


def summarize_sentiment_results(d: Dict[str, Dict[str, dict]]) -> dict:
    keys = set()
    n_sentences = 0
    confused = 0
    not_confused = 0
    no_confusion_field = 0
    for interview_id, questions in d.items():
        for question_id, payload in questions.items():
            keys.add((interview_id, question_id))
            sentences = payload.get("sentences", [])
            n_sentences += len(sentences)
            for s in sentences:
                c = s.get("sentiment", {}).get("confusion")
                if c is None:
                    no_confusion_field += 1
                elif c.get("status") == "CONFUSED":
                    confused += 1
                elif c.get("status") == "NOT_CONFUSED":
                    not_confused += 1
    return {
        "n_interviews": len(d),
        "n_responses": len(keys),
        "n_sentences": n_sentences,
        "confused": confused,
        "not_confused": not_confused,
        "no_confusion_field": no_confusion_field,
        "confusion_total_check": confused + not_confused + no_confusion_field == n_sentences,
        "confused_pct": round(100 * confused / n_sentences, 1) if n_sentences else None,
        "keys": keys,
    }


def summarize_topic_results(d: dict) -> dict:
    topics = d.get("topics", {})
    per_topic = {name: t.get("count", 0) for name, t in topics.items()}
    assigned_total = sum(per_topic.values())
    diagnostics = d.get("diagnostics", {})
    metadata = d.get("metadata", {})
    doc_count = metadata.get("document_count")
    outlier_pct = diagnostics.get("outlier_percentage")
    size_dist = diagnostics.get("topic_size_distribution", {})
    total_from_size_dist = sum(int(k) * v for k, v in size_dist.items())
    outliers_implied = total_from_size_dist - assigned_total
    return {
        "n_named_topics": len(topics),
        "per_topic_counts": per_topic,
        "assigned_total": assigned_total,
        "document_count_metadata": doc_count,
        "outlier_percentage_reported": outlier_pct,
        "total_from_size_distribution": total_from_size_dist,
        "outliers_implied_by_size_distribution": outliers_implied,
        "assigned_plus_outliers_equals_doc_count": (
            (assigned_total + outliers_implied) == doc_count
            if doc_count is not None else None
        ),
    }


def diff_keys(name_a: str, keys_a: set, name_b: str, keys_b: set) -> dict:
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    return {
        "in_" + name_a + "_not_" + name_b: sorted(list(only_a))[:20] + (
            ["... (%d more)" % (len(only_a) - 20)] if len(only_a) > 20 else []
        ),
        "in_" + name_b + "_not_" + name_a: sorted(list(only_b))[:20] + (
            ["... (%d more)" % (len(only_b) - 20)] if len(only_b) > 20 else []
        ),
        "n_only_in_" + name_a: len(only_a),
        "n_only_in_" + name_b: len(only_b),
        # returned for programmatic use (e.g. the exact-match assertion below)
        "_only_a_set": only_a,
        "_only_b_set": only_b,
    }


# ---------------------------------------------------------------------------
# Duplicate-file audit: data/*.json vs data/output/*.json
# ---------------------------------------------------------------------------

def audit_duplicate_files() -> dict:
    results = {}
    for fname in DUPLICATED_FILENAMES:
        root_path = os.path.join(DATA_ROOT, fname)
        output_path = os.path.join(DATA_OUTPUT, fname)
        entry = {
            "root_path": root_path,
            "output_path": output_path,
            "root_exists": os.path.exists(root_path),
            "output_exists": os.path.exists(output_path),
        }
        if entry["root_exists"] and entry["output_exists"]:
            entry["root_size_bytes"] = os.path.getsize(root_path)
            entry["output_size_bytes"] = os.path.getsize(output_path)
            with open(root_path, "rb") as f:
                root_bytes = f.read()
            with open(output_path, "rb") as f:
                output_bytes = f.read()
            entry["byte_identical"] = root_bytes == output_bytes

            if fname == "topic_modeling_input.json":
                root_docs, root_stats = load_topic_modeling_input(DATA_ROOT)
                out_docs, out_stats = load_topic_modeling_input(DATA_OUTPUT)
                entry["root_summary"] = {
                    "n_documents": len(root_docs),
                    "generated_at": root_stats.get("generated_at"),
                }
                entry["output_summary"] = {
                    "n_documents": len(out_docs),
                    "generated_at": out_stats.get("generated_at"),
                }
            elif fname == "sentiment_input_ca.json":
                root_d = load_sentiment_input(DATA_ROOT)
                out_d = load_sentiment_input(DATA_OUTPUT)
                root_s = summarize_sentiment_input(root_d)
                out_s = summarize_sentiment_input(out_d)
                entry["root_summary"] = {"n_responses": root_s["n_responses"], "n_sentences": root_s["n_sentences"]}
                entry["output_summary"] = {"n_responses": out_s["n_responses"], "n_sentences": out_s["n_sentences"]}
            elif fname == "sentiment_results_ca.json":
                root_d = load_sentiment_results(DATA_ROOT)
                out_d = load_sentiment_results(DATA_OUTPUT)
                root_s = summarize_sentiment_results(root_d)
                out_s = summarize_sentiment_results(out_d)
                entry["root_summary"] = {
                    "n_responses": root_s["n_responses"], "n_sentences": root_s["n_sentences"],
                    "confused": root_s["confused"], "not_confused": root_s["not_confused"],
                    "confused_pct": root_s["confused_pct"],
                }
                entry["output_summary"] = {
                    "n_responses": out_s["n_responses"], "n_sentences": out_s["n_sentences"],
                    "confused": out_s["confused"], "not_confused": out_s["not_confused"],
                    "confused_pct": out_s["confused_pct"],
                }
        results[fname] = entry
    return results


# ---------------------------------------------------------------------------
# Interviewer / boilerplate contamination heuristic
# ---------------------------------------------------------------------------

def normalize_sentence(text: str) -> str:
    text = text.strip()
    text = text.lstrip("-").strip()
    text = re.sub(r"[^\w\sáéíóúàèòïüçÁÉÍÓÚÀÈÒÏÜÇñÑ]", "", text)
    return text.lower().strip()


def audit_interviewer_contamination(sentiment_input: Dict[str, Dict[str, List[str]]]) -> dict:
    """For each question_id, look for a short (<=SHORT_SENTENCE_MAX_TOKENS
    token) sentence that recurs near-verbatim across an unusually large
    fraction of the interviews answering that question. This does NOT
    prove interviewer contamination (no speaker diarisation exists in this
    project — see the "-" turn markers in the raw transcripts), but a
    short phrase repeating across many different interviews at the same
    question is a reasonable, cheap proxy worth a human look before it is
    trusted as respondent content, and it is exactly the kind of thing
    R1 #5 ("clarify whether interviewer questions were removed") is
    asking about.
    """
    by_question: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    interviews_per_question: Dict[str, set] = defaultdict(set)

    for interview_id, questions in sentiment_input.items():
        for question_id, sentences in questions.items():
            interviews_per_question[question_id].add(interview_id)
            for sent in sentences:
                norm = normalize_sentence(sent)
                if not norm:
                    continue
                n_tokens = len(norm.split())
                if n_tokens <= SHORT_SENTENCE_MAX_TOKENS:
                    by_question[question_id][norm].add(interview_id)

    flagged = []
    for question_id, phrase_map in by_question.items():
        total_interviews = len(interviews_per_question[question_id])
        if total_interviews == 0:
            continue
        best_phrase, best_interviews = max(
            phrase_map.items(), key=lambda kv: len(kv[1]), default=(None, set())
        )
        if best_phrase is None:
            continue
        coverage = len(best_interviews) / total_interviews
        if coverage >= CONTAMINATION_COVERAGE_THRESHOLD:
            flagged.append({
                "question_id": question_id,
                "phrase": best_phrase,
                "n_interviews_with_phrase": len(best_interviews),
                "n_interviews_with_question": total_interviews,
                "coverage": round(coverage, 2),
            })

    flagged.sort(key=lambda x: -x["coverage"])
    return {
        "threshold_used": CONTAMINATION_COVERAGE_THRESHOLD,
        "short_sentence_max_tokens": SHORT_SENTENCE_MAX_TOKENS,
        "n_questions_flagged": len(flagged),
        "n_questions_total": len(interviews_per_question),
        "flagged": flagged[:30],
        "caveat": (
            "No speaker diarisation exists anywhere in this pipeline. A flagged "
            "phrase here is NOT proof of interviewer speech — it is a cheap "
            "recurrence signal worth a manual look, nothing more."
        ),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Raw parse + duplicate-marker check (list-based, pre-dict) ---------
    raw_txt_list = load_raw_txt_files_list()
    raw_marker_dupes = duplicate_keys_in_raw_parse(raw_txt_list)
    raw_txt_dict = {iid: dict(pairs) for iid, pairs in raw_txt_list.items()}
    raw_txt_summary = summarize_qa_dict(raw_txt_dict)
    raw_txt_summary["n_files_present"] = len(raw_txt_dict)

    interviews_json = load_interviews_json()
    interviews_summary = summarize_qa_dict(interviews_json)

    fully_processed = load_fully_processed()
    fully_processed_summary = summarize_qa_dict(fully_processed)

    tm_documents, tm_stats = load_topic_modeling_input(DATA_OUTPUT)
    tm_summary = summarize_topic_modeling_input(tm_documents)

    sentiment_input = load_sentiment_input(DATA_OUTPUT)
    sentiment_input_summary = summarize_sentiment_input(sentiment_input)

    sentiment_results = load_sentiment_results(DATA_OUTPUT)
    sentiment_results_summary = summarize_sentiment_results(sentiment_results)

    topic_results = load_topic_results_bertopic()
    topic_summary = summarize_topic_results(topic_results)

    duplicate_files_report = audit_duplicate_files()
    contamination_report = audit_interviewer_contamination(sentiment_input)

    # --- Cross-stage key diffs ----------------------------------------------
    raw_vs_interviews = diff_keys(
        "raw_txt", raw_txt_summary["keys"], "interviews_json", interviews_summary["keys"]
    )
    interviews_vs_fully_processed = diff_keys(
        "interviews_json", interviews_summary["keys"], "fully_processed", fully_processed_summary["keys"]
    )
    fully_processed_vs_topic_input = diff_keys(
        "fully_processed", fully_processed_summary["keys"], "topic_modeling_input", tm_summary["keys"]
    )
    fully_processed_vs_sentiment_input = diff_keys(
        "fully_processed", fully_processed_summary["keys"], "sentiment_input", sentiment_input_summary["keys"]
    )
    sentiment_input_vs_results = diff_keys(
        "sentiment_input", sentiment_input_summary["keys"], "sentiment_results", sentiment_results_summary["keys"]
    )

    # --- Assertions ----------------------------------------------------------
    # Fixed per review: topic_modeling_input.json is EXPECTED to drop
    # responses that are empty after processing. The correct check is that
    # (a) the counts reconcile exactly once empties are subtracted, and
    # (b) the exact set of missing keys equals the exact set of empty keys
    # (i.e. nothing OTHER than empty responses went missing).
    fully_processed_nonempty_count = (
        fully_processed_summary["n_responses"] - fully_processed_summary["n_empty_responses"]
    )
    missing_from_topic_input = fully_processed_vs_topic_input["_only_a_set"]
    empty_fully_processed_keys = fully_processed_summary["empty_keys"]

    assertions = {
        "sentiment_results_partition_holds (confused+not_confused+missing == total)": (
            sentiment_results_summary["confusion_total_check"]
        ),
        "topic_assigned_plus_outliers_equals_document_count": (
            topic_summary["assigned_plus_outliers_equals_doc_count"]
        ),
        "topic_modeling_input_doc_count_matches_stats_field": (
            len(tm_documents) == tm_stats.get("total_documents")
        ),
        "no_duplicate_keys_in_topic_modeling_input_documents_list": (
            tm_summary["n_duplicate_keys"] == 0
        ),
        "no_duplicate_question_markers_in_any_raw_transcript": (
            len(raw_marker_dupes) == 0
        ),
        "nonempty_fully_processed_count_matches_topic_modeling_input_count": (
            fully_processed_nonempty_count == tm_summary["n_responses_unique_keys"]
        ),
        "topic_modeling_input_drops_are_EXACTLY_the_empty_fully_processed_responses": (
            missing_from_topic_input == empty_fully_processed_keys
        ),
        "no_responses_lost_between_fully_processed_and_sentiment_input": (
            fully_processed_vs_sentiment_input["n_only_in_fully_processed"] == 0
        ),
        "no_responses_lost_between_sentiment_input_and_results": (
            sentiment_input_vs_results["n_only_in_sentiment_input"] == 0
        ),
        "root_and_output_copies_of_duplicated_files_are_byte_identical": all(
            entry.get("byte_identical") for entry in duplicate_files_report.values()
            if entry.get("root_exists") and entry.get("output_exists")
        ),
    }

    # --- Stage table ----------------------------------------------------------
    stage_table = [
        {
            "stage": "Raw transcripts (data/input/*.txt, re-parsed)",
            "n_interviews": raw_txt_summary["n_files_present"],
            "n_responses": raw_txt_summary["n_responses"],
            "n_sentences": None,
            "note": (
                f"Only {raw_txt_summary['n_files_present']}/30 raw .txt files present; "
                f"{len(raw_marker_dupes)} file(s) contain a repeated question marker "
                f"(pre-dict duplicate-key check)."
            ),
        },
        {
            "stage": "interviews.json (parsed raw responses)",
            "n_interviews": interviews_summary["n_interviews"],
            "n_responses": interviews_summary["n_responses"],
            "n_sentences": None,
            "note": f"{interviews_summary['n_empty_responses']} empty responses.",
        },
        {
            "stage": "fully_processed_ca.json (lemmatised/cleaned)",
            "n_interviews": fully_processed_summary["n_interviews"],
            "n_responses": fully_processed_summary["n_responses"],
            "n_sentences": None,
            "note": f"{fully_processed_summary['n_empty_responses']} empty responses (expected to disappear at the next stage).",
        },
        {
            "stage": "topic_modeling_input.json [data/output/, pipeline-wired copy]",
            "n_interviews": tm_summary["n_interviews"],
            "n_responses": tm_summary["n_responses_unique_keys"],
            "n_sentences": None,
            "note": (
                f"{tm_summary['n_documents_total']} list entries, "
                f"{tm_summary['n_duplicate_keys']} duplicate keys. "
                f"Drop vs. fully_processed is exactly the {len(empty_fully_processed_keys)} empty responses: "
                f"{assertions['topic_modeling_input_drops_are_EXACTLY_the_empty_fully_processed_responses']}."
            ),
        },
        {
            "stage": "sentiment_input_ca.json [data/output/, pipeline-wired copy]",
            "n_interviews": sentiment_input_summary["n_interviews"],
            "n_responses": sentiment_input_summary["n_responses"],
            "n_sentences": sentiment_input_summary["n_sentences"],
            "note": f"{sentiment_input_summary['n_empty_responses']} responses produced zero sentences.",
        },
        {
            "stage": "sentiment_results_ca.json [data/output/, pipeline-wired copy]",
            "n_interviews": sentiment_results_summary["n_interviews"],
            "n_responses": sentiment_results_summary["n_responses"],
            "n_sentences": sentiment_results_summary["n_sentences"],
            "note": (
                f"confused={sentiment_results_summary['confused']} "
                f"({sentiment_results_summary['confused_pct']}%), "
                f"not_confused={sentiment_results_summary['not_confused']}"
            ),
        },
        {
            "stage": "topic_results_bertopic.json (response-level topics)",
            "n_interviews": None,
            "n_responses": topic_summary["document_count_metadata"],
            "n_sentences": None,
            "note": (
                f"{topic_summary['n_named_topics']} named topics sum to "
                f"{topic_summary['assigned_total']}; implied outliers = "
                f"{topic_summary['outliers_implied_by_size_distribution']} "
                f"({topic_summary['outlier_percentage_reported']}% reported)."
            ),
        },
    ]

    report = {
        "stage_table": stage_table,
        "pipeline_wired_directory": PIPELINE_WIRED_DIR,
        "manuscript_reported_numbers": {
            "responses": 921,
            "sentences": 6073,
            "confused_pct": 73.2,
            "confused_count_figure5": 4442,
            "clear_count_figure5": 1629,
        },
        "duplicate_pipeline_files_on_disk": duplicate_files_report,
        "raw_txt_files_missing_from_package": [
            iid for iid in sorted({k[0] for k in interviews_summary["keys"]})
            if iid not in raw_txt_dict
        ],
        "raw_transcripts_with_duplicate_question_markers": raw_marker_dupes,
        "interviewer_or_boilerplate_contamination_check": contamination_report,
        "cross_stage_key_diffs": {
            "raw_txt_vs_interviews_json": {k: v for k, v in raw_vs_interviews.items() if not k.startswith("_")},
            "interviews_json_vs_fully_processed": {k: v for k, v in interviews_vs_fully_processed.items() if not k.startswith("_")},
            "fully_processed_vs_topic_modeling_input": {k: v for k, v in fully_processed_vs_topic_input.items() if not k.startswith("_")},
            "fully_processed_vs_sentiment_input": {k: v for k, v in fully_processed_vs_sentiment_input.items() if not k.startswith("_")},
            "sentiment_input_vs_sentiment_results": {k: v for k, v in sentiment_input_vs_results.items() if not k.startswith("_")},
        },
        "assertions": assertions,
        "all_assertions_passed": all(v is True for v in assertions.values()),
    }

    json_path = os.path.join(RESULTS_DIR, "data_audit_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # --- Markdown summary ------------------------------------------------------
    md_lines = ["# Data audit report", ""]
    md_lines.append(f"Pipeline-wired directory (what `terminal.py` actually reads/writes): `{PIPELINE_WIRED_DIR}`")
    md_lines.append("")
    md_lines.append("| Stage | Interviews | Responses | Sentences | Note |")
    md_lines.append("|---|---|---|---|---|")
    for row in stage_table:
        md_lines.append(
            f"| {row['stage']} | {row['n_interviews']} | {row['n_responses']} | "
            f"{row['n_sentences']} | {row['note']} |"
        )
    md_lines.append("")
    md_lines.append("## Assertions")
    for k, v in assertions.items():
        md_lines.append(f"- [{'x' if v else ' '}] {k}: **{v}**")
    md_lines.append("")
    md_lines.append("## Duplicate pipeline files on disk (data/*.json vs data/output/*.json)")
    md_lines.append(f"```json\n{json.dumps(duplicate_files_report, ensure_ascii=False, indent=2, default=str)}\n```")
    md_lines.append("")
    md_lines.append("## Manuscript-reported numbers (for reference)")
    md_lines.append("- 921 responses -> 6,073 sentences -> 73.2% confused (Figure 5: 4,442 confused + 1,629 clear)")
    md_lines.append("")
    md_lines.append("## Raw .txt files missing from the delivered package")
    md_lines.append(str(report["raw_txt_files_missing_from_package"]))
    md_lines.append("")
    md_lines.append("## Raw transcripts with a duplicated question marker (pre-dict check)")
    md_lines.append(f"```json\n{json.dumps(raw_marker_dupes, ensure_ascii=False, indent=2, default=str)}\n```")
    md_lines.append("")
    md_lines.append("## Interviewer / boilerplate contamination check")
    md_lines.append(f"```json\n{json.dumps(contamination_report, ensure_ascii=False, indent=2, default=str)}\n```")
    md_lines.append("")
    md_lines.append("## Key-level diffs (first 20 shown per side)")
    for name, d in report["cross_stage_key_diffs"].items():
        md_lines.append(f"### {name}")
        md_lines.append(f"```json\n{json.dumps(d, ensure_ascii=False, indent=2, default=str)}\n```")

    md_path = os.path.join(RESULTS_DIR, "data_audit_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n".join(md_lines[: len(stage_table) + 6]))
    print(f"\nAll assertions passed: {report['all_assertions_passed']}")
    for k, v in assertions.items():
        if not v:
            print(f"  FAILED: {k}")
    print(f"\nFull report written to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
