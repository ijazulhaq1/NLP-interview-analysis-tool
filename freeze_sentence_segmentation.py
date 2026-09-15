#!/usr/bin/env python3
"""Round 7, step 5: freeze the source-side sentence-ID structure for the
full 950-response corpus, using segment_source_sentences() (the canonical
split_sentences_strict + merge_ellipsis_continuations path) -- the same
function the eventual one-sentence-per-translation redesign will call.

No NLLB, no translation. This produces the reference sentence segmentation
+ IDs that translation will be built on top of, and that response is
reconstructed from later (response text = join of its sentence texts, in
order).
"""
import json
import os
import re
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import preprocess_v2  # noqa: E402
from preprocessing import Preprocessor  # noqa: E402
from preprocess_v2 import (  # noqa: E402
    compute_interview_primary_language,
    determine_response_language,
    segment_source_sentences,
)

RAW_PATH = os.path.join(_PROJECT_ROOT, "data", "input", "interviews.json")
OUT_PATH = os.path.join(_PROJECT_ROOT, "data", "frozen_source_sentence_segmentation_v1.json")


def main():
    raw_data = json.load(open(RAW_PATH, encoding="utf-8"))
    preprocessor = Preprocessor()

    responses = {}
    all_merge_log = []
    seen_sentence_ids = set()
    duplicate_ids = []
    total_sentences = 0

    for interview_id, questions in raw_data.items():
        primary_lang, primary_detail = compute_interview_primary_language(questions)
        for question_id, raw_text in questions.items():
            response_id = f"{interview_id}::{question_id}"
            if not raw_text or not raw_text.strip():
                continue
            source_language, method, _ = determine_response_language(raw_text, primary_lang)

            sentences, merge_log = segment_source_sentences(raw_text, source_language, preprocessor)

            sentence_records = []
            for idx, sent_text in enumerate(sentences):
                sentence_id = f"{response_id}::s{idx:03d}"
                if sentence_id in seen_sentence_ids:
                    duplicate_ids.append(sentence_id)
                    continue
                seen_sentence_ids.add(sentence_id)
                sentence_records.append({
                    "sentence_id": sentence_id,
                    "sentence_index": idx,
                    "text_source": sent_text,
                })
                total_sentences += 1

            for m in merge_log:
                all_merge_log.append({"response_id": response_id, **m})

            responses[response_id] = {
                "response_id": response_id,
                "interview_id": interview_id,
                "question_id": question_id,
                "source_language": source_language,
                "language_detection_method": method,
                "original_text": raw_text,
                "sentence_count": len(sentence_records),
                "sentence_ids": [r["sentence_id"] for r in sentence_records],
                "sentences": sentence_records,
            }

    assert not duplicate_ids, f"duplicate sentence IDs generated: {duplicate_ids}"

    # Reconstruction check: joining a response's frozen sentence texts with
    # a single space must reproduce (modulo whitespace normalization) the
    # same lexical content as the original raw response text -- this is
    # the structural guarantee the whole redesign depends on.
    reconstruction_word_mismatches = []
    for response_id, r in responses.items():
        original_words = re.findall(r"[^\W\d_]+|\d+", r["original_text"], flags=re.UNICODE)
        reconstructed_text = " ".join(s["text_source"] for s in r["sentences"])
        reconstructed_words = re.findall(r"[^\W\d_]+|\d+", reconstructed_text, flags=re.UNICODE)
        if original_words != reconstructed_words:
            reconstruction_word_mismatches.append(response_id)

    result = {
        # 2, not 1: input_sha256 below became a required field so main()
        # can hard-verify the frozen structure still matches the corpus it
        # was generated from -- see preprocess_v2.
        # validate_frozen_segmentation_against_input(). A schema_version 1
        # file (written before that check existed) has no input_sha256 at
        # all and main() now refuses to run against one.
        "schema_version": preprocess_v2.FROZEN_SEGMENTATION_SCHEMA_VERSION,
        "generated_from": "data/input/interviews.json",
        "input_sha256": preprocess_v2._hash_file(RAW_PATH),
        "segmentation_function": "preprocess_v2.segment_source_sentences (split_sentences_strict + merge_ellipsis_continuations)",
        "response_count": len(responses),
        "total_sentence_count": total_sentences,
        "ellipsis_continuation_merges": all_merge_log,
        "responses": responses,
    }
    json.dump(result, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 78)
    print("SENTENCE-ID FREEZE -- source-side segmentation")
    print("=" * 78)
    print(f"Responses:                {len(responses)}")
    print(f"Total frozen sentences:   {total_sentences}")
    print(f"Duplicate IDs:            {len(duplicate_ids)} (must be 0)")
    print(f"Total merges applied:     {len(all_merge_log)} (expect 38)")
    print(f"Reconstruction mismatches: {len(reconstruction_word_mismatches)} (must be 0)")
    print(f"Schema version:           {result['schema_version']}")
    print(f"Input SHA-256:            {result['input_sha256']}")
    print(f"Output written to:        {OUT_PATH}")
    print(f"Output file size:         {__import__('os').path.getsize(OUT_PATH):,} bytes")

    if reconstruction_word_mismatches:
        print("MISMATCHES:", reconstruction_word_mismatches[:10])


if __name__ == "__main__":
    main()
