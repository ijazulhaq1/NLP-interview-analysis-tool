# Data audit report

Pipeline-wired directory (what `terminal.py` actually reads/writes): `data/output`

| Stage | Interviews | Responses | Sentences | Note |
|---|---|---|---|---|
| Raw transcripts (data/input/*.txt, re-parsed) | 27 | 838 | None | Only 27/30 raw .txt files present; 1 file(s) contain a repeated question marker (pre-dict duplicate-key check). |
| interviews.json (parsed raw responses) | 30 | 950 | None | 0 empty responses. |
| fully_processed_ca.json (lemmatised/cleaned) | 30 | 950 | None | 20 empty responses (expected to disappear at the next stage). |
| topic_modeling_input.json [data/output/, pipeline-wired copy] | 30 | 930 | None | 930 list entries, 0 duplicate keys. Drop vs. fully_processed is exactly the 20 empty responses: True. |
| sentiment_input_ca.json [data/output/, pipeline-wired copy] | 30 | 950 | 6284 | 0 responses produced zero sentences. |
| sentiment_results_ca.json [data/output/, pipeline-wired copy] | 30 | 950 | 6284 | confused=5154 (82.0%), not_confused=1130 |
| topic_results_bertopic.json (response-level topics) | None | 930 | None | 10 named topics sum to 597; implied outliers = 333 (35.8% reported). |

## Assertions
- [x] sentiment_results_partition_holds (confused+not_confused+missing == total): **True**
- [x] topic_assigned_plus_outliers_equals_document_count: **True**
- [x] topic_modeling_input_doc_count_matches_stats_field: **True**
- [x] no_duplicate_keys_in_topic_modeling_input_documents_list: **True**
- [ ] no_duplicate_question_markers_in_any_raw_transcript: **False**
- [x] nonempty_fully_processed_count_matches_topic_modeling_input_count: **True**
- [x] topic_modeling_input_drops_are_EXACTLY_the_empty_fully_processed_responses: **True**
- [x] no_responses_lost_between_fully_processed_and_sentiment_input: **True**
- [x] no_responses_lost_between_sentiment_input_and_results: **True**
- [ ] root_and_output_copies_of_duplicated_files_are_byte_identical: **False**

## Duplicate pipeline files on disk (data/*.json vs data/output/*.json)
```json
{
  "topic_modeling_input.json": {
    "root_path": "data/topic_modeling_input.json",
    "output_path": "data/output/topic_modeling_input.json",
    "root_exists": true,
    "output_exists": true,
    "root_size_bytes": 539705,
    "output_size_bytes": 505880,
    "byte_identical": false,
    "root_summary": {
      "n_documents": 931,
      "generated_at": "2025-05-21T12:22:55.768620"
    },
    "output_summary": {
      "n_documents": 930,
      "generated_at": "2025-05-22T13:55:16.645528"
    }
  },
  "sentiment_input_ca.json": {
    "root_path": "data/sentiment_input_ca.json",
    "output_path": "data/output/sentiment_input_ca.json",
    "root_exists": true,
    "output_exists": true,
    "root_size_bytes": 680523,
    "output_size_bytes": 672903,
    "byte_identical": false,
    "root_summary": {
      "n_responses": 950,
      "n_sentences": 6528
    },
    "output_summary": {
      "n_responses": 950,
      "n_sentences": 6284
    }
  },
  "sentiment_results_ca.json": {
    "root_path": "data/sentiment_results_ca.json",
    "output_path": "data/output/sentiment_results_ca.json",
    "root_exists": true,
    "output_exists": true,
    "root_size_bytes": 5199977,
    "output_size_bytes": 5026925,
    "byte_identical": false,
    "root_summary": {
      "n_responses": 950,
      "n_sentences": 6528,
      "confused": 4822,
      "not_confused": 1706,
      "confused_pct": 73.9
    },
    "output_summary": {
      "n_responses": 950,
      "n_sentences": 6284,
      "confused": 5154,
      "not_confused": 1130,
      "confused_pct": 82.0
    }
  }
}
```

## Manuscript-reported numbers (for reference)
- 921 responses -> 6,073 sentences -> 73.2% confused (Figure 5: 4,442 confused + 1,629 clear)

## Raw .txt files missing from the delivered package
['n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt', 'n11_AardborstSpain_231013_Mobil_timestamps_corrected_PP.txt', 'n17_RamaderiaSegre_221223_Mobil_timestamps_corrected_PP.txt']

## Raw transcripts with a duplicated question marker (pre-dict check)
```json
{
  "n06_Cincaporc_230726_Gravadora_timestamps_corrected_PP.txt": [
    "4.2"
  ]
}
```

## Interviewer / boilerplate contamination check
```json
{
  "threshold_used": 0.25,
  "short_sentence_max_tokens": 4,
  "n_questions_flagged": 11,
  "n_questions_total": 53,
  "flagged": [
    {
      "question_id": "4.1a",
      "phrase": "sí",
      "n_interviews_with_phrase": 1,
      "n_interviews_with_question": 1,
      "coverage": 1.0
    },
    {
      "question_id": "4.1b",
      "phrase": "bueno",
      "n_interviews_with_phrase": 1,
      "n_interviews_with_question": 1,
      "coverage": 1.0
    },
    {
      "question_id": "2.3c",
      "phrase": "ens queden dos telediaris",
      "n_interviews_with_phrase": 1,
      "n_interviews_with_question": 1,
      "coverage": 1.0
    },
    {
      "question_id": "6.5",
      "phrase": "això depèn",
      "n_interviews_with_phrase": 1,
      "n_interviews_with_question": 2,
      "coverage": 0.5
    },
    {
      "question_id": "2.3",
      "phrase": "sí",
      "n_interviews_with_phrase": 3,
      "n_interviews_with_question": 8,
      "coverage": 0.38
    },
    {
      "question_id": "2.1",
      "phrase": "sí",
      "n_interviews_with_phrase": 10,
      "n_interviews_with_question": 30,
      "coverage": 0.33
    },
    {
      "question_id": "6.11",
      "phrase": "atenció",
      "n_interviews_with_phrase": 1,
      "n_interviews_with_question": 3,
      "coverage": 0.33
    },
    {
      "question_id": "3.3",
      "phrase": "sí",
      "n_interviews_with_phrase": 1,
      "n_interviews_with_question": 3,
      "coverage": 0.33
    },
    {
      "question_id": "6.12",
      "phrase": "no",
      "n_interviews_with_phrase": 4,
      "n_interviews_with_question": 13,
      "coverage": 0.31
    },
    {
      "question_id": "6.12b",
      "phrase": "no",
      "n_interviews_with_phrase": 4,
      "n_interviews_with_question": 15,
      "coverage": 0.27
    },
    {
      "question_id": "6.12a",
      "phrase": "no",
      "n_interviews_with_phrase": 4,
      "n_interviews_with_question": 16,
      "coverage": 0.25
    }
  ],
  "caveat": "No speaker diarisation exists anywhere in this pipeline. A flagged phrase here is NOT proof of interviewer speech — it is a cheap recurrence signal worth a manual look, nothing more."
}
```

## Key-level diffs (first 20 shown per side)
### raw_txt_vs_interviews_json
```json
{
  "in_raw_txt_not_interviews_json": [
    [
      "n08_Friselva_230727_Gravadora_timestamps_corrected_PP.txt",
      "2.1a"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "1.3"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "2.2"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "3.1"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "6.12a"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "6.2"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "6.4"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "6.5"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "1.3"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "2.2"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "3.1"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "6.2"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "6.5"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "6.8"
    ]
  ],
  "in_interviews_json_not_raw_txt": [
    [
      "n08_Friselva_230727_Gravadora_timestamps_corrected_PP.txt",
      "2.2a"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "1.1"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "1.2"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "1.3a"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "2.1"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "2.2a"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "2.2b"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "3.1"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "3.2"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "3.3a"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "3.3b"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "3.4"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "4.1"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "4.2"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "4.3"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "5.1"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "5.2"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "5.3"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "6.1"
    ],
    [
      "n09_Miquelo_230925_Gravadora_timestamps_corrected_PP.txt",
      "6.10"
    ],
    "... (106 more)"
  ],
  "n_only_in_raw_txt": 14,
  "n_only_in_interviews_json": 126
}
```
### interviews_json_vs_fully_processed
```json
{
  "in_interviews_json_not_fully_processed": [],
  "in_fully_processed_not_interviews_json": [],
  "n_only_in_interviews_json": 0,
  "n_only_in_fully_processed": 0
}
```
### fully_processed_vs_topic_modeling_input
```json
{
  "in_fully_processed_not_topic_modeling_input": [
    [
      "n07_lloret_230726_Mobil_timestamps_corrected_PP.txt",
      "1.3b"
    ],
    [
      "n12_RBS_231013_Mobil_timestamps_corrected_PP.txt",
      "6.12a"
    ],
    [
      "n12_RBS_231013_Mobil_timestamps_corrected_PP.txt",
      "6.3"
    ],
    [
      "n13_saura_231023_Mobil_timestamps_corrected_PP.txt",
      "6.2a"
    ],
    [
      "n14_Daniel_231218_Gravadora_timestamps_corrected_PP.txt",
      "1.3b"
    ],
    [
      "n17_RamaderiaSegre_221223_Mobil_timestamps_corrected_PP.txt",
      "3.1a"
    ],
    [
      "n17_RamaderiaSegre_221223_Mobil_timestamps_corrected_PP.txt",
      "3.4"
    ],
    [
      "n17_RamaderiaSegre_221223_Mobil_timestamps_corrected_PP.txt",
      "6.7"
    ],
    [
      "n18_Matges_231227_Pc_timestamps_corrected_PP.txt",
      "5.1"
    ],
    [
      "n21_TorretaPecuaria_240104_timestamps_corrected_PP.txt",
      "3.1"
    ],
    [
      "n21_TorretaPecuaria_240104_timestamps_corrected_PP.txt",
      "6.12"
    ],
    [
      "n22_Andrimner_240108_Pc_timestamps_corrected_PP.txt",
      "5.3"
    ],
    [
      "n23_Marcal_240117_Pc_timestamps_corrected_PP.txt",
      "6.2a"
    ],
    [
      "n26_Felbert_240119_Pc_timestamps_corrected_PP.txt",
      "5.3"
    ],
    [
      "n26_Felbert_240119_Pc_timestamps_corrected_PP.txt",
      "6.2a"
    ],
    [
      "n29_FinquesMontclar_240208_Gravadora_timestamps_corrected_PP.txt",
      "1.3b"
    ],
    [
      "n29_FinquesMontclar_240208_Gravadora_timestamps_corrected_PP.txt",
      "5.3"
    ],
    [
      "n29_FinquesMontclar_240208_Gravadora_timestamps_corrected_PP.txt",
      "6.8"
    ],
    [
      "n31_RamaderiesEroles_240214_Gravadora_timestamps_corrected_PP.txt",
      "1.3b"
    ],
    [
      "n31_RamaderiesEroles_240214_Gravadora_timestamps_corrected_PP.txt",
      "6.12b"
    ]
  ],
  "in_topic_modeling_input_not_fully_processed": [],
  "n_only_in_fully_processed": 20,
  "n_only_in_topic_modeling_input": 0
}
```
### fully_processed_vs_sentiment_input
```json
{
  "in_fully_processed_not_sentiment_input": [],
  "in_sentiment_input_not_fully_processed": [],
  "n_only_in_fully_processed": 0,
  "n_only_in_sentiment_input": 0
}
```
### sentiment_input_vs_sentiment_results
```json
{
  "in_sentiment_input_not_sentiment_results": [],
  "in_sentiment_results_not_sentiment_input": [],
  "n_only_in_sentiment_input": 0,
  "n_only_in_sentiment_results": 0
}
```