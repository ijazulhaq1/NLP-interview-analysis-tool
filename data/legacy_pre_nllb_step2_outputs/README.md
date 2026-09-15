# Legacy pre-NLLB Step 2 bridge outputs

These three files were previously at `data/output/`:

- `topic_modeling_input.json`
- `sentiment_input_ca.json`
- `fully_processed_ca.json`

They are historical outputs from the pipeline **before** the NLLB redesign
(Catalan-sourced content, built by the old `translate_to_catalan()` /
`full_preprocess()` path). `topic_modeling_input.json` in particular is
called out as "PRESERVED" in `RUN_EVALUATION_ONLY.txt` from an earlier,
separate manuscript-evaluation task (the 930-document LDA/NMF/SVD-at-K=10
comparison against the saved BERTopic result) -- kept here for that
reason, not deleted.

They were moved out of `data/output/` (not deleted) because
`terminal.py`'s `_analyze_topics()` / `_analyze_sentiment()` check for
files at exactly those three names in `data/output/` before deciding
whether to run Step 2. Leaving stale copies at that exact path would have
let `2. Analyze topics` or `3. Analyze sentiment` run against this
historical, pre-NLLB content without ever executing the new Step 2 -- a
real bug, caught in review before this delivery. `terminal.py` now also
checks a metadata sidecar (`step2_bridge_metadata.json`, written only by a
`STEP2_VALID=True` run of the current pipeline) before trusting any file
at those paths, as defense in depth -- but keeping stale copies out of the
live path in the first place is the simpler, more obviously-correct fix.
See `STEP2_NLLB_CHANGES.md` for the full explanation.

If you specifically need the old 930-document manuscript comparison
inputs, they're here, unchanged.
