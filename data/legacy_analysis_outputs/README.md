# Legacy pre-NLLB downstream analysis outputs

These five files were previously at `data/output/`:

- `sentiment_results_ca.json`
- `topic_results_bertopic.json`
- `topic_results_lda.json`
- `topic_results_nmf.json`
- `topic_results_svd.json`

They are historical **Stage 3/5 output** -- topic modeling and sentiment
analysis results computed against the pipeline's pre-NLLB, Catalan-sourced
content (from `full_preprocess()` / `translate_to_catalan()`), not against
the new NLLB-based Step 2. `topic_results_bertopic.json` in particular is
one of the "PRESERVED" files referenced in `RUN_EVALUATION_ONLY.txt` from an
earlier, separate manuscript-evaluation task (the 930-document LDA/NMF/SVD-
at-K=10 comparison against the saved BERTopic result) -- kept here for that
reason, not deleted.

They were moved out of `data/output/` (not deleted) because `terminal.py`'s
menu option "4. Visualize results" (`_visualize_results()`) only checks
whether files named exactly `sentiment_results_ca.json` and
`topic_results_*.json` exist in `data/output/` -- it has no freshness or
Step-2-validity gating of its own (unlike `_analyze_topics()` /
`_analyze_sentiment()`, which check the `step2_bridge_metadata.json`
sidecar before trusting `data/output/`'s bridge-input files). Left in
place at that exact path, "4. Visualize results" would have happily
rendered these old, pre-NLLB figures immediately after unzipping this
project -- before Step 2, or Stage 3/5, had ever run under the new
pipeline -- a real bug, caught in review before this delivery.

This is a data-relocation fix, not a code change: once these files are out
of `data/output/`, running "4. Visualize results" against a freshly
unzipped/cloned project correctly reports "No sentiment analysis results
found" / "No topic analysis results found" until a real Stage 3/5 run
(itself gated on a `STEP2_VALID=True` Step 2 run via the bridge-freshness
sidecar) populates `data/output/` again, for real, under the new pipeline.
See `STEP2_NLLB_CHANGES.md` for the full explanation, including a noted
possible future improvement (not implemented here, matching how the
bridge-freshness sidecar was handled): `_visualize_results()` could
additionally check the same `step2_bridge_metadata.json` sidecar as
defense in depth, the same way `_analyze_topics()` / `_analyze_sentiment()`
already do, rather than relying solely on `data/output/` being clean.

If you specifically need the old pre-NLLB topic/sentiment results (e.g. for
the manuscript comparison), they're here, unchanged.

See also `data/legacy_pre_nllb_step2_outputs/README.md` for the three
related **Step-2-bridge-input** files (`topic_modeling_input.json`,
`sentiment_input_ca.json`, `fully_processed_ca.json`) that were moved out
of `data/output/` in the previous revision for the same underlying reason
-- those are pre-NLLB Step 2 *inputs* to Stage 3/5; these five are Stage
3/5 *outputs*. Kept in separate folders since they are different files or
serving different pipeline stages, not because the underlying problem
(stale pre-NLLB content sitting where the live pipeline looks for current
content) is any different.
