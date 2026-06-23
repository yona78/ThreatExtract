# Fork 1 Report Inventory

## Primary Evidence

- `benchmark_summary.md` is the paper-style synthesis.
- `methodology_baseline.md` is the headline strict/partial/type/exact scorer report.
- `methodology_per_label_strict.jsonl` is the full per-class strict PRF table.
- `methodology_confusion_matrix.jsonl` is the entity-level DNRTI confusion matrix.
- `methodology_error_analysis.md` and `.jsonl` contain strict-correct, type-error, boundary-error, missed, and spurious examples.
- `oov_entity_analysis.md` and `.jsonl` compare train-seen vs test-unseen entity surfaces.
- `master_table.md` and `.jsonl` aggregate cross-experiment model rows.

## Supporting Evidence

- `preprocessing_sensitivity.md`, `subset_study.md`, `protocol_comparison.md`, `intrinsic_metrics.md`, `operational_envelope.md`, `calibration.md`, and `robustness.md` are direction-level reports.
- `final_selection.md` is an evidence-leader summary only; final product model selection is intentionally left to the reviewer/product owner.

## Future TODOs

- Rerun E9 robustness after the perturbation RNG fix. The code is fixed, but the long robustness regeneration was intentionally skipped in this cleanup pass.
- Run full `make reproduce` when time allows to regenerate every report from one end-to-end command.
- Rerun CPU/MPS accuracy parity on an Apple host where `torch.backends.mps.is_available()` is true.
- Continue unifying legacy `benchmark.py` report terminology with the newer `src/fork1/run_experiment.py` pipeline.
- Decide whether to promote `normalization="none"` or `normalization="nfkc"` as the formal final preprocessing standard; current headline evidence uses sentence-level, original casing, no lowercasing, no document context, and strict raw span scoring.
