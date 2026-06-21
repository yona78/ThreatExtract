# Final Model Selection

Decision: deploy **securebert**.

Selection basis: full test split exact micro-F1.

## Headline Evidence

- Full-test exact micro-F1: `0.2823`.
- Full-test exact precision / recall: `0.2239` / `0.3820`.
- Full-test relaxed F1: `0.5086`.
- Inference elapsed on full test split: `24.468` seconds.

## Full-Test Comparison

| Model | Exact F1 | Exact P | Exact R | Relaxed F1 | Elapsed s | RSS peak MB |
|---|---:|---:|---:|---:|---:|---:|
| cyner | 0.1047 | 0.1201 | 0.0928 | 0.2604 | 26.489 | 890.046875 |
| securebert | 0.2823 | 0.2239 | 0.3820 | 0.5086 | 24.468 | 699.46875 |

## Operational Fit

| Model | Parameters | Cached bytes | Load seconds |
|---|---:|---:|---:|
| cyner | 277461515 | 1124083139 | 0.472 |
| securebert | 124085800 | 996220887 | 0.153 |

## Caveats

- This run used CPU because the current process reported `mps_available=false`.
  On an Apple M4 host with MPS exposed, rerun with `--device mps`.
- DNRTI `Purp` and `Features` cannot be emitted by either assigned model under
  the PDF mapping, so downstream product requirements for those labels would
  require an extra model, rules, or fine-tuning.
- The benchmark treats CyberNER-derived checkpoints as contaminated because
  CyberNER includes DNRTI by construction.
