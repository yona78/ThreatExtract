# Evidence Leader Summary

Evidence leader: **SecureBERT-NER**.

Evidence basis: full test split exact micro-F1.
Deployment selection is intentionally left to the reviewer/product owner.

## Headline Evidence

- Full-test exact micro-F1: `0.2823`.
- Full-test exact precision / recall: `0.2239` / `0.3820`.
- Full-test relaxed F1: `0.5086`.
- Inference elapsed on full test split: `24.140` seconds.

## Full-Test Comparison

| Model | Exact F1 | Exact P | Exact R | Relaxed F1 | Elapsed s | RSS peak MB |
|---|---:|---:|---:|---:|---:|---:|
| SecureBERT-NER | 0.2823 | 0.2239 | 0.3820 | 0.5086 | 24.140 | 812.765625 |
| CyNER | 0.1047 | 0.1201 | 0.0928 | 0.2604 | 27.400 | 948.53125 |

## Operational Fit

| Model | Parameters | Cached bytes |
|---|---:|---:|
| SecureBERT-NER | 124085800 | 996220887 |
| CyNER | 277461515 | 1124083139 |

## Caveats

- Neither assigned model covers DNRTI `Purp` or `Features` under the PDF
  mapping.
- Energy is estimated, not directly measured.
- MPS latency should be rerun on the target M4 host.
- The benchmark treats CyberNER-derived checkpoints as contaminated because
  CyberNER includes DNRTI by construction.
