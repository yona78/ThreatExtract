# Dataset Size Scaling Effects

The benchmark evaluates deterministic paired subsets so model comparisons use
the same DNRTI sentences at each size.

| Model | Subset | Samples | Exact F1 | Exact P | Exact R | Relaxed F1 |
|---|---|---:|---:|---:|---:|---:|
| cyner | 10 | 10 | 0.0303 | 0.0286 | 0.0323 | 0.2424 |
| cyner | 100 | 100 | 0.1180 | 0.1324 | 0.1065 | 0.2721 |
| cyner | all | 664 | 0.1047 | 0.1201 | 0.0928 | 0.2604 |
| securebert | 10 | 10 | 0.1782 | 0.1286 | 0.2903 | 0.3960 |
| securebert | 100 | 100 | 0.2423 | 0.1855 | 0.3491 | 0.4887 |
| securebert | all | 664 | 0.2823 | 0.2239 | 0.3820 | 0.5086 |

## Interpretation

- The 10-sentence subset is useful only as a smoke test; the ranking is noisy.
- The 100-sentence and full-test results agree on the winner.
- Exact and relaxed matching preserve the same ranking, which reduces the
  chance that the result is only a boundary-tokenization artifact.
