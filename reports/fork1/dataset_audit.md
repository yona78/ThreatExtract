# DNRTI Dataset Audit

## Summary

The authoritative dataset source is the user-provided GitHub repository.
The archive contains `train.txt`, `valid.txt`, and `test.txt` files in a
CoNLL-like token/tag format. Evaluation uses the published `test.txt` split
without resampling for the final claim.

![DNRTI label distribution](figures/dnrti_label_distribution.svg)

| Split | Sentences | Tokens | Labeled BIO tokens | Collapsed spans | Malformed lines |
|---|---:|---:|---:|---:|---:|
| test | 664 | 17716 | 3606 | 2348 | 16 |

## Label Counts

| Label | Gold spans | Share |
|---|---:|---:|
| HackOrg | 369 | 15.7% |
| Tool | 315 | 13.4% |
| SamFile | 248 | 10.6% |
| Area | 216 | 9.2% |
| Time | 169 | 7.2% |
| SecTeam | 152 | 6.5% |
| OffAct | 150 | 6.4% |
| Org | 137 | 5.8% |
| Exp | 132 | 5.6% |
| Idus | 129 | 5.5% |
| Features | 116 | 4.9% |
| Purp | 115 | 4.9% |
| Way | 100 | 4.3% |

## Data Quality Notes

- The source contains occasional tag-only `O` rows. The parser skips these
  rows and counts them as malformed source lines rather than treating `O`
  as a literal token.
- Published DNRTI entity totals often count labeled BIO tokens. This report
  distinguishes labeled BIO tokens from collapsed spans because strict NER
  evaluation is span based.
- The test split is imbalanced: `HackOrg`, `Tool`, and `SamFile` dominate
  support, while `Way`, `Purp`, and `Features` have low but nontrivial
  support. This is why per-label analysis is necessary.

## Evaluation Implication

A single micro-F1 number hides whether a model is useful for the product.
For example, detecting `Time` and `Area` may matter for incident timelines
and affected geography even if these labels are not the most frequent.
