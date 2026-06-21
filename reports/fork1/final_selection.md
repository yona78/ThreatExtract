# Final Model Selection

## Decision

Deploy **SecureBERT-NER** as the default offline on-premise model.

![Model comparison](figures/model_comparison_metrics.svg)

## Evidence

| Criterion | SecureBERT-NER | CyNER | Winner |
|---|---:|---:|---|
| Exact micro-F1 | 0.2823 | 0.1047 | SecureBERT |
| Exact recall | 0.3820 | 0.0928 | SecureBERT |
| Relaxed F1 | 0.5086 | 0.2604 | SecureBERT |
| Full-test elapsed seconds | 24.468 | 26.489 | SecureBERT |
| RSS peak MB | 699.46875 | 890.046875 | SecureBERT |
| Parameter count | 124085800 | 277461515 | SecureBERT |
| Cache bytes | 996220887 | 1124083139 | SecureBERT |

## Discussion

SecureBERT wins by a large margin on the primary scientific criterion, exact
micro-F1 on the full DNRTI test split. It also wins after relaxed matching,
which reduces the probability that the result is explained only by boundary
offsets. Operationally, SecureBERT is smaller and faster in this CPU run.

CyNER remains attractive as a simple five-label model, but that simplicity is
also the main weakness here. The DNRTI assignment requires label distinctions
that CyNER cannot express, notably `Time` and `Area`, and its broad
`Organization` and `System` labels blur several DNRTI classes.

## Deployment Recommendation

Package SecureBERT-NER in the offline Docker path. Keep the benchmark script
and cached-model workflow as a regression gate: if future fine-tuned models
are proposed, they should beat SecureBERT on exact F1, relaxed F1, and the
high-value classes before replacing it.

## Residual Risks

- Neither assigned model covers DNRTI `Purp` or `Features` under the PDF
  mapping.
- Energy is estimated, not directly measured.
- MPS latency should be rerun on the target M4 host.
