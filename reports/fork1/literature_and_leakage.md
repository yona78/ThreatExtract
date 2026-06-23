# Literature Review And Leakage Analysis

## Research Questions

1. Were the assigned checkpoints trained on DNRTI?
2. Do related corpora create leakage or overlap risk?
3. How should leakage risk change interpretation of the benchmark?

## Findings

- `CyberPeace-Institute/SecureBERT-NER` is documented as SecureBERT fine-tuned on APTNER,
  not DNRTI. The base SecureBERT model was pretrained on broad cybersecurity
  text, so raw public-report overlap remains possible, but supervised DNRTI
  exposure was not found.
- `AI4Sec/cyner-xlm-roberta-base` exposes CyNER's five-label taxonomy: `System`,
  `Organization`, `Vulnerability`, `Malware`, and `Indicator`. Public CyNER
  descriptions point to a separate Android-malware CTI corpus, not DNRTI.
- CyberNER is not a valid replacement model for this benchmark unless DNRTI
  is explicitly held out, because CyberNER harmonizes DNRTI together with
  CyNER, APTNER, and Attacker.

## Leakage Risk Table

| Artifact | DNRTI supervised exposure | Risk level | Interpretation |
|---|---|---|---|
| SecureBERT-NER | No public evidence found | Medium | APTNER domain transfer; possible raw-report overlap. |
| CyNER HF checkpoint | No public evidence found | Medium | Separate CTI corpus; broad taxonomy hurts direct DNRTI comparability. |
| CyberNER-derived models | Yes, by construction | High | Contaminated unless DNRTI is held out. |

## Discussion

The correct reading of this experiment is cross-dataset transfer under a
forced taxonomy projection. That is exactly the deployment setting: the chosen
model must work on threat-intelligence text it was not explicitly fine-tuned
for and must emit entities useful to an offline product. The benchmark should
not be presented as an in-domain supervised DNRTI leaderboard.

## Sources

- SecureBERT-NER Hugging Face: https://huggingface.co/CyberPeace-Institute/SecureBERT-NER
- SecureBERT paper: https://arxiv.org/abs/2204.02685
- CyNER paper: https://arxiv.org/abs/2204.05754
- CyNER Hugging Face checkpoint: https://huggingface.co/AI4Sec/cyner-xlm-roberta-base
- DNRTI repository: https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence
- CyberNER paper: https://arxiv.org/abs/2510.26499
