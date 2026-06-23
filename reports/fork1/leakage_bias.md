# Leakage & Bias Audit

## Lineage

| Model | Public lineage evidence | Bias implication |
|---|---|---|
| SecureBERT-NER | The Hugging Face card states that `CyberPeace-Institute/SecureBERT-NER` is fine-tuned with SecureBERT on the APTNER dataset, and links SecureBERT (`arXiv:2204.02685`) plus APTNER (`IEEE 9776031`). SecureBERT's paper describes a cybersecurity language model trained on a large cybersecurity corpus with a customized tokenizer. Sources: [model card](https://huggingface.co/CyberPeace-Institute/SecureBERT-NER), [SecureBERT paper](https://arxiv.org/abs/2204.02685). | Stacked advantage: cyber-domain pretraining, cybersecurity tokenizer, APTNER fine-tuning, and a label space close to DNRTI. |
| CyNER | The Hugging Face card identifies `AI4Sec/cyner-xlm-roberta-base` as an XLM-RoBERTa token-classification model; its README content is empty. The CyNER paper describes a library combining transformer NER, IOC heuristics, and generic NER models for cybersecurity extraction, with event classes from MALOnt/MALOnt2.0. Sources: [model card](https://huggingface.co/AI4Sec/cyner-xlm-roberta-base), [CyNER paper](https://arxiv.org/abs/2204.05754). | Broader multilingual base and much coarser ontology; the DNRTI projection compresses several DNRTI labels into each CyNER label. |

## APTNER/DNRTI Overlap

`data/aptner/` is not present in this offline workspace, so 8-gram and sentence-hash overlap could not be computed. The overlap risk remains unquantified, not proven. The risk is still material because SecureBERT-NER is explicitly fine-tuned on APTNER and the APTNER ontology/source type is close to DNRTI threat-report NER.

## Mapping Handicap

Intrinsic metrics from `reports/fork1/intrinsic_metrics.jsonl`:

| Model | Actual strict F1 | Expressible labels | Oracle recall | Oracle F1 | Oracle FN |
|---|---:|---|---:|---:|---:|
| SecureBERT | 0.2823 | Area, Exp, HackOrg, Idus, OffAct, Org, SamFile, SecTeam, Time, Tool, Way | 0.9016 | 0.9483 | 231 |
| CyNER | 0.1038 | Exp, HackOrg, Idus, OffAct, Org, SamFile, SecTeam, Tool, Way | 0.7376 | 0.8490 | 616 |

The oracle F1 ceiling gap is 0.0992 in SecureBERT's favor. The actual PDF-mapping strict-F1 gap is 0.1785, so the mapping/expressibility ceiling difference is large enough to explain about 56% of the observed gap magnitude if treated as a ceiling handicap. This is not an additive causal decomposition, but it shows that a substantial part of SecureBERT's advantage is available before any recognition quality is measured.

SecureBERT also uses more of its oracle ceiling: 0.2823 / 0.9483 = 29.8%. CyNER uses 0.1038 / 0.8490 = 12.2%. That remaining utilization gap points to either stronger recognition capability, more compatible training data, better label projection behavior, or all three.

## Interpretation

SecureBERT is both better positioned and better performing under the current DNRTI/PDF-mapping setup. The evidence does not support treating the raw 0.1785 F1 gap as pure model capability. The fair reading is: SecureBERT is the operationally better choice for this mapped DNRTI task, but the study must carry a structural-bias caveat because CyNER's ontology ceiling and unique-label coverage are much lower.
