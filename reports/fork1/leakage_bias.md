# Leakage & Bias Audit

## Lineage

| Model | Public lineage evidence | Bias implication |
|---|---|---|
| SecureBERT-NER | Hugging Face identifies `CyberPeace-Institute/SecureBERT-NER` as SecureBERT fine-tuned on APTNER; the SecureBERT paper describes cybersecurity-domain pretraining. | Cyber-domain pretraining, APTNER fine-tuning, and a DNRTI-adjacent ontology all structurally favor SecureBERT. |
| CyNER | Hugging Face identifies `AI4Sec/cyner-xlm-roberta-base` as an XLM-R token-classification model; the CyNER paper describes a broader cyber NER library and coarser event ontology. | The DNRTI projection compresses several labels into each CyNER label and leaves a lower ceiling. |

## APTNER/DNRTI Overlap

`data/aptner/` is not present in this offline workspace, so 8-gram and sentence-hash overlap could not be computed. The overlap risk is unquantified, not proven.

## Mapping Handicap

| Model | Oracle recall | Oracle F1 | Expressible labels |
|---|---:|---:|---|
| SecureBERT | 0.9016 | 0.9483 | Area, Exp, HackOrg, Idus, OffAct, Org, SamFile, SecTeam, Time, Tool, Way |
| CyNER | 0.7376 | 0.8490 | Exp, HackOrg, Idus, OffAct, Org, SamFile, SecTeam, Tool, Way |

The raw PDF-mapping strict-F1 gap is 0.1785. The oracle ceiling gap is 0.0992, about 55.6% of the raw gap magnitude. Treating that ceiling difference as a conservative handicap leaves a bias-adjusted residual gap of 0.0793.

## Interpretation

SecureBERT is the better mapped-DNRTI choice, but the full raw gap should not be read as pure recognition capability. Direction 08 carries this caveat into the final verdict.
