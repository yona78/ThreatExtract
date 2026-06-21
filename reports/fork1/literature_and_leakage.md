# Literature And Leakage Review

## Sources

- SecureBERT-NER Hugging Face: CyberPeace-Institute/SecureBERT-NER
- SecureBERT paper: https://arxiv.org/abs/2204.02685
- CyNER paper: https://arxiv.org/abs/2204.05754
- CyNER Hugging Face checkpoint: AI4Sec/cyner-xlm-roberta-base
- DNRTI repository: https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence
- CyberNER paper: https://arxiv.org/abs/2510.26499

## Leakage Assessment

- SecureBERT-NER is publicly documented as SecureBERT fine-tuned on APTNER,
  not DNRTI. This is a cross-dataset transfer evaluation, not an in-domain
  supervised DNRTI test.
- CyNER is documented as trained on a separate Android-malware CTI corpus with
  five labels: Malware, Indicator, System, Organization, and Vulnerability.
  The assigned Hugging Face checkpoint exposes exactly those labels.
- No public source found during this run says either assigned checkpoint was
  supervised on DNRTI. Raw-text pretraining/source overlap remains possible
  because all corpora draw from public CTI reports.
- CyberNER explicitly harmonizes CyNER, DNRTI, APTNER, and Attacker into a
  shared STIX-style corpus. Any CyberNER-trained checkpoint is therefore
  contaminated for an independent DNRTI benchmark unless DNRTI is held out.

## Interpretation

Metrics from this benchmark should be read as transfer plus taxonomy alignment
performance. The label mapping can dominate the score, especially because DNRTI
`Purp` and `Features` have no mapped output in either assigned model, and CyNER
cannot natively emit `Time` or `Area` while SecureBERT can.
