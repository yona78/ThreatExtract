# Direction 05 — Capability vs Bias

**Question:** How much of SecureBERT's lead is genuine recognition capability, and how much is structural advantage baked in before evaluation?
**Maps to:** RQ4 · Experiments **E6 (intrinsic) + E8 (leakage/bias)** · Implementation: appendix Phase 4 (Tasks 4.1–4.2).
**Why it matters for the choice:** This is the reviewer-grade question. SecureBERT-NER is cyber-pretrained **and** fine-tuned on **APTNER**, whose 21-label ontology is **near-identical to DNRTI** and whose source APT reports plausibly overlap DNRTI; CyNER is general-pretrained, fine-tuned on a 5-label Android-malware corpus. The PDF mapping is near-lossless for SecureBERT but compresses CyNER. A fair choice must separate skill from a stacked deck.

## Setup
- Models (frozen): both. Device: MPS for tokenizer probes (no model forward needed for most intrinsic metrics).
- **Preprocessing-invariant / intrinsic metrics (E6):** tokenizer fertility (subwords/word on DNRTI entity surfaces + a cyber probe list), domain single-token coverage, label **expressibility** (which DNRTI labels each model can emit at all), **oracle upper-bound F1** under the mapping (best achievable given expressibility), parameter count, cache size.
- **Bias/leakage audit (E8):** training-data lineage from model cards/papers; APTNER↔DNRTI overlap (8-gram + sentence-hash) **if `data/aptner/` is available offline**, else a documented qualitative argument; mapping-fairness handicap (oracle vs actual).

## What to do
1. Implement intrinsic metrics with math tests (e.g., CyNER cannot express `Time`/`Area`; oracle precision=1.0) (Task 4.1).
2. Compute fertility/coverage/oracle for both models; tabulate.
3. Document lineage; quantify or justify APTNER↔DNRTI overlap (Task 4.2).
4. Combine: state CyNER's structural handicap (max achievable F1) vs its actual F1.

## Outputs
- `reports/fork1/intrinsic_metrics.md`, `reports/fork1/leakage_bias.md` + a bias-chain figure.

## Findings (write from evidence)
- Fertility and domain coverage (E6): SecureBERT tokenizes DNRTI entity-surface words with lower fertility than CyNER (1.5058 vs 1.9395 subwords/word) and higher single-token coverage (0.7840 vs 0.4886). On the fixed cyber probe list, SecureBERT fertility is 2.5714 with 0.4286 single-token coverage; CyNER fertility is 3.4286 with 0.0000 single-token coverage. SecureBERT is smaller by parameter count (124,085,800 vs 277,461,515) and cache size in this local cache (950.1 MB vs 1,072.0 MB).
- Expressibility and oracle upper bound: SecureBERT can express 11 DNRTI labels under the PDF mapping (`Area`, `Exp`, `HackOrg`, `Idus`, `OffAct`, `Org`, `SamFile`, `SecTeam`, `Time`, `Tool`, `Way`) and cannot express `Purp`/`Features`; oracle recall 0.9016, oracle F1 0.9483, oracle FN 231. CyNER can express 9 labels (`Exp`, `HackOrg`, `Idus`, `OffAct`, `Org`, `SamFile`, `SecTeam`, `Tool`, `Way`) and cannot express `Area`, `Time`, `Purp`, `Features`; oracle recall 0.7376, oracle F1 0.8490, oracle FN 616.
- APTNER↔DNRTI overlap: `data/aptner/` is not available offline, so 8-gram/sentence-hash overlap was not computed. The overlap remains an unquantified risk, not a measured fact. The public model-card/paper lineage still creates a material bias concern: SecureBERT-NER is explicitly fine-tuned on APTNER after SecureBERT cybersecurity pretraining, while CyNER is an XLM-R token classifier with a coarse five-label ontology.
- Capability-vs-handicap decomposition: the actual PDF-mapping strict-F1 gap is +0.1785. The oracle F1 ceiling gap is +0.0992 in SecureBERT's favor, which is about 56% of the observed gap magnitude if treated as a ceiling handicap. SecureBERT also uses more of its oracle ceiling (0.2823 / 0.9483 = 29.8%) than CyNER (0.1038 / 0.8490 = 12.2%), so the remaining gap cannot be dismissed as expressibility alone.

## Conclusion → evidence contribution
- SecureBERT is both better positioned and better performing in the current DNRTI/PDF-mapping setup. The bias adjustment weakens the interpretation of the raw F1 gap as pure capability, but it does not erase SecureBERT's advantage because SecureBERT also has higher oracle utilization and better tokenizer/domain fit.
- A fairer head-to-head would need a shared ontology or two separate native-ontology evaluations, plus either APTNER data for overlap quantification or a DNRTI slice excluding labels unreachable by CyNER. The current forced mapping is product-relevant for this assignment, but not a clean capability-only benchmark.
- Direction 05 evidence contribution: SecureBERT leads for the mapped product task, with the strongest caveat so far: a substantial share of the lead is structural advantage from ontology/training lineage and should not be overclaimed as general NER superiority.
