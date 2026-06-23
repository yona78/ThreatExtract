# Direction 04 — Protocol Faithfulness

**Question:** Does the winner change when each model is scored the way its own authors intended, rather than under the assignment's forced mapping?
**Maps to:** RQ3 · Experiment **E3** · Implementation: appendix Phase 3 (Task 3.1).
**Why it matters for the choice:** SecureBERT-NER (APTNER) and CyNER both report **seqeval entity-level micro-F1, BIO, max_len 128**. Evaluating under a foreign protocol can unfairly penalize a model. Reproducing each paper's native protocol tests whether our verdict is an artifact of the assignment's mapping/preprocessing or a genuine result.

## Setup
- Models (frozen): both. Device: MPS.
- Protocols compared:
  - **PDF-mapping** preset (assignment's projection; the default everywhere else).
  - **Paper-native** preset: punctuation-aware detok, per-sentence context, **max_len 128**, overlap alignment, seqeval-strict scoring.
- Cross-check: on the 1:1-mappable labels, our strict-F1 must match `seqeval` strict within ±0.005 (validates the instrument against the field standard).

## What to do
1. Run both models under `PRESETS["pdf_mapping"]` and `PRESETS["paper_native"]` (Task 3.1).
2. Build per-token BIO via alignment for the seqeval cross-check on 1:1 labels.
3. Report ranking + strict-F1 + gap + CI under each protocol; quantify how much the gap moves.
4. Note labels that are **unscorable** under paper-native protocol due to one-to-many mapping (report as coverage, not silent drop).

## Outputs
- `reports/fork1/protocol_comparison.md`.

## Findings (write from evidence)
- Side-by-side protocol comparison is in `reports/fork1/protocol_comparison.md` and `reports/fork1/protocol_comparison.jsonl`. Under PDF-mapping, SecureBERT strict-F1 was 0.2823 and CyNER was 0.1038, gap +0.1785 with 95% CI [0.1518, 0.2069], flip `no`. Under the paper-native preset (`punct_aware`, sentence context, max_len 128, overlap), SecureBERT strict-F1 was 0.2767 and CyNER was 0.1030, gap +0.1736 with 95% CI [0.1481, 0.2011], flip `no`.
- Gap movement from PDF-mapping to paper-native was -0.0049. Both models moved slightly down under the paper-native preset, but the head-to-head interval remained far from 0.
- Seqeval cross-check on the unique-label BIO subset passed with delta 0.0000 for every protocol/model pair. PDF-mapping unique-label F1 was SecureBERT 0.7300 and CyNER 0.3341; paper-native unique-label F1 was SecureBERT 0.7024 and CyNER 0.2969. These are token-aligned unique-label checks, not the headline raw-span strict scores.
- Unique-label coverage is asymmetric. SecureBERT has 1,601/2,348 uniquely scorable gold spans (68.19% coverage); 747 spans are not uniquely covered: `Features` 116, `Idus` 129, `OffAct` 150, `Org` 137, `Purp` 115, `Way` 100. CyNER has only 695/2,348 uniquely scorable gold spans (29.60% coverage); 1,653 spans are not uniquely covered: `Area` 216, `Features` 116, `HackOrg` 369, `Idus` 129, `OffAct` 150, `Org` 137, `Purp` 115, `SecTeam` 152, `Time` 169, `Way` 100.

## Conclusion → evidence contribution
- The ranking survives the paper-native preset. SecureBERT remains ahead with a CI-excluding-0 gap under both protocols, and no flip flag is raised.
- The assignment mapping is not the source of the direction-level ranking: changing to the paper-native preprocessing/max-length preset narrows the gap by only 0.0049. The larger protocol caveat is coverage: CyNER's ontology can be scored on a much smaller unique-label subset than SecureBERT's, which reinforces the taxonomy-mismatch disadvantage measured elsewhere rather than explaining it away.
- Direction 04 evidence contribution: SecureBERT leads, with a protocol caveat that paper-native/token-BIO validation is clean but CyNER has low uniquely scorable DNRTI coverage.
