# 06 — Model Selection: Which NER Model to Use

**Decision: adopt `CyberPeace-Institute/SecureBERT-NER` as the production NER model for the
on-prem threat-intelligence product. Confidence: high.**

SecureBERT-NER wins on every axis that matters for this product — recognition quality, per-class
coverage, stability, and deployment cost — and the win is statistically significant and survives
a conservative bias adjustment. CyNER is not competitive for DNRTI-style output under the required
label mapping. This document lays out the full evidence and the conditions under which the decision
should be revisited.

> **Reading the numbers in this report.** Two complementary views are used:
> - **Token-level F1** (seqeval, on each model's expressible labels) — the best estimate of *real
>   recognition capability*, free of sub-word boundary artifacts. **Use this for capability claims.**
> - **Entity-level strict char-span F1** — a deliberately strict, boundary-exact lower bound.
>
> A code fix changed the inference aggregation from `simple` to `first` (see §7); the entity-level
> *absolute* numbers below are the conservative pre-fix figures and will rise after a full
> `make reproduce`. **The model ranking is identical under both views and is not affected by the
> fix.**

---

## 1. The recommendation in one table

| Criterion | SecureBERT-NER | CyNER | Winner |
|---|---|---|---|
| Token-level F1 (expressible labels) | **0.730** | 0.334 | SecureBERT |
| Entity strict F1 (conservative bound) | **0.282** | 0.104 | SecureBERT |
| Entity type F1 (boundaries relaxed) | **0.501** | 0.248 | SecureBERT |
| Strict-F1 gap (95% CI) | **+0.179 [0.152, 0.207]** | — | SecureBERT (significant) |
| DNRTI labels expressible | **11 / 13** | 9 / 13 | SecureBERT |
| Oracle recall ceiling | **0.902** | 0.738 | SecureBERT |
| Parameters | **124M** | 277M | SecureBERT |
| Peak RSS | **408 MB** | 1564 MB | SecureBERT |
| p50 latency @256 tok, batch 1 (CPU) | **115.6 ms** | 129.2 ms | SecureBERT |
| Load time | **0.12 s** | 0.41 s | SecureBERT |
| Cross-config head-to-head (strict) | **90 wins / 7 ties / 0 losses** | — | SecureBERT |

There is no axis on which CyNER wins. The decision is not close.

---

## 2. Recognition capability

The realistic, boundary-artifact-free capability estimate (token-level, on labels each model can
express):

| Model | Token-level F1 | Eval spans |
|---|---:|---:|
| **SecureBERT-NER** | **0.730** | 1,601 |
| CyNER | 0.334 | 695 |

Same direction, larger magnitude, than the conservative entity-level bound (0.282 vs 0.104). The
gap is ~2.2× either way. Under entity-level scoring with all four SemEval schemes, SecureBERT leads
on every one and no confidence interval crosses zero:

| Scheme | SecureBERT F1 | CyNER F1 | Gap | 95% CI |
|---|---:|---:|---:|---|
| strict | 0.2823 | 0.1038 | +0.1785 | [0.1518, 0.2069] |
| exact | 0.2999 | 0.1581 | +0.1419 | [0.1127, 0.1729] |
| partial | 0.4230 | 0.2710 | +0.1520 | [0.1259, 0.1799] |
| type | 0.5013 | 0.2484 | +0.2530 | [0.2230, 0.2840] |

The `type` gap (+0.25) is the most telling: even ignoring boundaries, SecureBERT assigns the right
DNRTI class roughly twice as often. CyNER's dominant failure is **recall** — it misses 1,436 of
2,348 gold spans versus SecureBERT's 557.

---

## 3. Per-class coverage (the operational reason)

Per-label strict metrics (false-positive counting corrected — see §7; recall reflects the
conservative pre-fix predictions and will improve after re-run). Read this as *relative class
difficulty within each model*.

| DNRTI label | SecureBERT P / R / F1 | CyNER P / R / F1 |
|---|---|---|
| Time | 0.685 / 0.746 / **0.714** | — / — / **0.000** (cannot express) |
| Area | 0.615 / 0.704 / **0.657** | — / — / **0.000** (cannot express) |
| Org | 0.901 / 0.730 / **0.806** | 0.706 / 0.088 / 0.156 |
| Idus | 0.490 / 0.760 / **0.596** | 0.800 / 0.031 / 0.060 |
| SecTeam | 0.482 / 0.691 / **0.568** | 0.705 / 0.566 / **0.628** |
| Way | 0.549 / 0.500 / **0.524** | 0.429 / 0.030 / 0.056 |
| HackOrg | 0.094 / 0.138 / 0.112 | 0.172 / 0.084 / 0.113 |
| Tool | 0.126 / 0.276 / 0.173 | 0.057 / 0.146 / 0.082 |
| SamFile | 0.116 / 0.306 / 0.168 | 0.063 / 0.056 / 0.060 |
| Exp | 0.015 / 0.061 / 0.024 | 0.078 / 0.152 / 0.103 |
| OffAct | 0.099 / 0.293 / 0.148 | 0.000 / 0.000 / 0.000 |
| Purp / Features | 0 (neither model maps these) | 0 |

Two product-critical facts:

1. **CyNER structurally scores 0 on `Time` and `Area`** — its five-label taxonomy has no mapping to
   them. For an incident-timeline / affected-geography product these are exactly the slots you want.
   SecureBERT covers both well (F1 0.71 / 0.66).
2. **CyNER's only competitive class is `SecTeam`** (0.63). Everywhere else it is weak or absent.
   SecureBERT has five classes at F1 ≥ 0.52 and the only label SecureBERT loses outright (`Exp`) is
   a minor one.

---

## 4. Stability — the ranking does not flip

The SecureBERT win is not a one-configuration fluke. Across the full sweeps (details in
[doc 03](03_robustness_sensitivity_and_subsets.md)):

- **Preprocessing** (detok, context, max-length, normalization, alignment): no full-split config flips the ranking. The only knobs that move *absolute* scores a lot (alignment leniency, lower-casing) move both models together.
- **Protocol** (PDF-mapping vs paper-native): gap barely moves (−0.005).
- **Robustness** (defang / refang / random-case / keyboard-typo): SecureBERT keeps a positive gap under all four. Random casing hurts both badly — normalize/monitor casing upstream.
- **Subset size & strategy** (5 strategies × 5 sizes × 3 seeds): from 50 sentences onward every strategy reproduces the win with a CI excluding zero. Only ≤10-sentence subsets are statistical ties (underpowered).

Aggregate: **90 wins / 7 ties / 0 losses** across strict head-to-head configurations.

---

## 5. Operational fit (on-prem Docker)

SecureBERT is also the cheaper model to ship — it is smaller, lighter, and faster:

| Metric | SecureBERT-NER | CyNER |
|---|---:|---:|
| Parameters | 124M | 277M |
| Hugging Face cache | 996 MB | 1124 MB |
| Peak RSS | **408 MB** | 1564 MB |
| Load time | 0.12 s | 0.41 s |
| p50 latency @256 tok (CPU, batch 1) | 115.6 ms | 129.2 ms |
| Full test-set inference | 24.1 s | 27.4 s |

The ~3.8× RSS difference (408 MB vs 1.56 GB) directly lowers the minimum customer hardware and the
container memory ceiling. Tokenizer intrinsics explain part of the speed and accuracy edge:
SecureBERT keeps 43% of cyber probe terms as single tokens; CyNER's XLM-R tokenizer keeps 0% and
fragments them ~3.4×. See [doc 05](05_operational_and_calibration.md) and
[doc 04](04_leakage_bias_and_intrinsics.md).

---

## 6. Honest caveats (why this isn't "SecureBERT is 7× better")

- **Part of the gap is structural, not skill.** The DNRTI label projection and SecureBERT's APTNER
  fine-tuning lineage both favor it. Removing the taxonomy-ceiling advantage (oracle gap 0.099 of
  the raw 0.179) leaves a **+0.079 bias-adjusted residual** — still positive, so SecureBERT remains
  the better *mapped-DNRTI* choice, but don't market the full raw gap as pure model quality. See
  [doc 04](04_leakage_bias_and_intrinsics.md).
- **Leakage is unquantified, not disproven.** APTNER/DNRTI raw-text overlap couldn't be measured
  offline (`data/aptner/` absent). Treat the result as cross-dataset transfer, not an in-domain
  leaderboard.
- **Neither model covers `Purp` or `Features`**, and SecureBERT over-predicts (high recall, lower
  precision with sub-word fragments pre-fix). Both are downstream-fixable (post-filtering,
  confidence gating); CyNER's low recall and type confusion are not.
- **Raw confidence is unusable for triage** for both models (ECE 0.65 / 0.70, no high-precision
  threshold). Add calibration (temperature/isotonic) + an abstention policy before exposing
  confidence to analysts. See [doc 05](05_operational_and_calibration.md).

None of these reverse the decision; they shape how the product should *present* and *post-process*
SecureBERT's output.

---

## 7. What changed in the code (bug fixes behind this report)

Three evaluation bugs were fixed in `src/fork1`:

1. **Sub-word fragmentation (root cause of deflated scores).** `runner.py` used
   `aggregation_strategy="simple"`, which emitted sub-word fragments (`StoneDrill`→`Stone`,
   `CrowdStrike`→`Crow`) that strict scoring punished as boundary errors + spurious FPs (92% of
   SecureBERT boundary errors were intra-word fragments; 67% of its "spurious" FPs were ≤3-char
   fragments). Changed to `aggregation_strategy="first"` (whole-word labels). Expected effect: the
   entity-level absolute F1 rises substantially toward the token-level figures (0.73 / 0.33);
   ranking unchanged.
2. **Per-label false-positive over-counting.** `metrics.py` charged one FP to *every* mapped DNRTI
   label for one-to-many projections, inflating CyNER's (and SecureBERT's `Org`/`Way`) FPs up to
   4×. Now charged once per prediction. This corrected, e.g., SecureBERT `Org` precision
   0.50→0.90 and CyNER `SecTeam` precision 0.34→0.71 (§3).
3. **First-overlap → best-overlap matching.** Predictions now bind to the maximally overlapping
   gold span, not the first one encountered.

New unit tests cover (2) and (3); the existing suite stays green.

> **TODO (run on the dev machine, not the sandbox).** Run `make reproduce` (venv + optional MPS) to
> regenerate every sweep's *absolute* numbers under the fixed aggregation. The conclusions in this
> document are derived from the unchanged-ranking evidence plus the deterministic recomputations
> that don't require re-running the models; only the entity-level absolute magnitudes are pending
> that refresh.

---

## 8. When to revisit this decision

Choose differently only if one of these becomes true:

- **The product's label set collapses to CyNER's coarse taxonomy** (just Organization / Malware /
  Vulnerability / Indicator / System) *and* multilingual input matters — then CyNER's XLM-R base
  could be worth re-testing on its native labels.
- **A DNRTI-supervised or DNRTI-adjacent checkpoint becomes available** that is verifiably not
  contaminated (note: CyberNER-derived models include DNRTI by construction and are disqualified
  unless DNRTI is explicitly held out).
- **The APTNER/DNRTI overlap is measured and found high**, which would shrink the capability claim
  (the operational and coverage advantages would still stand).

Otherwise: **ship SecureBERT-NER**, add casing normalization and confidence calibration upstream,
and post-filter its IOC labels per the product's needs.
