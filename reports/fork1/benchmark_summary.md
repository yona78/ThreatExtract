# Benchmark Summary

## Scope

- SecureBERT-NER model: `CyberPeace-Institute/SecureBERT-NER`
- CyNER model: `AI4Sec/cyner-xlm-roberta-base`
- DNRTI source: https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence
- Split: `test`
- Device requested: `auto`
- Offline mode: `True`

## Dataset

| Split | Sentences | Tokens | Labeled BIO tokens | Collapsed spans |
|---|---:|---:|---:|---:|
| test | 664 | 17716 | 3606 | 2348 |

## Model Load Metadata

| Model | Device | Load seconds | Parameters | Labels |
|---|---|---:|---:|---|
| securebert | cpu | 0.153 | 124085800 | ACT, APT, DOM, EMAIL, ENCR, FILE, IDTY, IP, LOC, MAL, MD5, OS, PROT, SECTEAM, SHA1, SHA2, TIME, TOOL, URL, VULID, VULNAME |
| cyner | cpu | 0.472 | 277461515 | Indicator, Malware, Organization, System, Vulnerability |

## Results

| Model | Samples | Repeat | Exact F1 | Exact P | Exact R | Relaxed F1 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| securebert | 10 | 0 | 0.1782 | 0.1286 | 0.2903 | 0.3960 | 0.0321 | 0.6383 |
| securebert | 100 | 0 | 0.2423 | 0.1855 | 0.3491 | 0.4887 | 0.0286 | 0.0384 |
| securebert | 664 | 0 | 0.2823 | 0.2239 | 0.3820 | 0.5086 | 0.0291 | 0.0340 |
| cyner | 10 | 0 | 0.0303 | 0.0286 | 0.0323 | 0.2424 | 0.0331 | 0.6477 |
| cyner | 100 | 0 | 0.1180 | 0.1324 | 0.1065 | 0.2721 | 0.0330 | 0.0559 |
| cyner | 664 | 0 | 0.1047 | 0.1201 | 0.0928 | 0.2604 | 0.0293 | 0.0362 |

## Selection Rule

Prefer the model with the best exact micro-F1 on supported DNRTI labels, then
break ties by recall on high-value CTI classes (`HackOrg`, `SecTeam`, `Exp`,
`Tool`, `SamFile`) and operational footprint. Penalize models that cannot
represent assignment-required labels, especially `Time`, `Area`, `Purp`, and
`Features` if those labels matter to the product.
