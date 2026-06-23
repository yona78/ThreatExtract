# Protocol Comparison

| Protocol | Model | Strict F1 | Gap | 95% CI | Flip? |
|---|---|---:|---:|---|---|
| pdf_mapping | securebert | 0.2823 | 0.1785 | [0.1518, 0.2069] | no |
| pdf_mapping | cyner | 0.1038 | -0.1785 | [-0.2069, -0.1518] | no |
| paper_native | securebert | 0.2767 | 0.1736 | [0.1481, 0.2011] | no |
| paper_native | cyner | 0.1030 | -0.1736 | [-0.2011, -0.1481] | no |

## Gap Movement

SecureBERT gap movement from PDF-mapping to paper-native: -0.0049.

## Seqeval Cross-Check

| Protocol | Model | Our unique-label F1 | Seqeval F1 | Delta | Unique gold spans |
|---|---|---:|---:|---:|---:|
| pdf_mapping | securebert | 0.7300 | 0.7300 | 0.0000 | 1601 |
| pdf_mapping | cyner | 0.3341 | 0.3341 | 0.0000 | 695 |
| paper_native | securebert | 0.7024 | 0.7024 | 0.0000 | 1601 |
| paper_native | cyner | 0.2969 | 0.2969 | 0.0000 | 695 |

## Unique-Label Coverage

| Model | Total gold spans | Unique-label spans | Non-unique spans | Coverage | Non-unique labels |
|---|---:|---:|---:|---:|---|
| securebert | 2348 | 1601 | 747 | 0.6819 | Features:116, Idus:129, OffAct:150, Org:137, Purp:115, Way:100 |
| cyner | 2348 | 695 | 1653 | 0.2960 | Area:216, Features:116, HackOrg:369, Idus:129, OffAct:150, Org:137, Purp:115, SecTeam:152, Time:169, Way:100 |
