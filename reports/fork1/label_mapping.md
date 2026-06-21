# Assignment Label Mapping And Taxonomy Analysis

## Mapping Extracted From The PDF

The assignment requires comparing models with incompatible taxonomies. The
benchmark therefore projects model outputs into DNRTI labels before scoring.

| CyNER | DNRTI | SecureBERT-NER | Consequence |
|---|---|---|---|
| Organization | HackOrg, SecTeam, Idus, Org | APT, SECTEAM, IDTY | CyNER cannot distinguish actor, security team, identity, and organization. |
| System | OffAct, Way | ACT, OS, TOOL | The broad `System` label creates ambiguity for behavior labels. |
| Vulnerability | Exp | VULID, VULNAME | Both models can target exploit/vulnerability mentions. |
| Malware | Tool | MAL | DNRTI `Tool` is malware/tooling in assignment terms. |
| Indicator | SamFile | FILE | File indicators are comparable. |
| Indicator | no DNRTI label | DOM, ENCR, IP, URL, MD5, PROT, EMAIL, SHA1, SHA2 | These SecureBERT IOCs are operationally useful but not credited by DNRTI. |
| no CyNER label | Time | TIME | SecureBERT can score timeline entities; CyNER cannot. |
| no CyNER label | Area | LOC | SecureBERT can score geography; CyNER cannot. |
| no CyNER label | Purp, Features | no SecureBERT label | Neither model can directly cover these classes. |

![Per-label exact F1](figures/per_label_exact_f1.svg)

## Discussion

This mapping is the central experimental design choice. It is not a clerical
detail: it changes the decision. SecureBERT benefits from finer APTNER labels
for `TIME`, `LOC`, `SECTEAM`, and `IDTY`, while CyNER's five-label ontology
compresses several DNRTI concepts into `Organization` or `System`. That makes
CyNER harder to use when downstream workflows require specific threat-actor,
geography, and identity slots.

The benchmark deliberately does not credit SecureBERT's IOC labels (`IP`,
`URL`, hashes, domains, and email) unless the PDF maps them to a DNRTI
label. Those predictions may be valuable in production, but including them
as true positives would violate the assignment evaluation contract.
