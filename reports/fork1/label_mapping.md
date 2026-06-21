# Assignment Label Mapping

This file encodes the class mapping table extracted from the assignment PDF.
Mappings are many-to-many. Metrics therefore report exact span matches after
projecting model labels into DNRTI labels, plus coverage gaps for DNRTI labels
that a model cannot represent.

| CyNER | DNRTI | SecureBERT-NER |
|---|---|---|
| CyNER Organization | HackOrg, SecTeam, Idus, Org | SecureBERT APT, SECTEAM, IDTY |
| CyNER System | OffAct, Way | SecureBERT ACT, OS, TOOL |
| CyNER Vulnerability | Exp | SecureBERT VULID, VULNAME |
| CyNER Malware | Tool | SecureBERT MAL |
| CyNER Indicator | SamFile | SecureBERT FILE |
| CyNER Indicator | no DNRTI label | SecureBERT DOM, ENCR, IP, URL, MD5, PROT, EMAIL, SHA1, SHA2 |
| no CyNER label | Time | SecureBERT TIME |
| no CyNER label | Area | SecureBERT LOC |
| no CyNER label | Purp, Features | no SecureBERT label |

Important consequence: SecureBERT `TOOL` is mapped to DNRTI `OffAct`/`Way` by
the assignment table, while DNRTI `Tool` is mapped to SecureBERT `MAL`.
