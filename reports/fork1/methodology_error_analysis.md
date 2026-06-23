# Methodology Error Analysis

Buckets use strict entity matching on the headline sentence-level, original-casing, no-normalization protocol.

## Bucket Counts

| Model | Bucket | Count |
|---|---|---:|
| cyner | boundary_error | 301 |
| cyner | spurious_fp | 903 |
| cyner | strict_correct | 216 |
| cyner | strict_drop_fn | 1436 |
| cyner | type_error | 395 |
| securebert | boundary_error | 696 |
| securebert | spurious_fp | 2216 |
| securebert | strict_correct | 897 |
| securebert | strict_drop_fn | 557 |
| securebert | type_error | 198 |

## Example Rows

| Model | Bucket | Sample | Gold | Predicted | Gold text | Predicted text |
|---|---|---|---|---|---|---|
| securebert | boundary_error | test-00000 | HackOrg | HackOrg | StoneDrill | Stone |
| securebert | strict_correct | test-00000 | SecTeam | SecTeam | Kaspersky | Kaspersky |
| securebert | strict_correct | test-00000 | HackOrg | HackOrg | Shamoon | Shamoon |
| securebert | spurious_fp | test-00000 | SPURIOUS | HackOrg |  | Dr |
| securebert | spurious_fp | test-00000 | SPURIOUS | HackOrg |  | ill groups |
| securebert | spurious_fp | test-00000 | SPURIOUS | HackOrg |  | groups |
| securebert | boundary_error | test-00001 | HackOrg | HackOrg | BlueNoroff | Nor |
| securebert | strict_correct | test-00001 | SecTeam | SecTeam | Kaspersky | Kaspersky |
| securebert | spurious_fp | test-00001 | SPURIOUS | HackOrg |  | Blue |
| securebert | spurious_fp | test-00001 | SPURIOUS | HackOrg |  | off |
| securebert | boundary_error | test-00002 | SecTeam | SecTeam | Eset‍ | E |
| securebert | strict_drop_fn | test-00002 | Area | O/MISSED | Russian |  |
| securebert | strict_drop_fn | test-00002 | HackOrg | O/MISSED | turla |  |
| securebert | boundary_error | test-00003 | SecTeam | SecTeam | Eset | E |
| securebert | strict_correct | test-00003 | HackOrg | HackOrg | Turla | Turla |
| securebert | boundary_error | test-00004 | HackOrg | HackOrg | NewsBeef | Be |
| securebert | strict_correct | test-00004 | Area | Area | SA | SA |
| securebert | strict_drop_fn | test-00005 | HackOrg | O/MISSED | COMMENT PANDA |  |
| securebert | type_error | test-00009 | HackOrg | Tool | PUTTER PANDA | PUT |
| securebert | strict_drop_fn | test-00010 | Tool | O/MISSED | dropper |  |
| securebert | type_error | test-00011 | Tool | OffAct|Way | RC4 | 4 |
| securebert | strict_drop_fn | test-00011 | Tool | O/MISSED | dropper |  |
| securebert | type_error | test-00012 | SamFile | OffAct|Way | Word document | Word |
| securebert | type_error | test-00032 | Tool | HackOrg | RTM | RTM |
| securebert | type_error | test-00036 | Tool | OffAct|Way | RC4 | 4 |
| cyner | strict_correct | test-00000 | SecTeam | SecTeam | Kaspersky | Kaspersky |
| cyner | type_error | test-00000 | HackOrg | Tool | StoneDrill | StoneDrill |
| cyner | type_error | test-00000 | HackOrg | Tool | Shamoon | Shamoon |
| cyner | strict_correct | test-00001 | SecTeam | SecTeam | Kaspersky | Kaspersky |
| cyner | type_error | test-00001 | HackOrg | Tool | BlueNoroff | BlueNoroff |
| cyner | boundary_error | test-00002 | SecTeam | SecTeam | Eset‍ | Eset |
| cyner | type_error | test-00002 | HackOrg | Tool | turla | tur |
| cyner | strict_drop_fn | test-00002 | Area | O/MISSED | Russian |  |
| cyner | strict_correct | test-00003 | SecTeam | SecTeam | Eset | Eset |
| cyner | boundary_error | test-00003 | HackOrg | HackOrg | Turla | Tur |
| cyner | spurious_fp | test-00003 | SPURIOUS | HackOrg|Idus|Org|SecTeam |  | la |
| cyner | strict_correct | test-00004 | SecTeam | SecTeam | Kaspersky | Kaspersky |
| cyner | type_error | test-00004 | HackOrg | Tool | NewsBeef | NewsBeef |
| cyner | strict_drop_fn | test-00004 | Area | O/MISSED | SA |  |
| cyner | strict_drop_fn | test-00005 | Area | O/MISSED | European |  |
| cyner | spurious_fp | test-00005 | SPURIOUS | Tool |  | ixen Panda |
| cyner | spurious_fp | test-00005 | SPURIOUS | Tool |  | MENT PANDA |
| cyner | boundary_error | test-00006 | SecTeam | SecTeam | CrowdStrike | Crow |
| cyner | strict_drop_fn | test-00006 | Area | O/MISSED | Shanghai |  |
| cyner | strict_drop_fn | test-00006 | Area | O/MISSED | China |  |
| cyner | spurious_fp | test-00006 | SPURIOUS | HackOrg|Idus|Org|SecTeam |  | dStrike |
| cyner | boundary_error | test-00008 | SecTeam | SecTeam | CrowdStrike Intelligence | Crow |
| cyner | strict_correct | test-00008 | SecTeam | SecTeam | CrowdStrike | CrowdStrike |
| cyner | spurious_fp | test-00008 | SPURIOUS | HackOrg|Idus|Org|SecTeam |  | dStrike Intelligence |
| cyner | boundary_error | test-00013 | HackOrg | HackOrg | PUTTER PANDA | P |
