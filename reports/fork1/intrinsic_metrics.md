# Intrinsic Metrics

| Model | Entity fertility | Probe fertility | Probe 1-token coverage | Parameters | Cache MB |
|---|---:|---:|---:|---:|---:|
| securebert | 1.5058 | 2.5714 | 0.4286 | 124085800 | 950.1 |
| cyner | 1.9395 | 3.4286 | 0.0000 | 277461515 | 1072.0 |

## Oracle upper bound

| Model | Oracle precision | Oracle recall | Oracle F1 | Expressible labels |
|---|---:|---:|---:|---|
| securebert | 1.0000 | 0.9016 | 0.9483 | Area, Exp, HackOrg, Idus, OffAct, Org, SamFile, SecTeam, Time, Tool, Way |
| cyner | 1.0000 | 0.7376 | 0.8490 | Exp, HackOrg, Idus, OffAct, Org, SamFile, SecTeam, Tool, Way |
