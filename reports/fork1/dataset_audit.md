# DNRTI Dataset Audit

Source repository: https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence

| Split | Sentences | Tokens | Labeled BIO tokens | Collapsed spans | Malformed lines |
|---|---:|---:|---:|---:|---:|
| test | 664 | 17716 | 3606 | 2348 | 16 |

## Label Counts

### test

| Label | Spans |
|---|---:|
| Area | 216 |
| Exp | 132 |
| Features | 116 |
| HackOrg | 369 |
| Idus | 129 |
| OffAct | 150 |
| Org | 137 |
| Purp | 115 |
| SamFile | 248 |
| SecTeam | 152 |
| Time | 169 |
| Tool | 315 |
| Way | 100 |

## Parsing Notes

- The raw files are token/tag rows with blank lines between sentences.
- Gold spans are reconstructed over a normalized single-space sentence string.
- Tag-only `O` lines are skipped and counted as malformed source rows.
- Published DNRTI entity totals often refer to labeled BIO tokens; this audit also reports collapsed entity spans for strict span matching.

## First 25 Warnings

- `test.txt:4734: skipped tag-only line 'O'`
- `test.txt:5938: skipped tag-only line 'O'`
- `test.txt:6515: skipped tag-only line 'O'`
- `test.txt:7520: skipped tag-only line 'O'`
- `test.txt:10729: skipped tag-only line 'O'`
- `test.txt:13961: skipped tag-only line 'O'`
- `test.txt:13964: skipped tag-only line 'O'`
- `test.txt:14027: skipped tag-only line 'O'`
- `test.txt:14666: skipped tag-only line 'O'`
- `test.txt:14732: skipped tag-only line 'O'`
- `test.txt:14741: skipped tag-only line 'O'`
- `test.txt:15281: skipped tag-only line 'O'`
- `test.txt:16007: skipped tag-only line 'O'`
- `test.txt:16010: skipped tag-only line 'O'`
- `test.txt:16073: skipped tag-only line 'O'`
- `test.txt:17265: skipped tag-only line 'O'`
