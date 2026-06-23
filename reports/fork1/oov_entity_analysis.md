# Seen vs Unseen Entity Surface Analysis

Entity surfaces are case-folded and whitespace-normalized before comparing train entity strings with test entity strings.

| Model | Surface status | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| securebert | seen | 2017 | 835 | 699 | 1182 | 0.5443 | 0.4140 | 0.4703 |
| securebert | unseen | 331 | 62 | 2411 | 269 | 0.0251 | 0.1873 | 0.0442 |
| cyner | seen | 2017 | 176 | 244 | 1841 | 0.4190 | 0.0873 | 0.1444 |
| cyner | unseen | 331 | 40 | 1355 | 291 | 0.0287 | 0.1208 | 0.0463 |
