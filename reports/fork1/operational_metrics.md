# Operational Metrics

This report records deployment-relevant metrics for offline/on-prem selection.

- Device requested: `auto`
- Cache directory: `model_cache/fork1`
- Estimated energy constant: `18.0` watts

## Model Footprint

| Model | Cache bytes | RSS before load MB | RSS after load MB |
|---|---:|---:|---:|
| securebert | 996220887 | 239.109375 | 292.125 |
| cyner | 1124083139 | 699.46875 | 699.46875 |

## Inference Runs

| Model | Samples | Elapsed s | RSS peak MB | Energy J | Docs/s | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| securebert | 10 | 0.987 | 636.375 | 17.768 | 10.131 | 229.964 |
| securebert | 100 | 3.666 | 666.3125 | 65.992 | 27.276 | 711.359 |
| securebert | 664 | 24.468 | 699.46875 | 440.426 | 27.137 | 724.045 |
| cyner | 10 | 1.032 | 834.9375 | 18.572 | 9.692 | 220.012 |
| cyner | 100 | 4.709 | 848.8125 | 84.768 | 21.235 | 553.796 |
| cyner | 664 | 26.489 | 890.046875 | 476.801 | 25.067 | 668.807 |

Energy values are estimates unless collected with hardware counters such as
`powermetrics` on macOS. The benchmark keeps this explicit so energy is not
mistaken for a direct measurement.
