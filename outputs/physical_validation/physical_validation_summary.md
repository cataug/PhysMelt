# PhysMelt physical validation

## Synchronization quality

- Intensity Pearson correlation: 0.999760
- Intensity Spearman correlation: 0.999677
- Intensity MAE: 0.012890 DL
- Area Spearman correlation: 0.9819

## Out-of-fold external validation

| Model | P(high)-power Spearman | Part-balanced Spearman | Monotonic parts | High-low power gap | Switch reduction |
|---|---:|---:|---:|---:|---:|
| kmeans | 0.625 | 0.515 | 98.2% | 35.11 W | 38.8% |
| gmm | 0.612 | 0.502 | 100.0% | 35.16 W | 37.8% |

## GMM temporal regime profiles

| Regime | Fraction | Power (W) | Area (mm²) | Length (mm) | Width (mm) | Intensity |
|---|---:|---:|---:|---:|---:|---:|
| low | 33.7% | 155.14 | 0.01396 | 0.1709 | 0.1074 | 3.797 |
| nominal | 42.7% | 169.21 | 0.01808 | 0.2181 | 0.1119 | 5.159 |
| high | 23.6% | 182.68 | 0.02296 | 0.2641 | 0.1172 | 6.484 |
