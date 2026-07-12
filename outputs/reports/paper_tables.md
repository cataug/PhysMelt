# PhysMelt experiment summary

| Features | Model | Silhouette | DB ↓ | Confidence | Separation | Switch reduction |
|---|---|---:|---:|---:|---:|---:|
| appearance | anchored_gmm | 0.457 | 0.702 | 0.982 | 2.412 | 25.1% |
| appearance | gmm | 0.458 | 0.699 | 0.982 | 2.421 | 24.9% |
| appearance | kmeans | 0.462 | 0.684 | 0.893 | 2.516 | 36.4% |
| basic | anchored_gmm | 0.486 | 0.637 | 0.921 | 2.463 | 38.4% |
| basic | gmm | 0.497 | 0.632 | 0.922 | 2.452 | 39.9% |
| basic | kmeans | 0.500 | 0.619 | 0.911 | 2.502 | 41.5% |
| physical | anchored_gmm | 0.226 | 1.395 | 0.986 | 2.270 | 23.8% |
| physical | gmm | 0.214 | 1.418 | 0.987 | 2.202 | 22.8% |
| physical | kmeans | 0.281 | 1.295 | 0.740 | 2.438 | 69.6% |
| physical_temporal | anchored_gmm | 0.258 | 2.115 | 0.990 | 1.452 | 38.3% |
| physical_temporal | gmm | 0.260 | 2.118 | 0.990 | 1.456 | 36.8% |
| physical_temporal | kmeans | 0.321 | 1.094 | 0.788 | 1.835 | 37.1% |

## Five-fold full method

- Silhouette: 0.219 $\pm$ 0.022
- Extreme separation: 1.710 $\pm$ 0.497
- Temporal switch reduction: 30.6%
