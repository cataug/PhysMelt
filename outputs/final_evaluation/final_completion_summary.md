# PhysMelt final evaluation

## Alarm evaluation

| Definition | Method | AUROC | AUPRC | Precision | Recall | F1 | Event recall | False alarms/part |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| top_10_percent_measured_power | gmm_probability_0.50 | 0.801 | 0.260 | 0.248 | 0.588 | 0.349 | 0.862 | 30.02 |
| top_10_percent_measured_power | gmm_probability_0.70 | 0.801 | 0.260 | 0.253 | 0.546 | 0.345 | 0.839 | 28.11 |
| top_10_percent_measured_power | temporal_gmm_high_state | 0.801 | 0.260 | 0.257 | 0.604 | 0.360 | 0.862 | 22.42 |
| measured_power_at_least_180W | gmm_probability_0.50 | 0.846 | 0.635 | 0.621 | 0.569 | 0.594 | 0.797 | 10.95 |
| measured_power_at_least_180W | gmm_probability_0.70 | 0.846 | 0.635 | 0.634 | 0.529 | 0.577 | 0.765 | 10.02 |
| measured_power_at_least_180W | temporal_gmm_high_state | 0.846 | 0.635 | 0.632 | 0.574 | 0.602 | 0.766 | 7.13 |

## Edge benchmark

- Basic features: threshold_area, mean_intensity
- GMM parameter count: 21
- Serialized GMM size: 1831 bytes
- GMM inference latency: 0.325 us/frame
- GMM inference throughput: 3075582 FPS
- Raw AVI decode plus minimal feature extraction: 0.638 ms/frame
- Raw decode plus minimal feature throughput: 1568.6 FPS
- Distilled logistic macro-F1 agreement: 0.942

## Qualitative panels

- `high_power`: RHF_MPM_P12, frames 644--948, correction fraction 0.7%
- `temporal_correction`: RHF_MPM_P01, frames 751--1055, correction fraction 18.7%
- `transition`: RHF_MPM_P11, frames 258--569, correction fraction 3.7%
- `representative`: RHF_MPM_P28, frames 1184--1494, correction fraction 1.0%

The descriptive best-F1 probability threshold is reported for inspection only; paper claims should rely primarily on threshold-free AUROC/AUPRC and fixed operating thresholds.
