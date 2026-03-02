# Supplementary Materials

**Paper:** Autoencoder-Based Constellation Redesign for Joint Operation of Dual Geostationary Earth Orbit Satellites

**Manuscript ID:** CL2025-2743

**Journal:** IEEE Communications Letters

---

Due to the page limitations of IEEE CL (5 pages), we provide additional experimental results and technical details in this repository as supplementary materials. These materials correspond to the revision responses referenced in our reply letter.

## Contents

| File | Reviewer Comment | Description |
|------|-----------------|-------------|
| [Channel_Specificity_Analysis.md](Channel_Specificity_Analysis.md) | R2.2 | Comparative analysis of performance in Shadowed-Rician vs. Rayleigh channels. |
| [Architecture_Optimization_Analysis.md](Architecture_Optimization_Analysis.md) | R2.3 | Ablation studies on layer depth and activation functions to justify AE design. |
| [Power_Difference_Robustness_Analysis.md](Power_Difference_Robustness_Analysis.md) | R2.4 | Robustness evaluation across various transmit power gaps (0 dB to 12 dB). |
| [Geometric_Structure_and_Dimensional_Separation.md](Geometric_Structure_and_Dimensional_Separation.md) | R3.5 | Geometric interpretation of dimensional separation and constellation shaping for $n=4$. |

## Default System Configuration

Unless otherwise stated, experiments use the following default parameters:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| GEO1 satellite modulation | $M1_$ | $8$ | GEO1 satellite modulation method |
| GEO2 satellite modulation | $M2_$ | $4$ | GEO2 satellite modulation method |
| GEO1 satellite dimensions | $n1_$ | $2$ | Number of dimensions GEO1 satellite|
| GEO2 satellite dimensions | $n2_$ | $2$ | Number of dimensions GEO2 satellite|
| Shadowed-Rician $b$ | $b$ | $0.126$ | Scattering parameter |
| Shadowed-Rician $m$ | $m$ | $10.1$ | Shape parameter |
| Shadowed-Rician $\Omega$ | $m$ | $9.97 \tiems 10^-4$ | Average power |
| Training samples | - | 500,000 | Number of training channel realizations |
| Epochs | - | 25 | Number of training iterations |
| Batch-size | - | 512 | Number of samples per gradient update |
| Optimizer | - | 0.006 | Learning rate for Adam optimizer | 
| Training SNR | - | 20 | SNR (dB) used during training |
| Test samples | - | 1,000,000 | Total samples for validation | 


## Reproducibility


All experiments can be reproduced using the source code in this repository. See the main [README.md](../README.md) for installation and usage instructions.

```bash
# Autoencoder-based NOMA freamwork
python main.py --num-samples 500000 --num-epochs 25 --n-test 10000000
```
