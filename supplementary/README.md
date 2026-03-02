# Supplementary Materials

**Paper:** Autoencoder-Based Constellation Redesign for Joint Operation of Dual Geostationary Earth Orbit Satellites

**Manuscript ID:** CL2025-2743

**Journal:** IEEE Communications Letters

---

Due to the page limitations of IEEE CL (5 pages), we provide additional experimental results and technical details in this repository as supplementary materials. These materials correspond to the revision responses referenced in our reply letter.

## Contents

| File | Reviewer Comment | Description |
|------|-----------------|-------------|
| [Channel_Specificity_Analysis.md](Channel_Specificity_Analysis.md) | R2.2 | A |
| [performance_gap_analysis.md](Channel_Specificity_Analysis.md) | R2.3 | B |
| [ablation_study.md](ablation_study.md) | R2.4 | C |
| [an_satellite_analysis.md](an_satellite_analysis.md) | R3.5 | D |

## Default System Configuration

Unless otherwise stated, experiments use the following default parameters:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Visible satellites | $N$ | $15$ | LEO satellites in GBS field of view |
| Scheduled satellites | $K$ | $10$ | Satellites selected for transmission |
| Data satellites | $K_d$ | $2$ | Satellites transmitting data |
| AN satellites | $K_{\text{AN}}$ | $K - K_d = 8$ | Satellites transmitting artificial noise |
| GBS antennas | $M_b$ | $2$ | Zero-forcing (ZF) receiver |
| Eve antennas | $M_e$ | $2$ | MMSE receiver |
| Shadowed-Rician $K$ | $K_{\text{SR}}$ | $3$ | Rician K-factor |
| Shadowed-Rician $m$ | $m$ | $5$ | Nakagami-$m$ parameter |
| Training samples | - | 40,000 | Number of training channel realizations |
| MC samples | - | 100 | Monte Carlo samples for ergodic rate estimation |

## Reproducibility


All experiments can be reproduced using the source code in this repository. See the main [README.md](../README.md) for installation and usage instructions.

```bash
# Example: Train with a specific system configuration
python main.py train --num-samples 40000 --num-epochs 12 --num-data-sats 2

# Example: Evaluate a trained model
python main.py evaluate checkpoints/model.pt --num-trials 1000
```
