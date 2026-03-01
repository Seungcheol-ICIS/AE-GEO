# ePASS: Autoencoder-Based Constellation Redesign for Joint Operation of Dual Geostationary Earth Orbit Satellites

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Communications%20Letters-blue)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.17-green.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2025a-orange.svg)](https://www.mathworks.com/)
<!-- [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) -->

<!-- Official implementation of **"Lightweight Physical-Layer Authentication via IQ Sample Super-Resolution in LEO Satellite Networks"** -->
Implementation code for **"Autoencoder-Based Constellation Redesign for Joint Operation of Dual Geostationary Earth Orbit Satellites"**

> **Authors:** Seungcheol Ha†, Seongmin Pyo, Dong-Hyo Lee, Taehoon Kim, and Inkyu Bang
>
> † These authors contributed equally to this work.
>
> **Submitted in:** IEEE Communications Letters

---

## Overview
ePASS is a practical and lightweight **autoencoder-based constellation redesign framework** for spectrum-sharing dual GEO satellite downlink systems. It leverages an end-to-end autoencoder to jointly learn signal encoding/decoding and channel-adaptive constellation reconfiguration, enabling a single ground base station (GBS) to reliably separate and reconstruct superimposed signals under satellite fading conditions, while achieving improved symbol error rate (SER) performance compared to conventional SIC and joint maximum-likelihood (JML) receivers. 

### Key Features
- **AE-based End-to-End Learning Framework**: Leverages a deep learning-based autoencoder (AE) to jointly optimize the signal encoding (GEO satellite) and decoding (GBS) processes.
- **Constellation Reconfiguration**: Minimizes inter-satellite interference (ISI) in shared frequency bands by reconfiguration the constellation.
- **Dual-Satellite Joint Operation with Single GBS**: Supports a backward-compatible strategy allowing two GEO satellites to transmit simultaneously on the same frequency.
- **Shadowed-Rician Fading**: Incorporates customized neural network layers and activation functions optimized directly for Shadowed-Rician fading channels.
- **SER Performance**: Demonstrates significantly lower Symbol Error Rate (SER) compared to conventional SIC (Successive Interference Cancellation) and JML (Joint Maximum Likelihood) receivers.

![System Model](figure/fig1_system_model.png)
- **Two GEO satellites**: GEO1 and GEO2
- **Single GBS**
- **Joint Signal Reception & Estimation**
- **Comparables**: SIC, JML and proposed AE end-to-end framework

## Installation

### Requirements

```bash
pip install -r requirements.txt
```
**Python Environment (3.13.7+)**
- TensorFlow >= 2.15.0 (Keras 3.0 compatible)
- numpy, pandas, matplotlib (Data processing & Visualization)

**MATLAB Environment (R2025b)**
- Deep Learning Toolbox
- Satellite Communications Toolbox

**Hardware (Recommended)**
- NVIDIA GPU with CUDA support for training the Autoencoder
- Deployment: NVIDIA Jetson Orin Nano or equivalent edge AI hardware for real-time inference


### 2.Autoencoder architecture
![Autoencoder architecture](figure/fig2_Autoencoder_architecture.png)

### Encoder(GEO satellite)
The encoder maps input messages into a complex-valued constellation space with power normalization.
- Input: One-hot encoded messages ($M_1=8, M_2=4$).
- Architecture: Dense layers with ReLU activation followed by $L_2$ normalization to satisfy power constraints.
- Output: $n$ dimensional complex symbols.

##### Channel 
The channel simulates real-world satellite impairments:
- Fading: Shadowed Rician Fading using parameters for Average Shadowing (AS).

| Shadowing Type | $b$ | $m$ | $\Omega$ |
|----------------|---|---|--------|
| Frequent Heavy Shadowing (FHS) | 0.063 | 0.739 | 8.97 × 10^-4|
| Average Shadowing (AS)  | 0.126 | 10.1 | 0.835
| Infrequent Light Shadowing (ILS) | 0.158 | 19.4 | 1.29|

- **$b$**: Scattering parameter
- **$m$**: Shape parameter
- **$\Omega$**: Average power
- Interference: Signals from both transmitters are summed to simulate NOMA.
- Noise: Additive White Gaussian Noise (AWGN) based on target SNR.
