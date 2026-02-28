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
  - Enables adaptive reconstruction of superimposed signals specifically tailored to non-terrestrial satellite channel conditions.

- **Constellation Reconfiguration**: Minimizes inter-satellite interference (ISI) in shared frequency bands by reconfiguration the constellation.
  - Exploits multidimensional constellation shaping gains by optimizing signal geometry over $n$ channel uses.

- **Dual-Satellite Joint Operation with Single GBS**: Supports a backward-compatible strategy allowing two GEO satellites to transmit simultaneously on the same frequency.
  - Dual GEO satellite to Single GBS.
  - Achieves precise separation and reconstruction of overlapped signals using only a single GBS, avoiding costly infrastructure expansion.

- **Shadowed-Rician Fading**: Incorporates customized neural network layers and activation functions optimized directly for Shadowed-Rician fading channels.
  - Ensures robust performance across various Land Mobile Satellite (LMS) conditions, including Frequent Heavy Shadowing (FHS) and Average Shadowing (AS).

- **SER Performance**: Demonstrates significantly lower Symbol Error Rate (SER) compared to conventional SIC (Successive Interference Cancellation) and JML (Joint Maximum Likelihood) receivers.
