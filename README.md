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

### Performance Highlights

- High reconstruction precision **98% or more at 20dB SNR** in shadowed-rician fading environments
- **Rapid training convergence** with exceptional data efficiency
- Advanced **multi-user(Satellite) interference (MUI)** suppression via geometric shaping
- Edge AI Optimized Architecture for **Real-Time Satellite Communications**
![System Model](figure/fig1_system_model.png)

## Project Structure

```

AE-GEO/
├── src/
│   ├── main.py              # Execution Control (Training Loops, Gradient Steps, Test Scenarios)
│   ├── models.py            # Neural Network Architectures (Encoder, Decoder, Normalization Layers)
│   ├── channel.py           # Physical Layer Simulation (Shadowed Rician H Gen, Fading/AWGN Layers)
│   ├── utils.py             # Numerical Ops & Data Handling (SNR-to-Noise, SER Calculation, Batch Gen)
├── data/
│   └── constellations/      # Storage for Trained Symbol Coordinates (CSV format)
├── figures/                 # Publication-ready Figures (SER Curves, Constellation PNGs)
└── results/                 # Numerical Simulation Data (SER results, Training Logs)

```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```
### Dependencies
- TensorFlow >= 2.15.0 (Keras 3.0 compatible)
- numpy, pandas, matplotlib (Data processing & Visualization)

---

## Usage

### Training
```bash
python main.py \
    --num-samples 100000 \
    --num-epochs 25 \
    --batch-size 256 \
    --lr 0.001 \
    --noma-snr 3 \
    --training-snr 20 \
    --device auto
```

### Command Line Arguments
```bash
| Argument | Description | Default |
|----------|-------------|---------|
| `--save-path` | Directory path where result files (CSV, PNG) will be stored | ./results |
| `--tag` | Prefix tag for identifying specific experiment runs | AE_NOMA_GEO |
| `--num-samples` | Training dataset size | 500000 |
| `--num-epochs` | Number of training epochs | 25 |
| `--batch-size` | Mini-batch size | 512 |
| `--lr` | Learning rate | 0.006 |
| `--noma-snr` | Power/SNR offset between the two GEO users (dB) | 3 |
| `--training-snr` | Baseline SNR used during the training phase (dB) | 20 |
| '--n-test' | Test dataset size | 1000000 |
| '--device' | Device: auto, cpu, cuda | auto |
```

### System parameters
```bash
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--save-path` | Directory path where result files (CSV, PNG) will be stored | ./results |
| `--tag` | Prefix tag for identifying specific experiment runs | AE_NOMA_GEO |
| `--num-samples` | Training dataset size | 500000 |
| `--num-epochs` | Number of training epochs | 25 |
| `--batch-size` | Mini-batch size | 512 |
| `--lr` | Learning rate | 0.006 |
| `--noma-snr` | Power/SNR offset between the two GEO users (dB) | 3 |
| `--training-snr` | Baseline SNR used during the training phase (dB) | 20 |
| '--n-test' | Test dataset size | 1000000 |
| '--device' | Device: auto, cpu, cuda | auto |
```

## Citation

*Citation information will be added upon publication.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **EDSR**: [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) - Super-resolution model architecture
- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - Classification backbone
- **Iridium Dataset**: [PAST-AI](https://github.com/spritz-group/PAST-AI) - Public satellite IQ sample dataset
- **AMC Framework**: T. K. Oikonomou et al., "CNN-Based Automatic Modulation Classification Under Phase Imperfections," IEEE WCL, 2024 - Polar coordinate transformation baseline

---

## Contact

For questions or issues, please open an issue or contact:
- Seungcheol Ha: scha@edu.hanbat.ac.kr
- Inkyu Bang: ikbang@hanbat.ac.kr
