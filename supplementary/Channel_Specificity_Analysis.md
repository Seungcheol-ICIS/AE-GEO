# Comparative Analysis of Channel Specificity and Model Robustness (R2.2)

**Reviewer Comment (R2.2):** "The innovation claim of using autoencoders for constellation redesign is not sufficiently novel, as similar approaches exist in NOMA contexts; the article does not clearly articulate how the satellite-specific adaptations."

## Overview

To validate the satellite-specific optimization of the proposed Autoencoder (AE) framework, we conduct a comparative analysis between a generic Rayleigh fading channel (typical for terrestrial NOMA) and the Shadowed-Rician (SR) fading channel (specific to satellite-to-ground links).


### Simulation Setup
* **Terrestrial Channel:** Rayleigh fading channel.
* **Satellite Channel:** Shadowed-Rician (SR) fading channel.
* **Metric:** Symbol Error Rate (SER) and relative improvement rate.

### Performance Comparison Table

| SNR (dB) | Method | SER (Rayleigh) | SER (SR) | Improvement (%) |
| :---: | :--- | :---: | :---: | :---: |
| | SIC | 0.498 | 0.557 | -11.668 |
| 8 | JML | 0.317 | 0.362 | -14.424 |
| | AE (Ours) | 0.129 | 0.066 | 48.513 |
| | SIC | 0.529 | 0.567 | -7.206 |
| 14 | JML | 0.078 | 0.087 | -12.467 |
| | AE (Ours) | 0.042 | 0.013 | 68.723 |
| | SIC | 0.556 | 0.587 | -5.560 |
| 20 | JML | 0.008 | 0.008 | -2.104 |
| | AE (Ours) | 0.015 | 0.004 | 76.122 |

![System Model](figure/fig1_system_model.png)


### Result
<table align="center">
  <tr>
    <td align="center">
      <img src="figure/fig_r22_Comparison_channel_Rayleigh.png" width="100%">
      <br>
      <b>Figure 1. Rayleigh Results</b>
    </td>
    <td align="center">
      <img src="figure/fig_r22_Comparison_channel_SRfading.png" width="100%">
      <br>
      <b>Figure 2. Shadowed-Rician Results</b>
    </td>
  </tr>
</table>


## Key Findings
While conventional NOMA reception techniques (SIC, JML) are designed for Rayleigh environments and suffer performance degradation in satellite-specific channels, our proposed AE model demonstrates inherent adaptability to the Shadowed-Rician characteristics.

- **Conventional Methods (SIC/JML)**: Experience performance degradation (negative improvement) in the satellite-specific SR channel compared to the Rayleigh channel. This suggests that conventional methods are not optimized for the specific fading characteristics of satellite links.

- **Proposed AE Scheme:** Demonstrates significant performance gains in the SR channel, achieving an improvement of up to **76.1% at 20 dB SNR**. This confirms that the proposed AE is highly specialized and robust for satellite environments.