# PoViT-UQ

**PoViT-UQ** (P-wave Polarity and Arrival Time Estimation using Vision Transformer with Uncertainty Quantification)  
A deep learning model for estimating initial P-wave polarity and arrival time from seismic waveform data, with uncertainty quantification via Monte Carlo Dropout.

---

## 🌍 Overview

PoViT-UQ is a Vision Transformer (ViT)-based model that simultaneously performs:
- Initial P-wave polarity classification (`Up`, `Down`, `Noise`)
- P-wave arrival time estimation  
with uncertainty quantification.

The model integrates **Monte Carlo Dropout (MCD)** to assess the uncertainty of each prediction, enabling selection of high-confidence data for robust focal mechanism estimation.

---

## 🧠 Key Features

- 📈 **High Accuracy**  
  - Polarity classification accuracy > 98%  
  - Arrival time estimation SD ≈ 0.027 s (250 Hz model)

- 🔍 **Uncertainty Quantification**  
  - Multiple forward passes using MCD  
  - Uses Interquartile Range (IQR) as an uncertainty metric

- ⚙️ **Dual Sampling Rates**  
  - Supports 100 Hz and 250 Hz waveform models

- ✅ **Data Filtering by Confidence**  
  - IQR-based selection (e.g., `IQR ≤ 0.15`) improves focal mechanism estimation

---

## 📁 Repository Structure

```bash
PoViT-UQ/
│
├── model/               # Model architecture and training scripts
├── data/                # Sample waveform formats and preprocessing tools
├── utils/               # Helper functions (evaluation, visualization, etc.)
├── notebooks/           # Example Jupyter notebooks
├── requirements.txt     # Required Python packages
└── README.md            # This file
