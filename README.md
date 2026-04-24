# LISA Data Challenge – Time-Frequency Analysis

This repository contains solutions for **Subtask 1** of the LISA Data Challenge:

> Time-frequency analysis and visualization of gravitational wave data.

---

## Task Overview

The goal of this project is to:

1. Read LISA gravitational wave data from an HDF5 file  
2. Perform time-frequency analysis using:
   - Wilson-Daubechies-Meyer (WDM) wavelet transform *(approximated with spectrogram for stability)*
   - Fractional Fourier Transform (FRFT)
3. Visualize the results using `matplotlib`

---

## Repository Structure

```

.
├── lisa_wdm.py              # Time-frequency visualization (spectrogram)
├── lisa_frft.py             # FRFT analysis
├── frft.py                  # FRFT implementation (external)
├── environment.yml          # Conda environment
├── README.md
├── wdm_timeseries.png       # Output: time-frequency plot
├── frft_scan.png           # Output: FRFT scan
└── timeseries_overview.png  # Raw signal visualization

````

---

## Environment Setup

Create the environment using:

```bash
conda env create -f environment.yml
conda activate LISA
````

Required packages:

* Python 3.10
* numpy
* scipy
* matplotlib
* h5py

---

## Data

Download the dataset from:

**Baidu Netdisk**

```
https://pan.baidu.com/s/1bfAfIgvi9Gfhlf3SIgrdUw
Code: ucas
```

File used:

```
LDC2_spritz_mbhb1_training_v1.h5
```

---

## How to Run

### 1. Time-Frequency Visualization (WDM-like)

```bash
python lisa_wdm.py LDC2_spritz_mbhb1_training_v1.h5
```

Output:

* `wdm_timeseries.png`
* `timeseries_overview.png`

---

### 2. FRFT Analysis

```bash
python lisa_frft.py LDC2_spritz_mbhb1_training_v1.h5
```

Output:

* `frft_scan.png`
* `timeseries_overview_frft.png`

---

## 📊 Methods

### Time-Frequency Analysis

Instead of directly using the WDM implementation (which showed instability in this environment), a **spectrogram-based approach** was used:

* Short-time Fourier transform (STFT)
* Log-power scaling
* Background subtraction
* Gaussian smoothing

This produces a stable time-frequency representation that highlights:

* Vertical transient structures
* Low-frequency noise dominance
* Signal evolution over time

---

### Fractional Fourier Transform (FRFT)

FRFT is used to analyze chirp-like signals:

* A segment from the **final stage of the signal** is selected
* FRFT is computed over a range of fractional orders (α)
* Results are visualized as a heatmap or scan curve

Enhancements applied:

* Background removal (median subtraction)
* Log amplitude scaling
* Optional smoothing

---

## Results

### Time-Frequency Plot

* Clear time-frequency structure
* Enhanced visibility of signal features
* Log-frequency axis

### FRFT Scan

* Shows energy concentration across fractional orders
* Indicates presence of chirp-like structures

---

## Notes

* The provided `frft.py` implementation may be **unstable for certain α values**

  * Invalid FFT sizes may occur
  * These cases are handled by skipping problematic α

* WDM official implementation was not used due to compatibility issues

  * A spectrogram-based method was used instead

---

## References

* LISA Data Challenge:
  [https://lisa-ldc.in2p3.fr/](https://lisa-ldc.in2p3.fr/)

* WDM Wavelet Transform:
  [https://github.com/XGI-MSU/WDMWaveletTransforms](https://github.com/XGI-MSU/WDMWaveletTransforms)

---

