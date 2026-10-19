# Physics-Informed Denoising for Two-Time Correlation (TTC) Functions in XPCS

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Domain: Synchrotron XPCS](https://img.shields.io/badge/Domain-Synchrotron%20XPCS-orange.svg)](#scientific-background)

A scientific computing and analysis framework for evaluating and reproducing **Physics-Informed Deep Learning (PI-PD-CNN / PINN)** denoising of Two-Time Correlation (TTC) functions in **X-ray Photon Correlation Spectroscopy (XPCS)**.

Developed by the **Gutt Lab** (Department of Physics, Universität Siegen).

---

## Table of Contents

- [Scientific Background](#scientific-background)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart: Reproduce Results](#quickstart-reproduce-results)
- [Dataset Specifications](#dataset-specifications)
- [Core Modules & API Reference](#core-modules--api-reference)
  - [TTCProcessor (`ttc_processor.py`)](#ttcprocessor-ttc_processorpy)
  - [Analysis Toolkit (`helper_functions.py`)](#analysis-toolkit-helper_functionspy)
- [Customizing Reproduction Parameters](#customizing-reproduction-parameters)
- [License & Citation](#license--citation)

---

## Scientific Background

In X-ray Photon Correlation Spectroscopy (XPCS), dynamic speckle patterns from coherent synchrotron beamlines (e.g., PETRA III, ESRF, APS) probe nanoscale dynamics in soft matter, glasses, and complex fluids.

The **Two-Time Correlation (TTC) function** $C(t_1, t_2, q)$ measures correlations between scattering intensities at times $t_1$ and $t_2$ at wavevector transfer $q$:

$$C(t_1, t_2, q) = \frac{\langle I(t_1, q) I(t_2, q) \rangle}{\langle I(t_1, q) \rangle \langle I(t_2, q) \rangle}$$

- **The Challenge:** Investigating fast sub-microsecond processes or radiation-sensitive samples yields photon-starved, low-statistics TTC matrices dominated by Poissonian shot noise and high standard error.
- **The Solution:** Physics-informed deep learning models (PI-PD-CNN) enforce physical symmetries ($C(t_1, t_2) = C(t_2, t_1)$ for stationary regimes), asymptotic baselines, and temporal smoothness to suppress noise while faithfully preserving the underlying relaxation dynamics ($g_2(\tau)$ Kohlrausch–Williams–Watts stretched exponential behavior).

---

## Key Features

- **Quantified Benchmark Comparison:** Rigorously benchmarks denoised TTCs against a high-statistics ground truth reference ($N = 2800$ scans), achieving **>75% to >82% RMSE reduction** on low-statistics scans ($N = 200$).
- **End-to-End Figure Reproduction:** One-command reproduction of manuscript figures (2D TTC maps, residual distributions, $g_2(\tau)$ relaxation curves, error propagation).
- **Logarithmic Time Binning & Error Propagation:** Robust diagonal cutting, logarithmic delay binning, and analytical error propagation through non-linear curve fitting.
- **Comprehensive Physics Diagnostics:**
  - Residual trace analysis and Gaussian normality verification
  - 2D Fast Fourier Transform (FFT) and diagonal ridge Signal-to-Noise Ratio (SNR)
  - Non-Negative Least Squares (NNLS) Inverse Laplace transforms for relaxation rate ($\Gamma$) and hydrodynamic radius ($R_h$) distributions
  - Residual whiteness tests (autocorrelation, Durbin–Watson)

---

## Repository Structure

```text
pinn-denoising/
├── denoised_ttc_v7/                # Precomputed TTC and SE datasets (.npy)
│   ├── Ref_TTC.npy                 # Benchmark reference TTC (2800 scans, 20 q, 100x100)
│   ├── Ref_SE.npy                  # Standard error of the reference TTC
│   ├── ttc_eggyolk.npy             # Raw noisy TTC across 11 averaging levels
│   ├── SE_ttc_eggyolk.npy          # Standard error of raw TTC
│   ├── ttc_denoised_parallel.npy   # PI-PD-CNN denoised TTC across 11 averaging levels
│   ├── SE_ttc_denoised_parallel.npy# Standard error of denoised TTC
│   └── ...                         # Symmetrized and combined variants
├── helper_functions.py             # Scientific analysis library (FFT, Laplace, SNR, fits)
├── ttc_processor.py                # TTC extraction, log-binning, and KWW fitting engine
├── results_plots_reproduce.py      # Main reproduction script for publication figures
├── plotPINN/                       # Output directory for generated figures and metrics
├── requirements.txt                # Python package dependencies
├── requirement.txt                 # Alias pointing to requirements.txt
├── LICENSE                         # GNU General Public License v3.0
└── README.md                       # Project documentation
```

---

## Installation

### 1. Prerequisites
- Python **3.9** or higher (tested on Python 3.10, 3.11, and 3.12).
- Git.

### 2. Clone the Repository
```bash
git clone https://github.com/Gutt-lab/pinn-denoising.git
cd pinn-denoising
```

### 3. Create and Activate a Virtual Environment

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The dependencies are also accessible via `pip install -r requirement.txt`.

The project requires:
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `matplotlib >= 3.7.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.2.0`

---

## Quickstart: Reproduce Results

To run the full evaluation pipeline and generate all publication-ready figures:

```bash
python results_plots_reproduce.py
```

### Output Artifacts
The script automatically populates the `./plotPINN/` directory with:

1. **2D TTC Comparison Maps (`ttc_comparison_*.png`, `*.pdf`):**
   - Side-by-side comparison of **Reference (2800 scans)**, **Raw Noisy**, and **PI-PD-CNN Denoised** TTC matrices.
   - Residual trace across pixels and Gaussian fit distribution demonstrating white Gaussian noise suppression.
2. **$g_2(\tau)$ Correlation Fits & Residuals (`g2_reference_residuals_*.png`, `*.pdf`):**
   - Extracted intensity autocorrelation curves $g_2(\tau)$ with error bars.
   - Residual curves ($\Delta g_2 = g_2 - g_{2,\mathrm{ref}}$) highlighting the precision gain of the denoising model.

   All plots saved to: ./plotPINN/
   ```

---

## Dataset Specifications

All precomputed matrices in `denoised_ttc_v7/` are stored in standard NumPy binary (`.npy`) format:

| File | Shape | Description |
| :--- | :--- | :--- |
| `Ref_TTC.npy` | `(20, 100, 100)` | Ground truth reference TTC averaged over **2800 independent scans** |
| `Ref_SE.npy` | `(20, 100, 100)` | Standard error of the 2800-scan reference TTC |
| `ttc_eggyolk.npy` | `(11, 20, 100, 100)` | Raw noisy TTC across 11 scan averaging levels |
| `SE_ttc_eggyolk.npy` | `(11, 20, 100, 100)` | Standard error of the raw TTC scans |
| `ttc_denoised_parallel.npy` | `(11, 20, 100, 100)` | PI-PD-CNN denoised TTC across 11 scan averaging levels |
| `SE_ttc_denoised_parallel.npy` | `(11, 20, 100, 100)` | Standard error of the denoised TTC |

### Coordinate Axes & Parameters
- **Averaging scan depths (Index 0..10):** `[5, 25, 50, 100, 200, 400, 600, 800, 1000, 1200, 1300]` scans.
- **Wavevector transfer values ($q$):** 20 discrete values ranging from $0.095\ \text{nm}^{-1}$ to $0.289\ \text{nm}^{-1}$.
- **Temporal grid:** 100 delay points with frame delay time $\Delta \tau = 0.44\ \mu\text{s}$ (bin centers spanning $0.22\ \mu\text{s}$ to $43.78\ \mu\text{s}$).
- **Optical Speckle Contrast ($\beta$):** Values ranging from $\approx 0.052$ at low $q$ to $\approx 0.040$ at high $q$.

---

## Core Modules & API Reference

### TTCProcessor (`ttc_processor.py`)

The `TTCProcessor` class encapsulates diagonal cut extraction, logarithmic delay binning, and Kohlrausch–Williams–Watts (KWW) stretched exponential fitting.

```python
import numpy as np
from pathlib import Path
from ttc_processor import TTCProcessor

# 1. Setup time delays and contrast vector
FRAME_DELAY = 0.44e-6
n_t = 100
delays = np.arange(2 * n_t) * FRAME_DELAY
time_binned = delays.reshape(n_t, 2).mean(axis=1)

contrast = np.array([...])  # 20 elements matching q-values

# 2. Load data
ttc_data = np.load('./denoised_ttc_v7/ttc_denoised_parallel.npy', mmap_mode='r')
se_data  = np.load('./denoised_ttc_v7/SE_ttc_denoised_parallel.npy', mmap_mode='r')

# 3. Process an ensemble (e.g. index 4 -> 200 scans)
processor = TTCProcessor()
results = processor.process_and_fit_all_q(
    ttc_stack=ttc_data[4],        # shape: (20, 100, 100)
    se_stack=se_data[4],          # shape: (20, 100, 100)
    time_vector=time_binned,      # length: 100
    betas_to_use=contrast,
    num_cuts=1,
    log_avg_n=100
)

# Extracted results dictionary:
# results['t_list']            -> Delay times per q and cut
# results['g2_list']           -> Extracted g2(tau) curves
# results['err_list']          -> Propagated uncertainties
# results['fitted_g2_list']    -> Evaluated KWW fits
# results['taus'], results['kwws'] -> Fitted relaxation times and stretching exponents
```

#### Key Methods:
- `process_and_fit_all_q(ttc_stack, se_stack, time_vector, betas_to_use, num_cuts, log_avg_n)`: Iterates over all $q$-vectors, bins diagonals logarithmically, and fits the KWW model:
  $$g_2(\tau) = A + \beta \exp\left[-2\left(\frac{\tau}{\tau_0}\right)^\gamma\right]$$
- `g2_section_area(ttc, std, time_delays_in_second, num, log_avg_n)`: Extracts horizontal cuts from the 2D correlation matrix, removes self-correlation ($\tau=0$), and computes weighted log-binned averages.
- `propagate_g2_error(t, popt, pcov)`: Calculates exact variance on fitted $g_2$ using Jacobian error propagation ($\sigma^2 = \mathbf{J} \mathbf{\Sigma} \mathbf{J}^T$).

---

### Analysis Toolkit (`helper_functions.py`)

A comprehensive library of statistical and physical diagnostic tools:

| Function | Purpose |
| :--- | :--- |
| `analyze_residuals(y_noisy, y_denoised, dt, qvalue)` | Computes difference matrix, residual trace, and fits a Gaussian distribution to test normality |
| `analyze_ridge_snr_many(ttc_dict, ...)` | Computes ridge signal-to-noise ratio in 2D FFT power spectrum with bootstrap confidence intervals |
| `inverse_laplace_nnls_g2(t, g2_norm, Gamma_grid)` | Inverts $g_2(\tau)$ via Non-Negative Least Squares (NNLS) to obtain relaxation rate distributions |
| `g2_goodness_of_fit(y_data, y_fit, *popt)` | Calculates $R^2$, $\chi^2$, reduced $\chi^2$, and RMSE metrics |
| `residual_whiteness_g2_single(residuals)` | Performs Durbin–Watson test, Ljung–Box test, and autocorrelation analysis to check whiteness |
| `fit_g2_kww_many(...)` | Performs Iteratively Reweighted Least Squares (IRLS) multi-curve KWW fitting with robust outlier rejection |

---

## Customizing Reproduction Parameters

You can edit `results_plots_reproduce.py` to evaluate different configurations:

```python
# ---------------------------- Configuration ----------------------------
DATA_DIR = Path('./denoised_ttc_v7')
OUTPUT_DIR = Path('./plotPINN')

# Scan averaging depth index to evaluate (0 = 5 scans, 4 = 200 scans, 5 = 400 scans, etc.)
LOW_STATS_INDEX = 4

# Q-indices to plot (0 to 19, e.g., index 10 is q = 0.195 nm^-1, index 12 is q = 0.215 nm^-1)
Q_INDICES = [10, 12]

# Number of lag points for logarithmic averaging
LOG_AVG_N = 100

# Set to True to display interactive GUI matplotlib windows during execution
SHOW_PLOTS = False
```

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.



## Contact & Inquiries

For questions, issues, or contributions, please contact the repository maintainers or open an issue on GitHub.
