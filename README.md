# Budget-Aware Industrial Anomaly Detection with Human-in-the-Loop Inspection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-SCADA%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Pytest Suite](https://img.shields.io/badge/Tests-8%2F8%20Passed-2BD4A0.svg)](#test-suite--verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset MVTec AD](https://img.shields.io/badge/Dataset-MVTec%20AD-success.svg)](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
[![Hardware Friendly](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB%20VRAM)-brightgreen.svg)](#hardware-efficiency)

An industrial-grade computer vision system that trains exclusively on defect-free manufactured parts to **prioritize suspicious items for limited human quality review**.

Rather than naively chasing academic AUROC scores in isolation, this system addresses the practical factory reality: *under fixed human inspection budgets (e.g., 10% daily review capacity), how can computer vision prioritize items to catch the maximum number of defects while minimizing the cost of escaped flaws?*

---

## Executive Summary & Core Results

In high-throughput manufacturing, inspecting 100% of items manually is cost-prohibitive, while fully autonomous rejection risks expensive line shutdowns from false alarms. This system implements an **intelligent triage queue**:

```text
[ Production Batch (1,000 items/shift) ]
                   │
                   ▼
[ Feature Backbone: ResNet-18 (512-D Latent Space) ]
                   │
                   ▼
[ Distance Metric: L2 Euclidean to Normal Centroid μ ]
                   │
                   ▼
[ Score-Ranked Queue (Highest Suspicion First) ]
                   │
          ┌────────┴────────┐
          │ (Top 10% items) │ (Remaining 90% items)
          ▼                 ▼
   [ Human Inspector ]  [ Released to Shipping ]
```

### Multi-Category Benchmark (1,000-Unit Shift Simulation @ 10% Human Review Quota)

| Category | Domain | Normal Train | Test Total | Image AUROC | Random Sampling Recall (10% Budget) | **AI Prioritized Recall (10% Budget)** | **Defect Capture Multiplier** | **Net Simulated Savings ($/1k units)** | Reliability Tier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zipper** | Object | 240 | 151 | **90.55%** | 10.2% | **73.1%** | **7.17x** | **$943.50** (62% cost cut) | **STRONG** |
| **Leather** | Texture | 245 | 124 | **89.84%** | 10.0% | **73.0%** | **7.30x** | **$945.00** (63% cost cut) | **MODERATE** |
| **Wood** | Texture | 247 | 79 | **88.68%** | 9.9% | **70.7%** | **7.14x** | **$912.00** (61% cost cut) | **MODERATE** |
| **Hazelnut** | Object | 391 | 110 | **84.71%** | 10.1% | **61.4%** | **6.08x** | **$769.50** (51% cost cut) | **MODERATE** |
| **Toothbrush** | Object | 60 | 42 | **73.89%** | 10.1% | **38.9%** | **3.85x** | **$432.00** (29% cost cut) | **WEAK** |
| **Grid** | Texture | 264 | 78 | **46.03%** | 10.0% | **6.8%** | **0.68x** | **$-37.20** (Negative ROI) | **UNRELIABLE** |

### Architectural Insights & Domain Failures
* **Surface Textures & Molded Parts:** High performance (*Zipper 90.5%*, *Leather 89.8%*, *Wood 88.7%*). Defects spanning >1% of image pixels strongly shift the global feature embedding away from the normal centroid.
* **The Grid Failure Mode (Scientific Honesty):** Grid consists of a repetitive, high-frequency wire mesh. A warped wire occupies <0.1% of spatial pixels. Global average pooling (`AdaptiveAvgPool2d`) dilutes small localized anomalies into the background, yielding an AUROC of 46.0% (sub-random). The system explicitly flags this category as **UNRELIABLE** to prevent negative financial impact.

---

## SCADA / Industrial Control-Room Interface

Designed from the ground up to reflect modern industrial supervisory systems (Siemens, Cognex, Landing AI) rather than a prototype demo:

* **Dual-Audience Architecture:**
  * **Operator View:** Plain-language operational terms, actionable verdicts, and direct cost savings without ML jargon.
  * **Engineer View:** Continuous anomaly distances, 95th percentile calibration thresholds, ROC curves, confusion matrices, and model cards.
* **Three-State Decision Engine:** Replaces naive binary outputs with `PASS`, `SEND TO HUMAN REVIEW` (calibrated 12% buffer below threshold), and `REJECT`.
* **Calibrated Horizontal Score Gauge:** Visual pin marker positioned along nominal pass, review buffer, and reject zones with plain-English distance notes.
* **Station Event Log:** Real-time inspection logging with one-click CSV export and duplicate-event suppression.
* **Offline Font & Asset Bundling:** Includes local `Barlow Condensed`, `IBM Plex Sans`, and `JetBrains Mono` fonts for air-gapped industrial operation.

---

## Cost Model & Economic Rationale

Manufacturing inspection costs are inherently **asymmetric**:
* **Cost of Review ($C_{\text{review}}$):** Labor cost for an inspector to examine one item ($\approx \$0.50$).
* **Cost of Escaped Defect ($C_{\text{miss}}$):** Severe warranty, recall, or scrap cost when a defect reaches a customer ($\approx \$30.00 - \$100.00+$).

Under a fixed review quota of $B$ items per shift:
$$\text{Total Cost} = (B \times C_{\text{review}}) + (\text{Missed Defects} \times C_{\text{miss}})$$

Because both random inspection and AI triage utilize the exact same review budget $B$, all cost savings stem directly from **reducing escaped defects**.

---

## Repository Architecture

```text
Defect Detection/
├── app.py                      # Main SCADA control-room application entrypoint
├── config.py                   # Centralized design tokens, reliability thresholds, constants
├── conftest.py                 # Pytest root path configuration
├── Dockerfile                  # Production container configuration
├── requirements.txt            # Locked project dependencies
├── LICENSE                     # MIT Open Source License
├── README.md                   # System documentation
├── assets/
│   ├── style.css               # Control-room SCADA CSS design system
│   └── fonts/                  # Bundled offline local TTF fonts (Barlow, IBM Plex, JetBrains)
├── ui/
│   ├── __init__.py
│   ├── components.py           # Header bar, KPI cards, score gauge, verdict cards
│   ├── charts.py               # Dark-theme Plotly visualizations with direct value labels
│   ├── terms.py                # Unified translation dictionary (Operator vs. Engineer)
│   └── tabs/
│       ├── __init__.py
│       ├── inspection.py       # Tab 1: Single-item inspection & session event log
│       ├── simulation.py       # Tab 2: Factory shift Monte Carlo simulation & sensitivity
│       └── benchmark.py        # Tab 3: 6-category cross-domain benchmark & failure analysis
├── src/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset loaders with ImageNet normalization
│   ├── model.py                # ResNet-18 Anomaly Detector & Centroid Scorer
│   └── simulation.py           # Monte Carlo shift simulator & cost model
├── tests/
│   ├── test_decision.py        # 3-state decision engine & 12% review band tests
│   ├── test_simulation.py      # Statistical simulation logic & budget constraints
│   └── test_invariance.py      # Numerical invariance & metric reproducibility tests
├── notebooks/
│   ├── 01_toothbrush_data_audit.ipynb          # Exploratory data audit & mask analysis
│   ├── 02_baseline_feature_embedding.ipynb     # Feature extraction, AUROC & shift simulation
│   └── 03_multicategory_benchmark.ipynb        # 6-category benchmark & cross-domain analysis
└── scripts/
    └── capture_screenshots.py  # Playwright automated UI verification across all views
```

---

## Quickstart & Setup Guide

### 1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/jaythakkar735/Defect-Detection.git
cd "Defect-Detection"

# Create Conda environment with Python 3.10
conda create -n defect-detection python=3.10 -y
conda activate defect-detection

# Install PyTorch with CUDA support (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download MVTec AD target categories from the [MVTec Portal](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and place them under `data/raw/<category>/`:
```text
data/raw/
├── toothbrush/
├── grid/
├── hazelnut/
├── leather/
├── wood/
└── zipper/
```

### 3. Launch the SCADA Web Application
```bash
conda activate defect-detection
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### 4. Run the Automated Test Suite
```bash
conda activate defect-detection
pytest tests/
```
Output:
```text
tests/test_decision.py ....                                              [ 50%]
tests/test_invariance.py .                                               [ 62%]
tests/test_simulation.py ...                                             [100%]
============================== 8 passed in 7.70s ===============================
```

### 5. Run the Jupyter Notebooks
```bash
jupyter lab
# or: jupyter notebook
```
Open `notebooks/01_toothbrush_data_audit.ipynb`, `02_baseline_feature_embedding.ipynb`, or `03_multicategory_benchmark.ipynb`.

---

## Docker Deployment

To build and run the system within a container:

```bash
# Build the Docker image
docker build -t defect-detection:latest .

# Run container mounting local data directory
docker run -p 8501:8501 -v $(pwd)/data/raw:/app/data/raw defect-detection:latest
```

---

## Hardware Efficiency

Tested and optimized for resource-constrained edge computing environments:
* **GPU Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
* **System RAM:** 8 GB
* **Optimization Highlights:** Single-pass frozen inference (`torch.no_grad()`), immediate tensor offloading to CPU NumPy arrays, and standardized $256 \times 256$ inputs maintain peak VRAM usage **below 450 MB**, preventing Out-Of-Memory (OOM) faults.

---

## Author & License

* **Author:** Jay Thakkar ([@jaythakkar735](https://github.com/jaythakkar735) / thakkarjay810@gmail.com)
* **Code License:** [MIT License](LICENSE)
* **Dataset License:** The MVTec Anomaly Detection benchmark dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/). Raw dataset files are not redistributed in this repository.
