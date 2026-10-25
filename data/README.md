# Dataset Provenance & Instructions

This project uses the **MVTec Anomaly Detection (MVTec AD)** benchmark dataset.

## Official Source
- **Official Website:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- **Paper Citation:**
  > Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger.  
  > *"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"*,  
  > In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

## License Terms
The dataset is published under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.
- **Non-Commercial:** Research, personal study, and portfolio demonstration only.
- **Attribution:** Any usage must credit MVTec Software GmbH and cite the original CVPR 2019 paper.
- **Repository Cleanliness:** Raw images and archives (`*.tar.xz`) are excluded from Git via `.gitignore` to maintain repository hygiene and respect distribution clauses.

## Directory Structure
To replicate our experiments, place the extracted categories under `data/raw/<category>/`:
```text
data/
└── raw/
    ├── toothbrush/
    │   ├── train/good/
    │   ├── test/good/
    │   ├── test/defective/
    │   └── ground_truth/defective/
    ├── grid/
    ├── hazelnut/
    ├── leather/
    ├── wood/
    └── zipper/
```
