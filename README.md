# RSNA Knee Abnormality Detection

Research code for the RSNA Knee Abnormality Detection 2026 competition.

## Objective

Develop a multimodal-inspired MRI-only inference pipeline for detecting 12 clinically important knee abnormalities from multiplanar MRI studies.

## Targets

- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Medial OA
- Lateral OA
- PF OA
- Effusion
- Synovitis
- Baker's
- Contusion
- Fracture

## Current pipeline

1. Load study and series metadata from the competition CSV files.
2. Select one representative Sagittal, Coronal and Axial series per study.
3. Order DICOM slices geometrically using `ImageOrientationPatient` and `ImagePositionPatient`.
4. Normalize MRI intensities using percentile clipping.
5. Uniformly select 16 slices per plane.
6. Resize slices to 288 x 288.
7. Combine official gold labels with report-derived soft pseudo-labels for training.
8. Train a multiplanar vision model and evaluate macro ROC AUC across the 12 targets.

## Data

Competition data, DICOM images, pseudo-label datasets and model checkpoints are not stored in this repository.

Expected Kaggle competition path:

```text
/kaggle/input/competitions/rsna-knee-abnormality-detection
```

## Project structure

```text
rsna-knee-abnormality-detection/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
├── notebooks/
├── scripts/
└── src/
```

## Status

The DICOM loading, geometric slice ordering, MRI preprocessing and multiplanar series-selection pipeline have been validated on the competition data. The next phase is the PyTorch dataset and DINOv2-based training pipeline.
