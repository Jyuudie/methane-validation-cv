# methane-validation-cv

## Overview
This repository contains the computer vision pipeline developed to validate methane sampling protocols in dairy cattle. Using **YOLOv8** and **RT-DETR**, the system tracks cow head positions in milking parlors to correlate behavioral data with sniffer sensor logs.

## Key Features
* **Amodal Detection:** robust tracking of cattle heads under severe occlusion (feed bins/stanchions).
* **Architecture Comparison:** Benchmarking CNN (YOLO) vs. Transformer (RT-DETR).
* **Biological Validation:** Algorithmic verification of the 90–410s sampling window via "Biphasic Feeding" analysis.

## Pipeline
1.  **Preprocessing:** Custom scripts to synchronize CVAT label exports with raw video frames.
2.  **Training:** Implementation of Ultralytics YOLOv8/RT-DETR with domain-specific augmentations (Mosaic, MixUp).
3.  **Inference:** Real-time tracking using ByteTrack to generate spatial time-series data.
4.  **Analysis:** Pandas/Matplotlib scripts to derive the "Herd Feeding Index."

## Usage
To reproduce the training loop:
```bash
python 2_training/train_yolo.py --epochs 100 --batch 16
