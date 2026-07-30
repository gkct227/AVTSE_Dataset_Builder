This directory contains the image quality evaluation script using the **Laplacian Variance** algorithm for blur detection. 

## Overview
The `quality_filter.py` script evaluates image sharpness by calculating the variance of the Laplacian response. It generates a high-contrast synthetic image (with sharp geometric shapes and text) and applies Gaussian blur to benchmark the algorithm.

## Features
- **Blur Assessment**: Calculates Laplacian variance to detect high-frequency edges.
- **Lighting Inspection**: Measures mean pixel intensity to screen extreme brightness or darkness.
- **Edge Visualization**: Saves the absolute Laplacian response maps for visual comparison.

## Quick Start
```bash
python quality_filter.py
```

## Generated Outputs (`output_vis/`)

* `01_clean_with_score.png`: Synthetic image with high-frequency edges and text (Score & PASS label).
* `02_blurry_with_score.png`: Gaussian blurred synthetic image (Score & FAIL label).
* `03_clean_edge_response.png`: Absolute Laplacian response map of the sharp input.
* `04_blurry_edge_response.png`: Absolute Laplacian response map of the blurred input.
