# Image Segmentation

This repository reads probability rasters from GeoTIFF files and performs image segmentation to detect and extract image regions based on their probability values.

## Installation

### 1. Create and activate the Conda environment

```bash
conda create -n image_segmentation python=3.12
conda activate image_segmentation
```

### 2. Install Python packages

```bash
pip install scikit-image tifffile matplotlib numpy
```

### 3. Install required Conda packages

```bash
conda install -c conda-forge gdal shapely
```

### 4. Optional: Install Jupyter support

If you want to use Jupyter notebooks, install `ipykernel`:

```bash
conda install -c conda-forge ipykernel
```