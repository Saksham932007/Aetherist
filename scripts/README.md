# Aetherist Data Processing Scripts

This directory contains utilities for preprocessing datasets for Aetherist training.

## preprocess_data.py

Main data preprocessing script with the following commands:

### Process FFHQ Dataset
```bash
python scripts/preprocess_data.py process-ffhq /path/to/ffhq /path/to/output --size 256 --workers 8
```

### Create Train/Val Split
```bash
python scripts/preprocess_data.py split /path/to/dataset --ratio 0.9
```

### Generate Camera Metadata
```bash
python scripts/preprocess_data.py generate-metadata /path/to/dataset
```

### Verify Dataset Integrity
```bash
python scripts/preprocess_data.py verify /path/to/dataset
```

## Features

- **Multi-threaded processing** for fast image resizing
- **Center cropping** to square aspect ratio
- **Quality-controlled JPEG compression**
- **Camera pose generation** for 3D-aware training
- **Dataset verification** with corruption detection
- **Train/validation splitting** with configurable ratios

## Usage Examples

1. **Full FFHQ preprocessing pipeline:**
   ```bash
   # Process raw FFHQ images
   python scripts/preprocess_data.py process-ffhq /data/ffhq-raw /data/ffhq-processed --size 256
   
   # Split into train/val
   python scripts/preprocess_data.py split /data/ffhq-processed --ratio 0.9
   
   # Generate camera metadata
   python scripts/preprocess_data.py generate-metadata /data/ffhq-processed
   
   # Verify final dataset
   python scripts/preprocess_data.py verify /data/ffhq-processed
   ```

2. **Quick processing for smaller datasets:**
   ```bash
   python scripts/preprocess_data.py process-ffhq /data/small-dataset /data/processed --max-images 1000 --workers 4
   ```