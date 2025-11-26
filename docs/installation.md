# 📦 Installation Guide

This comprehensive guide covers all installation methods for Aetherist, from quick setup to advanced deployment configurations.

## 🎯 Quick Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU**: CUDA-capable GPU recommended (NVIDIA RTX series or better)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 50GB free space for models and datasets

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/username/aetherist.git
cd aetherist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import aetherist; print('Installation successful!')"
```

## 🐍 Python Environment Setup

### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n aetherist python=3.10 -y
conda activate aetherist

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Aetherist
pip install -e .
```

### Using Poetry

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### Using pip with virtual environment

```bash
# Create and activate virtual environment
python -m venv aetherist-env
source aetherist-env/bin/activate  # On Windows: aetherist-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install with development dependencies
pip install -e ".[dev]"
```

## 🐋 Docker Installation

### Pre-built Docker Image

```bash
# Pull the official image
docker pull aetherist/aetherist:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 aetherist/aetherist:latest

# Run with volume mounts for persistence
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  aetherist/aetherist:latest
```

### Build from Source

```bash
# Build the image
docker build -t aetherist:local .

# Run the container
docker run --gpus all -p 8000:8000 aetherist:local
```

### Docker Compose Setup

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☸️ Kubernetes Installation

### Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA Device Plugin installed
- Persistent volume provisioner

### Quick Deploy

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=aetherist

# Access the service
kubectl port-forward service/aetherist-service 8000:8000
```

### Helm Installation

```bash
# Add Aetherist Helm repository
helm repo add aetherist https://charts.aetherist.ai
helm repo update

# Install with default values
helm install aetherist aetherist/aetherist

# Install with custom values
helm install aetherist aetherist/aetherist -f values.yaml
```

## 🔧 Development Installation

### Full Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/username/aetherist.git
cd aetherist

# Create development environment
conda env create -f environment-dev.yml
conda activate aetherist-dev

# Install in editable mode with all extras
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### IDE Setup

#### VS Code

```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.mypy-type-checker

# Open workspace
code aetherist.code-workspace
```

#### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter to use conda environment
3. Enable code formatting with Black
4. Configure type checking with MyPy

## 📊 Hardware Requirements

### Minimum Requirements

- **CPU**: 4 cores, 2.5GHz
- **RAM**: 16GB
- **GPU**: 8GB VRAM (GTX 1080, RTX 2060)
- **Storage**: 50GB SSD

### Recommended Requirements

- **CPU**: 8+ cores, 3.0GHz
- **RAM**: 32GB+
- **GPU**: 16GB+ VRAM (RTX 3080, RTX 4080, A100)
- **Storage**: 200GB+ NVMe SSD

### Production Requirements

- **CPU**: 16+ cores, 3.5GHz
- **RAM**: 64GB+
- **GPU**: Multiple GPUs with 24GB+ VRAM each
- **Storage**: 1TB+ high-speed SSD
- **Network**: High-bandwidth connection for model serving

## 🔧 Configuration

### Environment Variables

```bash
# Required
export AETHERIST_MODEL_PATH="/path/to/models"
export AETHERIST_DATA_PATH="/path/to/datasets"

# Optional
export AETHERIST_LOG_LEVEL="INFO"
export AETHERIST_CACHE_DIR="/path/to/cache"
export AETHERIST_NUM_WORKERS="8"
```

### Configuration Files

```bash
# Copy example configurations
cp configs/model_config.example.yaml configs/model_config.yaml
cp configs/train_config.example.yaml configs/train_config.yaml
cp configs/api_config.example.yaml configs/api_config.yaml

# Edit configurations as needed
nano configs/model_config.yaml
```

## ✅ Installation Verification

### Basic Verification

```bash
# Test imports
python -c "
import aetherist
import torch
print(f'Aetherist version: {aetherist.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

### Model Loading Test

```bash
# Test model initialization
python scripts/verify_installation.py
```

### API Server Test

```bash
# Start development server
python -m uvicorn src.api.main:app --reload

# Test API endpoint
curl http://localhost:8000/health

# Check API documentation
open http://localhost:8000/docs
```

### Training Test

```bash
# Run minimal training test
python scripts/test_training.py --steps 10 --no-save
```

## 🚨 Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch size
export AETHERIST_BATCH_SIZE=1

# Enable gradient checkpointing
export AETHERIST_GRADIENT_CHECKPOINTING=true
```

#### Missing Dependencies

```bash
# Install missing system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y build-essential libgl1-mesa-glx libglib2.0-0

# Install missing dependencies (macOS)
brew install pkg-config cairo pango gdk-pixbuf libxml2 libxslt libffi
```

#### Import Errors

```bash
# Ensure proper installation
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Platform-Specific Issues

#### Windows

```powershell
# Install Visual Studio Build Tools
# Use conda for PyTorch installation
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Set environment variables
$env:AETHERIST_MODEL_PATH = "C:\aetherist\models"
```

#### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Use MPS backend for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### Linux

```bash
# Install NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-525 nvidia-cuda-toolkit

# Verify NVIDIA setup
nvidia-smi
nvcc --version
```

## 🔄 Updating

### Update from Git

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall in editable mode
pip install -e .
```

### Update via Package Manager

```bash
# Update via pip
pip install --upgrade aetherist

# Update via conda
conda update aetherist

# Update via poetry
poetry update
```

### Migration Guide

When updating between major versions, check the migration guide:

- [v1.0 to v1.1 Migration](migration/v1.0-to-v1.1.md)
- [v1.1 to v1.2 Migration](migration/v1.1-to-v1.2.md)

## 📞 Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [existing issues](https://github.com/username/aetherist/issues)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Error messages and logs
   - Hardware specifications

---

**Next Steps**: After successful installation, check out the [Getting Started Guide](getting_started.md) for your first avatar generation!

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM
- 10GB disk space
- CPU with AVX support

### Recommended Requirements
- Python 3.9-3.11
- 16GB+ RAM
- 50GB+ SSD storage
- NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- CUDA 11.8 or higher

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/username/aetherist.git
cd aetherist
```

### 2. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv aetherist-env
source aetherist-env/bin/activate  # On Windows: aetherist-env\Scripts\activate

# Or using conda
conda create -n aetherist python=3.9
conda activate aetherist
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt
pip install -e .

# Development installation (includes testing and linting tools)
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

## Platform-Specific Installation

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev build-essential cmake git

# Install CUDA (for GPU support)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# Follow general installation steps
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.9 cmake git

# Install PyTorch with MPS support (Apple Silicon)
pip install torch torchvision torchaudio

# Follow general installation steps
```

### Windows

```powershell
# Install Python from https://python.org or Microsoft Store
# Install Git from https://git-scm.com/download/win
# Install Visual Studio Build Tools

# Install CUDA (for NVIDIA GPUs)
# Download from https://developer.nvidia.com/cuda-downloads

# Open Command Prompt or PowerShell as Administrator
git clone https://github.com/username/aetherist.git
cd aetherist

# Create virtual environment
python -m venv aetherist-env
aetherist-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/username/aetherist.git
cd aetherist

# Build and start services
docker-compose up -d

# Access API at http://localhost:8000
```

### Using Docker Directly

```bash
# Build image
docker build -t aetherist:latest .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data aetherist:latest
```

## GPU Support

### NVIDIA GPUs (CUDA)

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

### AMD GPUs (ROCm)

```bash
# Install ROCm (Linux only)
sudo apt install rocm-dkms

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon (MPS)

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Verification

### Check Installation

```bash
# Test basic import
python -c "import aetherist; print('Installation successful!')"

# Run system check
python scripts/check_installation.py

# Run basic tests
python -m pytest tests/unit/test_basic.py -v
```

### Performance Test

```bash
# Quick performance benchmark
python scripts/benchmark.py --test performance --quick

# Memory test
python scripts/benchmark.py --test memory
```

## Optional Dependencies

### For Advanced Features

```bash
# ONNX export support
pip install onnx onnxruntime-gpu

# TensorRT support (Linux only)
pip install nvidia-tensorrt

# Web interface dependencies
pip install gradio streamlit

# Development tools
pip install black isort mypy pre-commit
```

### For Training

```bash
# Distributed training
pip install torch-distributed

# Experiment tracking
pip install wandb tensorboard

# Data augmentation
pip install albumentations
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or use gradient accumulation
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --batch-size 2
```

**2. Import Errors**
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall in development mode
pip install -e .
```

**3. Dependency Conflicts**
```bash
# Create fresh environment
conda create -n aetherist-clean python=3.9
conda activate aetherist-clean
pip install -r requirements.txt
```

**4. Permission Issues (Linux/macOS)**
```bash
# Fix permissions
sudo chown -R $USER:$USER ./aetherist
chmod +x scripts/*.sh
```

### Getting Help

1. Check the [troubleshooting guide](troubleshooting.md)
2. Search existing [GitHub issues](https://github.com/username/aetherist/issues)
3. Create a new issue with:
   - System information (`python --version`, `pip list`)
   - Complete error message
   - Steps to reproduce

## Configuration

After installation, configure Aetherist:

```bash
# Copy default configuration
cp configs/default_config.yaml configs/my_config.yaml

# Edit configuration
vim configs/my_config.yaml

# Set environment variable
export AETHERIST_CONFIG=configs/my_config.yaml
```

## Next Steps

1. Read the [Getting Started Guide](getting_started.md)
2. Try the [examples](../examples/)
3. Explore the [API documentation](api_reference.md)
4. Join the [community discussions](https://github.com/username/aetherist/discussions)
