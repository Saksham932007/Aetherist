# 🔧 Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using Aetherist.

## 🚨 Common Issues

### Installation Problems

#### "No module named 'aetherist'" Error

**Symptoms:**
```python
ImportError: No module named 'aetherist'
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep aetherist
   # or
   conda list aetherist
   ```

2. **Reinstall in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Check Python environment:**
   ```bash
   which python
   python -c "import sys; print(sys.path)"
   ```

4. **Environment mismatch:**
   ```bash
   # Activate correct environment
   conda activate aetherist  # or
   source venv/bin/activate
   ```

#### CUDA/GPU Issues

**Symptoms:**
```
CUDA out of memory
RuntimeError: No CUDA-capable device is detected
```

**Solutions:**

1. **Check CUDA availability:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install CUDA-compatible PyTorch:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Reduce memory usage:**
   ```python
   # Use smaller batch size
   config = AetheristConfig(batch_size=1)
   
   # Enable gradient checkpointing
   generator.enable_gradient_checkpointing()
   
   # Use CPU fallback
   device = torch.device("cpu")
   ```

#### Missing Dependencies

**Symptoms:**
```
ImportError: libGL.so.1: cannot open shared object file
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y build-essential libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install mesa-libGL-devel libX11-devel
```

**macOS:**
```bash
brew install pkg-config cairo pango gdk-pixbuf libxml2 libxslt libffi
```

**Windows:**
- Install Visual Studio Build Tools
- Use conda for complex dependencies:
  ```cmd
  conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
  ```

### Model Loading Issues

#### "Model checkpoint not found" Error

**Symptoms:**
```
FileNotFoundError: Model checkpoint not found at path/to/model.pth
```

**Solutions:**

1. **Download pretrained models:**
   ```bash
   python scripts/download_models.py --model aetherist_v1
   ```

2. **Check model path:**
   ```python
   import os
   model_path = "models/aetherist_v1.pth"
   print(f"Model exists: {os.path.exists(model_path)}")
   ```

3. **Use absolute path:**
   ```python
   import os
   model_path = os.path.abspath("models/aetherist_v1.pth")
   generator = AetheristGenerator.from_pretrained(model_path)
   ```

#### "Model architecture mismatch" Error

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

1. **Check model version:**
   ```python
   checkpoint = torch.load("models/aetherist_v1.pth", map_location="cpu")
   print(f"Model version: {checkpoint.get('version', 'unknown')}")
   ```

2. **Use compatible config:**
   ```python
   # Load config from checkpoint
   config = checkpoint["config"]
   generator = AetheristGenerator(config=config)
   generator.load_state_dict(checkpoint["model_state_dict"])
   ```

3. **Strict loading control:**
   ```python
   # Allow partial loading
   generator.load_state_dict(checkpoint, strict=False)
   ```

### Generation Quality Issues

#### Poor Avatar Quality

**Symptoms:**
- Blurry or distorted avatars
- Inconsistent multi-view results
- Artifacts or noise

**Solutions:**

1. **Check input parameters:**
   ```python
   # Ensure latent codes are in valid range
   latent_code = torch.clamp(latent_code, -2, 2)
   
   # Verify camera parameters
   assert camera_params.shape == (batch_size, 25)
   ```

2. **Use higher resolution:**
   ```python
   config = AetheristConfig(
       resolution=1024,      # Higher output resolution
       triplane_res=128,     # Higher triplane resolution
       neural_renderer_depth=8  # More rendering layers
   )
   ```

3. **Improve model settings:**
   ```python
   generator.eval()  # Set to evaluation mode
   
   with torch.no_grad():  # Disable gradients for inference
       avatar = generator(latent_code, camera_params)
   ```

#### Inconsistent Multi-view Results

**Symptoms:**
- Different views don't match
- Identity changes between angles
- Geometric inconsistencies

**Solutions:**

1. **Verify camera pose generation:**
   ```python
   from aetherist.utils import validate_camera_poses
   
   camera_poses = generate_camera_poses(num_views=8)
   validate_camera_poses(camera_poses)  # Check for valid ranges
   ```

2. **Use consistent random seed:**
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   ```

3. **Check triplane consistency:**
   ```python
   # Enable consistency loss during generation
   generator.enable_consistency_loss()
   ```

### Performance Issues

#### Slow Generation Speed

**Symptoms:**
- Long generation times
- High memory usage
- CPU bottlenecks

**Solutions:**

1. **Enable optimizations:**
   ```python
   # Compile model (PyTorch 2.0+)
   generator = torch.compile(generator)
   
   # Use mixed precision
   with torch.cuda.amp.autocast():
       avatar = generator(latent_code, camera_params)
   ```

2. **Optimize model configuration:**
   ```python
   fast_config = AetheristConfig(
       resolution=512,           # Lower resolution
       triplane_res=64,          # Smaller triplane
       neural_renderer_depth=4,  # Fewer layers
       batch_size=1             # Single sample
   )
   ```

3. **Profile performance:**
   ```python
   import cProfile
   
   def generate_avatar():
       return generator(latent_code, camera_params)
   
   cProfile.run('generate_avatar()', 'generation_profile.prof')
   ```

#### Memory Issues

**Symptoms:**
```
RuntimeError: CUDA out of memory
RuntimeError: DefaultCPUAllocator: not enough memory
```

**Solutions:**

1. **Monitor memory usage:**
   ```python
   import psutil
   import GPUtil
   
   # CPU memory
   print(f"RAM usage: {psutil.virtual_memory().percent}%")
   
   # GPU memory
   gpus = GPUtil.getGPUs()
   for gpu in gpus:
       print(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
   ```

2. **Implement memory management:**
   ```python
   def generate_with_memory_management(latent_codes, camera_poses):
       results = []
       
       for latent, camera in zip(latent_codes, camera_poses):
           # Clear cache before each generation
           torch.cuda.empty_cache()
           
           with torch.no_grad():
               result = generator(latent.unsqueeze(0), camera.unsqueeze(0))
               results.append(result.cpu())  # Move to CPU immediately
           
           # Force garbage collection
           import gc
           gc.collect()
       
       return torch.cat(results, dim=0)
   ```

3. **Use gradient checkpointing:**
   ```python
   # For training only
   generator.enable_gradient_checkpointing()
   ```

### API and Web Interface Issues

#### API Server Won't Start

**Symptoms:**
```
uvicorn: command not found
Port already in use
```

**Solutions:**

1. **Install FastAPI dependencies:**
   ```bash
   pip install "fastapi[all]" uvicorn
   ```

2. **Check port availability:**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Use different port
   uvicorn src.api.main:app --port 8001
   ```

3. **Run with proper permissions:**
   ```bash
   # Linux/macOS
   sudo uvicorn src.api.main:app --host 0.0.0.0 --port 80
   ```

#### Web Demo Not Loading

**Symptoms:**
- Blank page in browser
- JavaScript errors
- Connection refused

**Solutions:**

1. **Check demo dependencies:**
   ```bash
   pip install gradio streamlit pillow
   ```

2. **Verify server status:**
   ```bash
   curl http://localhost:7860/health
   ```

3. **Check firewall settings:**
   ```bash
   # Open port in firewall (Linux)
   sudo ufw allow 7860
   
   # Check Windows Firewall settings
   ```

### Training Issues

#### Training Won't Start

**Symptoms:**
```
Dataset not found
Configuration error
CUDA initialization failed
```

**Solutions:**

1. **Verify dataset:**
   ```bash
   python scripts/verify_dataset.py --dataset-path /path/to/dataset
   ```

2. **Check configuration:**
   ```bash
   python scripts/validate_config.py --config configs/train_config.yaml
   ```

3. **Test training setup:**
   ```bash
   python scripts/test_training.py --dry-run
   ```

#### Training Crashes or Diverges

**Symptoms:**
- NaN losses
- Memory errors during training
- Model outputs become invalid

**Solutions:**

1. **Monitor training:**
   ```python
   # Add gradient clipping
   config.gradient_clipping = 1.0
   
   # Reduce learning rate
   config.learning_rate = 1e-5
   
   # Enable gradient monitoring
   config.log_gradients = True
   ```

2. **Check data preprocessing:**
   ```python
   # Verify data ranges
   assert images.min() >= -1 and images.max() <= 1
   
   # Check for NaN values
   assert not torch.isnan(images).any()
   ```

3. **Use stable training settings:**
   ```yaml
   # configs/stable_train_config.yaml
   optimizer:
     type: Adam
     lr: 0.0002
     betas: [0.5, 0.999]
   
   training:
     batch_size: 8
     gradient_clipping: 1.0
     mixed_precision: true
   ```

### Platform-Specific Issues

#### Windows Issues

**Common Problems:**
- Long path names
- Permission errors
- DLL loading issues

**Solutions:**

1. **Enable long paths:**
   ```cmd
   # Run as Administrator
   reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
   ```

2. **Use conda for dependencies:**
   ```cmd
   conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
   conda install pillow opencv numpy
   ```

3. **Set environment variables:**
   ```cmd
   set PYTHONPATH=%PYTHONPATH%;%CD%\src
   set AETHERIST_MODEL_PATH=C:\aetherist\models
   ```

#### macOS Issues

**Common Problems:**
- Apple Silicon compatibility
- OpenMP issues
- Permission errors

**Solutions:**

1. **Install Xcode tools:**
   ```bash
   xcode-select --install
   ```

2. **Use MPS backend (Apple Silicon):**
   ```python
   if torch.backends.mps.is_available():
       device = torch.device("mps")
   else:
       device = torch.device("cpu")
   ```

3. **Install OpenMP:**
   ```bash
   brew install libomp
   ```

#### Linux Issues

**Common Problems:**
- NVIDIA driver conflicts
- Permission errors
- Library version mismatches

**Solutions:**

1. **Update NVIDIA drivers:**
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-525
   
   # Verify installation
   nvidia-smi
   ```

2. **Fix library issues:**
   ```bash
   # Install missing libraries
   sudo apt install libcudnn8-dev libcublas-dev
   
   # Update library path
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## 🔍 Debugging Tools

### Diagnostic Script

```python
#!/usr/bin/env python3
"""
Aetherist Diagnostic Script
Run this to check your installation and identify issues.
"""

import sys
import os
import torch
import importlib.util

def check_python_version():
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_pytorch():
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("✅ PyTorch OK")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def check_aetherist():
    try:
        spec = importlib.util.find_spec("aetherist")
        if spec is None:
            print("❌ Aetherist not installed")
            return False
        
        import aetherist
        print(f"Aetherist version: {aetherist.__version__}")
        print(f"Aetherist path: {aetherist.__file__}")
        print("✅ Aetherist OK")
        return True
    except Exception as e:
        print(f"❌ Aetherist error: {e}")
        return False

def check_models():
    model_paths = [
        "models/aetherist_v1.pth",
        "models/discriminator.pth"
    ]
    
    found_models = []
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**3)  # GB
            print(f"✅ Found {path} ({size:.1f}GB)")
            found_models.append(path)
        else:
            print(f"❌ Missing {path}")
    
    return len(found_models) > 0

def check_dependencies():
    required_packages = [
        "numpy", "PIL", "torchvision", "matplotlib",
        "opencv-cv2", "tqdm", "yaml", "scipy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    return len(missing) == 0

def main():
    print("🔍 Aetherist Diagnostic Tool")
    print("=" * 40)
    
    checks = [
        ("Python version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Aetherist", check_aetherist),
        ("Models", check_models),
        ("Dependencies", check_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 40)
    print("📊 Summary:")
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All checks passed! Aetherist is ready to use.")
    else:
        print("\n⚠️  Some checks failed. See troubleshooting guide for solutions.")

if __name__ == "__main__":
    main()
```

Save this as `scripts/diagnose.py` and run:
```bash
python scripts/diagnose.py
```

### Performance Profiler

```python
import time
import torch
import cProfile
import pstats
from aetherist import AetheristGenerator

def profile_generation():
    """Profile avatar generation performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = AetheristGenerator.from_pretrained("models/aetherist_v1.pth")
    generator = generator.to(device)
    generator.eval()
    
    latent_code = torch.randn(1, 512, device=device)
    camera_params = torch.randn(1, 25, device=device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = generator(latent_code, camera_params)
    
    # Profile
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            avatar = generator(latent_code, camera_params)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"Average generation time: {avg_time:.3f}s")
    print(f"Throughput: {1/avg_time:.1f} avatars/second")
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Peak GPU memory: {max_memory:.2f}GB")

# Run profiler
cProfile.run('profile_generation()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)
```

## 📞 Getting Help

### Before Asking for Help

1. **Run diagnostic script:** `python scripts/diagnose.py`
2. **Check existing issues:** [GitHub Issues](https://github.com/username/aetherist/issues)
3. **Review documentation:** [Full Documentation](https://aetherist.readthedocs.io/)

### Reporting Issues

When reporting a bug, include:

1. **System information:**
   ```bash
   python --version
   pip list | grep torch
   nvidia-smi  # If using GPU
   uname -a    # Linux/macOS
   ```

2. **Error message:**
   ```
   Full traceback and error message
   ```

3. **Minimal reproduction code:**
   ```python
   # Code that reproduces the issue
   ```

4. **Expected vs actual behavior:**
   - What you expected to happen
   - What actually happened

### Community Support

- **GitHub Discussions:** [https://github.com/username/aetherist/discussions](https://github.com/username/aetherist/discussions)
- **Discord:** [https://discord.gg/aetherist](https://discord.gg/aetherist)
- **Stack Overflow:** Use tag `aetherist`
- **Email:** support@aetherist.ai

---

**Remember:** Most issues have been encountered before. Check the documentation and existing issues first!