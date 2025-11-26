# 🚀 Getting Started Guide

Welcome to Aetherist! This guide will walk you through your first avatar generation, from basic usage to advanced features.

## 🎯 Your First Avatar

### Quick Start (5 minutes)

```python
import torch
from aetherist import AetheristGenerator
from aetherist.utils import save_image

# Initialize the generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = AetheristGenerator.from_pretrained("models/aetherist_v1.pth")
generator = generator.to(device)
generator.eval()

# Generate your first avatar
with torch.no_grad():
    # Create random latent code
    latent_code = torch.randn(1, 512, device=device)
    
    # Generate front-facing view
    front_view = generator.generate_front_view(latent_code)
    
    # Save the result
    save_image(front_view, "my_first_avatar.png")
    
print("Avatar saved as 'my_first_avatar.png'!")
```

### Multi-View Generation

```python
from aetherist.utils import generate_camera_poses

# Generate multiple camera angles
camera_poses = generate_camera_poses(
    num_views=8,
    radius=2.5,
    elevation_range=(-0.2, 0.4),
    azimuth_range=(0, 2*3.14159),
    device=device
)

# Generate avatar from all angles
avatar_views = []
for i, camera_params in enumerate(camera_poses):
    view = generator(latent_code, camera_params.unsqueeze(0))
    avatar_views.append(view)
    save_image(view, f"avatar_view_{i:02d}.png")

print(f"Generated {len(avatar_views)} views of your avatar!")
```

## 🎨 Customizing Your Avatar

### Controlled Generation

```python
from aetherist import StyleController, CameraConfig

# Initialize style controller
style_controller = StyleController()

# Define avatar characteristics
avatar_config = {
    "age": "young",           # young, middle-aged, elderly
    "gender": "female",       # male, female, neutral
    "ethnicity": "mixed",     # various options available
    "hair_color": "brown",    # blonde, brown, black, red, etc.
    "expression": "smile"     # smile, neutral, serious, etc.
}

# Generate controlled avatar
controlled_latent = style_controller.encode_attributes(avatar_config)
controlled_avatar = generator(controlled_latent, camera_poses[0].unsqueeze(0))
save_image(controlled_avatar, "controlled_avatar.png")
```

### Style Transfer

```python
from aetherist import StyleTransfer
from PIL import Image

# Load style reference
style_image = Image.open("artistic_style.jpg")
style_transfer = StyleTransfer(generator)

# Apply artistic style
stylized_avatar = style_transfer.transfer(
    latent_code=latent_code,
    style_image=style_image,
    camera_params=camera_poses[0],
    style_weight=0.7
)

save_image(stylized_avatar, "stylized_avatar.png")
```

## 🔧 Configuration and Settings

### Model Configuration

```python
from aetherist import AetheristConfig

# Custom configuration
config = AetheristConfig(
    resolution=1024,          # Output resolution
    triplane_res=128,         # Triplane resolution (higher = better quality)
    neural_renderer_depth=8,  # Neural renderer complexity
    use_mixed_precision=True, # Faster inference
    enable_gradient_checkpointing=False  # Lower memory usage
)

# Load generator with custom config
generator = AetheristGenerator(config=config)
generator.load_state_dict(torch.load("models/aetherist_v1.pth"))
```

### Quality vs Speed Trade-offs

```python
# High quality (slow)
high_quality_config = AetheristConfig(
    resolution=1024,
    triplane_res=128,
    neural_renderer_depth=12,
    supersampling_factor=2
)

# Balanced (recommended)
balanced_config = AetheristConfig(
    resolution=512,
    triplane_res=64,
    neural_renderer_depth=8,
    supersampling_factor=1
)

# Fast (lower quality)
fast_config = AetheristConfig(
    resolution=256,
    triplane_res=32,
    neural_renderer_depth=4,
    supersampling_factor=1
)
```

## 📊 Batch Processing

### Multiple Avatars

```python
from aetherist.batch import BatchProcessor

# Initialize batch processor
processor = BatchProcessor(
    model=generator,
    batch_size=8,
    device=device
)

# Generate multiple random avatars
num_avatars = 16
latent_codes = torch.randn(num_avatars, 512, device=device)

# Process in batches
all_avatars = processor.process_latent_batch(
    latent_codes=latent_codes,
    camera_params=camera_poses[0].unsqueeze(0).repeat(num_avatars, 1)
)

# Save results
for i, avatar in enumerate(all_avatars):
    save_image(avatar, f"batch_avatar_{i:03d}.png")
```

### Batch Style Transfer

```python
# Multiple style references
style_images = [
    Image.open("style1.jpg"),
    Image.open("style2.jpg"),
    Image.open("style3.jpg"),
    Image.open("style4.jpg")
]

# Apply different styles to same avatar
for i, style_img in enumerate(style_images):
    stylized = style_transfer.transfer(
        latent_code=latent_code,
        style_image=style_img,
        camera_params=camera_poses[0]
    )
    save_image(stylized, f"style_{i}_avatar.png")
```

## 🌐 Using the Web Interface

### Launch the Demo

```bash
# Start the interactive web demo
python examples/avatar_generator_demo.py --mode web --port 7860

# Open browser to http://localhost:7860
```

### Web Interface Features

- **Real-time Generation**: Generate avatars as you adjust parameters
- **Interactive Controls**: Sliders for age, gender, style attributes
- **Camera Control**: Adjust viewing angle with mouse
- **Style Gallery**: Choose from preset artistic styles
- **Batch Export**: Generate and download multiple views

### API Integration

```python
import requests
import base64
from io import BytesIO
from PIL import Image

# Start API server first: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Generate avatar via API
response = requests.post(
    "http://localhost:8000/generate/avatar",
    json={
        "num_views": 4,
        "resolution": 512,
        "style_config": {
            "age": "young",
            "gender": "female",
            "expression": "smile"
        },
        "camera_config": {
            "radius": 2.5,
            "elevation_range": [-0.2, 0.4]
        }
    }
)

# Process response
if response.status_code == 200:
    result = response.json()
    for i, img_data in enumerate(result["images"]):
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        img.save(f"api_avatar_{i}.png")
```

## 🔍 Advanced Features

### Latent Space Interpolation

```python
from aetherist.utils import interpolate_latents

# Create two different avatars
latent_a = torch.randn(1, 512, device=device)
latent_b = torch.randn(1, 512, device=device)

# Generate interpolation sequence
num_steps = 10
interpolated_latents = interpolate_latents(latent_a, latent_b, num_steps)

# Generate morphing sequence
for i, latent in enumerate(interpolated_latents):
    avatar = generator(latent, camera_poses[0].unsqueeze(0))
    save_image(avatar, f"morph_{i:02d}.png")

print("Morphing sequence saved! Create a GIF with:")
print("ffmpeg -i morph_%02d.png -vf palettegen palette.png")
print("ffmpeg -i morph_%02d.png -i palette.png -lavfi paletteuse morph.gif")
```

### Camera Animation

```python
from aetherist.utils import create_orbit_camera_path

# Create smooth camera orbit
orbit_cameras = create_orbit_camera_path(
    radius=2.5,
    num_frames=60,
    elevation=0.2,
    full_rotation=True
)

# Generate rotating avatar sequence
for i, camera in enumerate(orbit_cameras):
    frame = generator(latent_code, camera.unsqueeze(0))
    save_image(frame, f"orbit_frame_{i:03d}.png")

print("Orbit animation frames saved! Create video with:")
print("ffmpeg -r 30 -i orbit_frame_%03d.png -c:v libx264 -pix_fmt yuv420p orbit.mp4")
```

### Fine-tuning and Editing

```python
from aetherist.editing import LatentEditor

# Initialize latent editor
editor = LatentEditor(generator)

# Semantic editing
edited_latent = editor.edit_attribute(
    latent_code=latent_code,
    attribute="age",
    direction="older",
    strength=0.5
)

# Generate edited avatar
edited_avatar = generator(edited_latent, camera_poses[0].unsqueeze(0))
save_image(edited_avatar, "edited_avatar.png")
```

## 📈 Performance Optimization

### Model Optimization

```python
from aetherist.optimization import ModelOptimizer

# Optimize model for inference
optimizer = ModelOptimizer()

# Apply optimizations
optimized_generator = optimizer.optimize(
    model=generator,
    optimizations=[
        "torch_compile",      # PyTorch 2.0 compilation
        "mixed_precision",    # FP16 inference
        "fuse_operations",    # Operator fusion
        "optimize_memory"     # Memory layout optimization
    ]
)

# Use optimized model
fast_avatar = optimized_generator(latent_code, camera_poses[0].unsqueeze(0))
```

### Memory Management

```python
import gc

def generate_large_batch(latent_codes, camera_poses):
    results = []
    
    # Process in smaller chunks to avoid OOM
    chunk_size = 4
    for i in range(0, len(latent_codes), chunk_size):
        chunk_latents = latent_codes[i:i+chunk_size]
        chunk_cameras = camera_poses[i:i+chunk_size]
        
        with torch.no_grad():
            chunk_results = generator(chunk_latents, chunk_cameras)
            results.append(chunk_results.cpu())
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    
    return torch.cat(results, dim=0)
```

## 🚨 Troubleshooting Common Issues

### Out of Memory Errors

```python
# Reduce batch size
batch_size = 1

# Enable gradient checkpointing (training only)
generator.enable_gradient_checkpointing()

# Use lower resolution
config.resolution = 256

# Clear cache regularly
torch.cuda.empty_cache()
```

### Poor Quality Results

```python
# Check model weights
assert os.path.exists("models/aetherist_v1.pth"), "Model file not found"

# Verify input ranges
assert latent_code.min() >= -3 and latent_code.max() <= 3, "Latent code out of range"

# Ensure proper normalization
latent_code = torch.clamp(latent_code, -2, 2)

# Use higher resolution
config.resolution = 1024
config.triplane_res = 128
```

### Slow Generation

```python
# Enable compilation (PyTorch 2.0+)
generator = torch.compile(generator)

# Use mixed precision
with torch.cuda.amp.autocast():
    avatar = generator(latent_code, camera_params)

# Reduce complexity
config.neural_renderer_depth = 4
```

## 📚 Next Steps

Congratulations! You've learned the basics of Aetherist. Here are suggested next steps:

1. **Explore Examples**: Check out `examples/` directory for more use cases
2. **Read API Documentation**: [API Reference](api_reference.md)
3. **Train Custom Models**: [Training Guide](training_guide.md)
4. **Deploy to Production**: [Deployment Guide](deployment/production.md)
5. **Join the Community**: [Discord](https://discord.gg/aetherist) | [GitHub Discussions](https://github.com/username/aetherist/discussions)

## 🎯 Quick Reference

### Essential Commands

```bash
# Generate single avatar
python -c "from aetherist import quick_generate; quick_generate()"

# Start web interface
python examples/avatar_generator_demo.py --mode web

# Benchmark performance
python scripts/benchmark.py

# Validate installation
python scripts/verify_installation.py
```

### Key Classes

- `AetheristGenerator`: Main generation model
- `StyleController`: Attribute-based control
- `StyleTransfer`: Artistic style transfer
- `BatchProcessor`: Efficient batch processing
- `CameraConfig`: Camera pose configuration

### Important Files

- `configs/model_config.yaml`: Model settings
- `configs/train_config.yaml`: Training configuration
- `examples/`: Example scripts and demos
- `scripts/`: Utility scripts

---

**Happy avatar generating!** 🎆