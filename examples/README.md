# Aetherist Examples Collection
# Comprehensive examples demonstrating various use cases

This directory contains practical examples showing how to use Aetherist for different applications and scenarios.

## 🚀 Quick Start Examples

### [`basic_usage.py`](basic_usage.py)
Basic image generation and manipulation examples.

```python
python examples/basic_usage.py
```

**Features demonstrated:**
- Simple image generation
- Style transfer
- Attribute editing
- Batch processing

### [`api_client_demo.py`](api_client_demo.py)
Complete API client implementation and usage.

```python
python examples/api_client_demo.py
```

**Features demonstrated:**
- REST API integration
- Authentication handling
- Async operations
- Error handling

## 🎨 Creative Applications

### [`artistic_style_transfer.py`](artistic_style_transfer.py)
Advanced artistic style transfer with multiple techniques.

```python
python examples/artistic_style_transfer.py --input portrait.jpg --style vangogh
```

**Features demonstrated:**
- Multiple artistic styles
- Style strength control
- Batch style application
- Custom style training

### [`face_editing_suite.py`](face_editing_suite.py)
Comprehensive face editing and enhancement tools.

```python
python examples/face_editing_suite.py --input face.jpg --edit age=+10,smile=0.8
```

**Features demonstrated:**
- Age progression/regression
- Expression manipulation
- Pose adjustment
- Identity preservation

### [`virtual_fashion.py`](virtual_fashion.py)
Virtual try-on and fashion design applications.

```python
python examples/virtual_fashion.py --person model.jpg --clothing shirt.jpg
```

**Features demonstrated:**
- Clothing transfer
- Fabric simulation
- Fit adjustment
- Color variation

## 🎬 Animation and Video

### [`video_processing.py`](video_processing.py)
Video-based generation and editing workflows.

```python
python examples/video_processing.py --input video.mp4 --effect aging
```

**Features demonstrated:**
- Frame-by-frame processing
- Temporal consistency
- Video style transfer
- Motion preservation

### [`animated_generation.py`](animated_generation.py)
Creating smooth animations between different generated images.

```python
python examples/animated_generation.py --start seed1 --end seed2 --frames 60
```

**Features demonstrated:**
- Latent space interpolation
- Smooth transitions
- GIF/MP4 export
- Custom trajectories

## 🏢 Professional Workflows

### [`batch_processing.py`](batch_processing.py)
High-performance batch processing for production use.

```python
python examples/batch_processing.py --input_dir photos/ --output_dir results/ --workers 4
```

**Features demonstrated:**
- Multi-threading
- Progress tracking
- Error recovery
- Output organization

### [`quality_assessment.py`](quality_assessment.py)
Automated quality control and image assessment.

```python
python examples/quality_assessment.py --input_dir generated/ --threshold 0.8
```

**Features demonstrated:**
- Quality scoring
- Automatic filtering
- Batch analysis
- Report generation

### [`model_comparison.py`](model_comparison.py)
Compare different model versions and configurations.

```python
python examples/model_comparison.py --models v1,v2,v3 --test_set validation/
```

**Features demonstrated:**
- A/B testing
- Performance metrics
- Visual comparisons
- Statistical analysis

## 🔬 Research and Development

### [`latent_space_exploration.py`](latent_space_exploration.py)
Tools for understanding and exploring the model's latent space.

```python
python examples/latent_space_exploration.py --mode interactive
```

**Features demonstrated:**
- Latent vector manipulation
- Semantic direction discovery
- Clustering analysis
- Visualization tools

### [`training_experiments.py`](training_experiments.py)
Experimental training configurations and techniques.

```python
python examples/training_experiments.py --config experiments/config1.yaml
```

**Features demonstrated:**
- Custom loss functions
- Regularization techniques
- Data augmentation
- Hyperparameter optimization

### [`ablation_studies.py`](ablation_studies.py)
Systematic component analysis and ablation studies.

```python
python examples/ablation_studies.py --component triplane --disable_layers 2,4,6
```

**Features demonstrated:**
- Component isolation
- Performance impact analysis
- Systematic testing
- Result visualization

## 🌐 Web Integration

### [`gradio_custom_interface.py`](gradio_custom_interface.py)
Custom Gradio interfaces for specific use cases.

```python
python examples/gradio_custom_interface.py
```

**Features demonstrated:**
- Custom UI components
- Real-time processing
- User interaction handling
- Result sharing

### [`streamlit_dashboard.py`](streamlit_dashboard.py)
Comprehensive Streamlit dashboard for model interaction.

```python
streamlit run examples/streamlit_dashboard.py
```

**Features demonstrated:**
- Interactive dashboards
- Parameter visualization
- Batch operations
- Export functionality

## 🔧 Utilities and Tools

### [`model_conversion.py`](model_conversion.py)
Convert between different model formats and optimize for deployment.

```python
python examples/model_conversion.py --input model.pth --output model.onnx --optimize
```

**Features demonstrated:**
- Format conversion
- Model optimization
- Quantization
- Compatibility testing

### [`dataset_preparation.py`](dataset_preparation.py)
Tools for preparing and validating training datasets.

```python
python examples/dataset_preparation.py --input raw_data/ --output processed/ --validate
```

**Features demonstrated:**
- Data preprocessing
- Quality validation
- Format standardization
- Metadata extraction

### [`performance_profiling.py`](performance_profiling.py)
Comprehensive performance analysis and optimization.

```python
python examples/performance_profiling.py --model aetherist_v1 --profile memory,speed
```

**Features demonstrated:**
- Memory profiling
- Speed analysis
- Bottleneck identification
- Optimization suggestions

## 📚 Learning Resources

### [`tutorial_notebooks/`](tutorial_notebooks/)
Interactive Jupyter notebooks for learning Aetherist.

- **`01_getting_started.ipynb`** - Basic concepts and first steps
- **`02_advanced_generation.ipynb`** - Advanced generation techniques
- **`03_custom_training.ipynb`** - Training your own models
- **`04_api_integration.ipynb`** - API and web integration

### [`use_case_studies/`](use_case_studies/)
Real-world applications and case studies.

- **`photography_enhancement/`** - Professional photography workflows
- **`content_creation/`** - Digital content and media production
- **`research_applications/`** - Academic and research use cases
- **`commercial_deployment/`** - Enterprise deployment strategies

## 🔄 Running Examples

### Prerequisites
```bash
# Install Aetherist with all dependencies
pip install aetherist[all]

# Or install specific requirements
pip install -r requirements.txt
pip install -r requirements-examples.txt
```

### Basic Execution
```bash
# Simple example
python examples/basic_usage.py

# With custom parameters
python examples/face_editing.py --input photo.jpg --age +5

# Batch processing
python examples/batch_processing.py --input_dir photos/ --workers 4
```

### Environment Variables
```bash
# API configuration
export AETHERIST_API_KEY="your-api-key"
export AETHERIST_MODEL_PATH="./models/aetherist_v1.pth"

# Performance tuning
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
```

## 📊 Performance Notes

- **GPU Memory**: Examples require 4-8GB VRAM for optimal performance
- **CPU Fallback**: All examples work on CPU but may be significantly slower
- **Batch Size**: Adjust batch sizes based on available memory
- **Model Loading**: First run may take longer due to model download/loading

## 🤝 Contributing Examples

We welcome community contributions! Guidelines for adding examples:

1. **Clear Documentation**: Each example should have comprehensive docstrings
2. **Error Handling**: Include proper error handling and user feedback
3. **Parameterization**: Use command-line arguments for key parameters
4. **Resource Management**: Clean up resources and handle memory efficiently
5. **Testing**: Include unit tests for complex examples

### Template Structure
```python
#!/usr/bin/env python3
"""
Example Title: Brief description

This example demonstrates [specific features/techniques].

Usage:
    python examples/example_name.py [arguments]

Requirements:
    - List specific requirements
    - Hardware recommendations
    - Model dependencies
"""

import argparse
import logging
from pathlib import Path

# Import Aetherist components
from aetherist import AetheristModel
from aetherist.utils import setup_logging

def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="Example description")
    # Add arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Example implementation
    # ...

if __name__ == "__main__":
    main()
```

## 📈 Performance Benchmarks

Example performance on different hardware configurations:

| Example | RTX 4090 | RTX 3080 | CPU (32 cores) | Memory Usage |
|---------|----------|----------|-----------------|--------------|
| Basic Usage | 0.1s | 0.2s | 2.5s | 2GB |
| Batch Processing | 5.0s/100imgs | 12s/100imgs | 180s/100imgs | 4GB |
| Video Processing | 2fps | 1.2fps | 0.1fps | 6GB |
| Training | 45min/epoch | 90min/epoch | 12hr/epoch | 8GB |

## 🔗 Related Resources

- **[API Documentation](../docs/api_reference.md)** - Complete API reference
- **[Training Guide](../docs/training.md)** - Model training documentation  
- **[Deployment Guide](../docs/deployment/)** - Production deployment
- **[Community Forum](https://github.com/aetherist/community)** - Discussion and support

---

For more information and support, visit our [documentation](../docs/) or [GitHub repository](https://github.com/aetherist/aetherist).