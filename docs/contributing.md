# 🤝 Contributing Guide

Welcome to Aetherist! We're excited to have you contribute to advancing 3D avatar generation. This guide will help you get started and make meaningful contributions.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Report bugs and issues
- Provide detailed reproduction steps
- Include system information and logs

### ✨ Feature Requests
- Suggest new features and improvements
- Discuss implementation approaches
- Provide use cases and requirements

### 💻 Code Contributions
- Fix bugs and implement features
- Improve performance and optimization
- Add tests and documentation

### 📚 Documentation
- Improve existing documentation
- Add tutorials and examples
- Translate documentation

### 🧪 Testing
- Write and improve tests
- Test new features and bug fixes
- Report test results and coverage

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/aetherist.git
cd aetherist

# Add upstream remote
git remote add upstream https://github.com/username/aetherist.git
```

### 2. Development Setup

```bash
# Create development environment
conda create -n aetherist-dev python=3.10 -y
conda activate aetherist-dev

# Install in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### 3. Development Dependencies

```bash
# Core development tools
pip install black isort mypy flake8 pytest pytest-cov

# Documentation tools
pip install sphinx sphinx-rtd-theme myst-parser

# Additional tools
pip install jupyter ipython tqdm
```

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Always create a new branch for your work
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow our coding standards and best practices:

```python
# Example: Adding a new feature
class NewFeature:
    """
    Brief description of the feature.
    
    Args:
        param1 (type): Description of parameter 1
        param2 (type): Description of parameter 2
    
    Examples:
        >>> feature = NewFeature(param1="value", param2=42)
        >>> result = feature.process()
        >>> assert isinstance(result, ExpectedType)
    """
    
    def __init__(self, param1: str, param2: int):
        self.param1 = param1
        self.param2 = param2
        
    def process(self) -> SomeType:
        """Process the feature and return result."""
        # Implementation here
        return result
```

### 3. Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/performance/ -m performance

# Run integration tests
pytest tests/integration/ -m integration
```

### 4. Code Quality

```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run all quality checks
pre-commit run --all-files
```

### 5. Documentation

```bash
# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html

# Run documentation tests
make doctest
```

### 6. Commit and Push

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "✨ Add new feature for avatar style transfer

- Implement StyleTransfer class with VGG-based loss
- Add support for multiple artistic styles
- Include comprehensive tests and documentation
- Update API endpoints for style transfer

Fixes #123"

# Push to your fork
git push origin feature/your-feature-name
```

## 📝 Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# String quotes: Double quotes preferred
# Import order: isort configuration

# Good examples
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np

from aetherist.models.base import BaseModel
from aetherist.utils.camera import CameraUtils


class ExampleClass(BaseModel):
    """Example class following our coding standards."""
    
    def __init__(
        self,
        required_param: str,
        optional_param: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.required_param = required_param
        self.optional_param = optional_param or 10
    
    def public_method(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Public method with clear documentation.
        
        Args:
            input_data: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Processed tensor of same shape as input
            
        Raises:
            ValueError: If input_data is empty or invalid shape
        """
        if input_data.numel() == 0:
            raise ValueError("Input data cannot be empty")
            
        return self._private_method(input_data)
    
    def _private_method(self, data: torch.Tensor) -> torch.Tensor:
        """Private method (internal implementation)."""
        # Implementation details
        return data * 2.0
```

### Documentation Standards

```python
def example_function(
    param1: str,
    param2: int,
    param3: Optional[bool] = None
) -> Dict[str, Union[str, int]]:
    """
    Brief one-line description of the function.
    
    Longer description if needed. Explain what the function does,
    any important behaviors, and how it fits into the larger system.
    
    Args:
        param1: Description of parameter 1. Include type information
            if not obvious from type hints.
        param2: Description of parameter 2. Mention valid ranges
            or constraints if applicable.
        param3: Optional parameter with default behavior.
            Defaults to None, which means...
    
    Returns:
        Dictionary containing processed results with keys:
        - 'result': Processed string value
        - 'count': Number of processing steps
    
    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string
    
    Examples:
        Basic usage:
        >>> result = example_function("hello", 42)
        >>> assert result['result'] == "processed_hello"
        
        With optional parameter:
        >>> result = example_function("world", 10, True)
        >>> assert result['count'] > 0
    
    Note:
        This function is thread-safe and can be used in parallel processing.
        For large inputs, consider using the batch processing version.
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
        
    # Implementation
    return {"result": f"processed_{param1}", "count": param2}
```

### Testing Standards

```python
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from aetherist.models import AetheristGenerator
from aetherist.testing import create_test_config, create_test_batch


class TestAetheristGenerator:
    """Test suite for AetheristGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator instance."""
        config = create_test_config()
        return AetheristGenerator(config)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample test data."""
        return create_test_batch(batch_size=2)
    
    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly."""
        assert isinstance(generator, AetheristGenerator)
        assert generator.latent_dim == 512
        assert generator.triplane_dim == 256
    
    def test_forward_pass_shape(self, generator, sample_batch):
        """Test that forward pass produces correct output shape."""
        latent_codes, camera_params = sample_batch
        
        with torch.no_grad():
            output = generator(latent_codes, camera_params)
        
        expected_shape = (2, 3, 512, 512)  # batch, channels, height, width
        assert output.shape == expected_shape
    
    def test_forward_pass_values(self, generator, sample_batch):
        """Test that forward pass produces valid output values."""
        latent_codes, camera_params = sample_batch
        
        with torch.no_grad():
            output = generator(latent_codes, camera_params)
        
        # Check output is in valid range for images
        assert output.min() >= -1.0
        assert output.max() <= 1.0
        
        # Check no NaN or infinite values
        assert torch.isfinite(output).all()
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, generator, batch_size):
        """Test generator with different batch sizes."""
        latent_codes = torch.randn(batch_size, 512)
        camera_params = torch.randn(batch_size, 25)
        
        with torch.no_grad():
            output = generator(latent_codes, camera_params)
        
        assert output.shape[0] == batch_size
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self, generator):
        """Test that generator works on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        generator = generator.cuda()
        latent_codes = torch.randn(1, 512).cuda()
        camera_params = torch.randn(1, 25).cuda()
        
        with torch.no_grad():
            output = generator(latent_codes, camera_params)
        
        assert output.device.type == 'cuda'
    
    def test_error_handling(self, generator):
        """Test proper error handling for invalid inputs."""
        # Wrong latent dimension
        with pytest.raises(ValueError):
            generator(torch.randn(1, 256), torch.randn(1, 25))
        
        # Wrong camera parameter dimension
        with pytest.raises(ValueError):
            generator(torch.randn(1, 512), torch.randn(1, 20))
    
    @patch('aetherist.models.neural_renderer.NeuralRenderer')
    def test_mocked_components(self, mock_renderer, generator):
        """Test with mocked components."""
        mock_renderer.return_value.forward.return_value = torch.zeros(1, 3, 512, 512)
        
        # Test interaction with mocked component
        output = generator(torch.randn(1, 512), torch.randn(1, 25))
        
        mock_renderer.assert_called_once()
        assert output.shape == (1, 3, 512, 512)
```

## 🎯 Contribution Areas

### 🏗️ Core Model Development

**Skills needed:** PyTorch, 3D graphics, neural networks

```python
# Example: Implementing a new neural renderer
class ImprovedNeuralRenderer(nn.Module):
    """
    Improved neural renderer with better quality and performance.
    
    Focus areas:
    - Faster rendering algorithms
    - Better quality outputs
    - Memory efficiency improvements
    """
    pass
```

**Getting started:**
- Read the architecture documentation
- Study existing neural renderer implementations
- Propose improvements in GitHub issues
- Start with small optimizations

### 🎨 Style Transfer and Control

**Skills needed:** Computer vision, GAN training, image processing

```python
# Example: New style transfer method
class AdvancedStyleTransfer:
    """
    Advanced style transfer with better preservation of 3D structure.
    
    Ideas to explore:
    - Multi-scale style transfer
    - 3D-aware style loss
    - Real-time style adaptation
    """
    pass
```

### 🌐 API and Web Interface

**Skills needed:** FastAPI, React/Vue.js, web development

```python
# Example: New API endpoint
@app.post("/generate/video")
async def generate_avatar_video(request: VideoGenerationRequest):
    """
    Generate rotating avatar video.
    
    New features to add:
    - Video generation endpoints
    - Real-time streaming
    - WebRTC integration
    """
    pass
```

### 📊 Performance and Optimization

**Skills needed:** CUDA, optimization, profiling

```python
# Example: CUDA kernel optimization
def optimize_triplane_sampling():
    """
    Optimize triplane feature sampling with custom CUDA kernels.
    
    Areas for improvement:
    - Memory access patterns
    - Compute utilization
    - Batch processing efficiency
    """
    pass
```

### 🧪 Testing and Quality Assurance

**Skills needed:** Testing frameworks, CI/CD, quality assurance

```python
# Example: Performance regression tests
def test_generation_performance():
    """
    Ensure generation performance doesn't regress.
    
    Tests to add:
    - Benchmark tests
    - Memory usage tests
    - Quality metrics tests
    """
    pass
```

### 📚 Documentation and Examples

**Skills needed:** Technical writing, example creation

- Improve existing documentation
- Create new tutorials and guides
- Add more examples and demos
- Translate documentation to other languages

## 🔄 Review Process

### Pull Request Guidelines

1. **Clear Description**
   ```markdown
   ## Summary
   Brief description of what this PR does.
   
   ## Changes
   - List of specific changes made
   - New features added
   - Bug fixes included
   
   ## Testing
   - How was this tested?
   - What test cases were added?
   
   ## Breaking Changes
   - Any breaking changes?
   - Migration guide if needed
   ```

2. **Small, Focused PRs**
   - One feature or bug fix per PR
   - Keep changes as small as possible
   - Split large changes into multiple PRs

3. **Tests Required**
   - All new code must have tests
   - Tests should cover edge cases
   - Maintain or improve code coverage

### Review Criteria

**Reviewers will check:**
- ✅ Code quality and style compliance
- ✅ Test coverage and quality
- ✅ Documentation completeness
- ✅ Performance impact
- ✅ Backward compatibility
- ✅ Security considerations

### Review Process Steps

1. **Automated Checks**
   - CI/CD pipeline runs automatically
   - Code quality checks (linting, formatting)
   - Test suite execution
   - Documentation build verification

2. **Manual Review**
   - Code review by maintainers
   - Design discussion if needed
   - Feedback and iteration
   - Final approval

3. **Merge**
   - Squash and merge preferred
   - Clear commit message
   - Update changelog if needed

## 🏆 Recognition

We value all contributions and recognize contributors in several ways:

### 📋 Contributor List
All contributors are listed in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

### 🎖️ Special Recognition
- Outstanding contributions featured in releases
- Contributor spotlight in documentation
- Special thanks in research papers

### 📈 Contribution Stats
Track your contributions on our contributor dashboard:
- Number of PRs merged
- Issues resolved
- Lines of code contributed
- Documentation improvements

## 📞 Getting Help

### 💬 Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions  
- **Discord**: Real-time chat with the community
- **Email**: Maintainer contact for sensitive issues

### 🆘 Getting Unstuck

If you're stuck or need help:

1. **Check existing resources**
   - Documentation and tutorials
   - Existing GitHub issues
   - Community discussions

2. **Ask for help**
   - Create a GitHub issue with "help wanted" label
   - Ask in Discord
   - Reach out to mentors

3. **Pair programming**
   - Schedule sessions with maintainers
   - Collaborate with other contributors

## 🎯 Mentorship Program

### For New Contributors

We offer mentorship for new contributors:

- **Onboarding sessions** to get familiar with the codebase
- **Guided first contributions** with step-by-step help
- **Regular check-ins** to answer questions and provide feedback

### Becoming a Mentor

Experienced contributors can become mentors:

- **Help new contributors** get started
- **Review pull requests** and provide feedback
- **Lead feature development** and guide discussions

## 📋 Code of Conduct

We are committed to providing a welcoming and inclusive environment:

### Our Standards

**Positive behaviors:**
- Being respectful and inclusive
- Constructive feedback and criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behaviors:**
- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Unprofessional conduct

### Enforcement

Violations of the code of conduct should be reported to the maintainers. All reports will be reviewed and investigated promptly and fairly.

## 🚀 Future Roadmap

### Short-term Goals (3 months)
- Improve generation quality and speed
- Add more style transfer options
- Better documentation and examples
- Mobile deployment support

### Medium-term Goals (6 months)
- Video generation capabilities
- Real-time inference optimization
- Advanced editing features
- Multi-modal conditioning

### Long-term Goals (12+ months)
- Full-body avatar generation
- Animation and rigging support
- VR/AR integration
- Federated learning capabilities

---

**Thank you for contributing to Aetherist!** 🎆 

Your contributions help advance the state-of-the-art in 3D avatar generation and make this technology accessible to everyone.