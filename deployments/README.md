# Aetherist Production Deployment

This directory contains production deployment tools and configurations for the Aetherist 3D-aware image generation model.

## Overview

The deployment system provides:

- **Model Optimization**: Convert models to optimized formats (ONNX, TorchScript, Quantized)
- **Production Server**: FastAPI-based serving infrastructure with async processing
- **Multi-Platform Deployment**: Support for Docker, Kubernetes, AWS, and local deployment
- **Configuration Management**: Environment-specific configurations with YAML/JSON support

## Quick Start

### 1. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# For GPU support, also install:
pip install nvidia-ml-py3
```

### 2. Test Deployment Infrastructure

```bash
python scripts/test_deployment.py
```

### 3. Development Server

```bash
# Start local development server
python scripts/deploy_server.py --config configs/server_config.json
```

### 4. Production Deployment

```bash
# Deploy with Docker
python scripts/deploy_full.py --config configs/deployment_prod.yaml --deployment-type docker

# Deploy to Kubernetes
python scripts/deploy_full.py --config configs/deployment_prod.yaml --deployment-type kubernetes
```

## Components

### Model Optimizer (`src/deployment/model_optimizer.py`)

Optimizes models for production deployment:

- **Quantization**: Reduces model size and improves inference speed
- **ONNX Export**: Cross-platform model format for various runtimes
- **TorchScript**: Optimized PyTorch format for production
- **Benchmarking**: Performance analysis of optimized models

```bash
# Optimize models
python scripts/deploy_optimize.py \
    --generator-checkpoint path/to/generator.pth \
    --output-dir deployments/optimized_models \
    --quantize --export-onnx --benchmark
```

### Model Server (`src/deployment/model_server.py`)

Production-ready API server with:

- **Async Processing**: Non-blocking image generation
- **Batch Processing**: Efficient handling of multiple requests
- **Health Monitoring**: System status and resource monitoring
- **Redis Integration**: Task queue and result caching
- **Authentication**: API key-based security

Server endpoints:
- `GET /health` - System health check
- `POST /generate` - Single image generation
- `POST /generate_batch` - Batch image generation
- `GET /task/{task_id}` - Task status and results
- `GET /image/{task_id}` - Download generated images

### Deployment Manager (`src/deployment/deployment_manager.py`)

Orchestrates complete deployment process:

- **Docker Deployment**: Containerized deployment with GPU support
- **Kubernetes**: Scalable container orchestration
- **AWS Deployment**: Cloud infrastructure with auto-scaling
- **Local Development**: Quick setup for development

## Configuration

### Environment Configurations

- `configs/deployment_dev.yaml` - Development environment
- `configs/deployment_config.yaml` - Base production configuration  
- `configs/deployment_prod.yaml` - Production environment with scaling
- `configs/server_config.json` - Server-specific settings

### Configuration Structure

```yaml
environment: production
deployment_type: kubernetes
model_config_path: configs/high_quality_config.yaml
optimize_models: true

optimization_config:
  quantize: true
  export_onnx: true
  export_torchscript: true

server_config:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  enable_auth: true

kubernetes_config:
  namespace: aetherist-prod
  replicas: 3
  resource_limits:
    cpu: "8"
    memory: "16Gi"
    nvidia.com/gpu: "1"
```

## Deployment Types

### Docker Deployment

Creates containerized deployment with:
- Multi-stage builds for optimization
- GPU support with NVIDIA runtime
- Health checks and monitoring
- Volume mounts for models and configs

Generated files:
- `deployments/docker/Dockerfile`
- `deployments/docker/docker-compose.yml`

```bash
# Build and run Docker container
cd deployments/docker
docker-compose up -d
```

### Kubernetes Deployment

Scalable container orchestration with:
- Horizontal pod autoscaling
- Resource limits and requests
- Load balancing and service discovery
- Ingress for external access

Generated files:
- `deployments/kubernetes/deployment.yaml`
- `deployments/kubernetes/service.yaml`
- `deployments/kubernetes/ingress.yaml` (if configured)

```bash
# Deploy to Kubernetes
kubectl apply -f deployments/kubernetes/
```

### AWS Deployment

Cloud infrastructure with:
- Auto Scaling Groups
- Load Balancers
- GPU-optimized instances
- CloudFormation templates

### Local Development

Quick development setup:
- Virtual environment activation
- Dependency installation
- Development server startup

## Monitoring and Operations

### Health Checks

The deployment includes comprehensive health monitoring:

```bash
# Check server health
curl http://localhost:8000/health
```

Response includes:
- Model loading status
- GPU availability
- Memory usage
- System uptime

### Logs and Debugging

- Application logs: Structured JSON logging
- Server metrics: Request/response times, error rates
- Resource monitoring: CPU, memory, GPU utilization

### Scaling and Load Management

- **Horizontal scaling**: Increase replica count
- **Vertical scaling**: Adjust resource limits
- **Load balancing**: Distribute requests across instances
- **Queue management**: Redis-based task queuing

## Security Considerations

- **API Authentication**: Token-based access control
- **CORS Configuration**: Controlled cross-origin access
- **Resource Limits**: Prevent resource exhaustion
- **Network Policies**: Kubernetes network isolation

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Memory Issues**:
   - Reduce batch size in configuration
   - Enable model quantization
   - Increase resource limits

3. **Model Loading Failures**:
   - Verify checkpoint paths
   - Check model configuration compatibility
   - Review optimization settings

### Debug Mode

Enable debug logging:
```bash
python scripts/deploy_server.py --log-level DEBUG
```

## Performance Optimization

### Model Optimization

- **Quantization**: 2-4x memory reduction, 1.5-3x speed improvement
- **ONNX**: Cross-platform compatibility, potential speed gains
- **TorchScript**: Production-optimized PyTorch models

### Server Optimization

- **Async Processing**: Non-blocking request handling
- **Batch Processing**: Efficient GPU utilization
- **Caching**: Redis-based result caching
- **Connection Pooling**: Efficient database connections

### Infrastructure Optimization

- **GPU Instances**: Use appropriate GPU instance types
- **Auto Scaling**: Dynamic resource allocation based on load
- **Load Balancing**: Distribute traffic efficiently
- **CDN Integration**: Fast image delivery

## Development Workflow

1. **Local Testing**:
   ```bash
   python scripts/test_deployment.py
   python scripts/deploy_server.py
   ```

2. **Model Optimization**:
   ```bash
   python scripts/deploy_optimize.py --generator-checkpoint path/to/model.pth
   ```

3. **Staging Deployment**:
   ```bash
   python scripts/deploy_full.py --config configs/deployment_dev.yaml
   ```

4. **Production Deployment**:
   ```bash
   python scripts/deploy_full.py --config configs/deployment_prod.yaml
   ```

## API Usage Examples

### Python Client

```python
import requests
import json

# Generate single image
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "seed": 42,
        "resolution": [512, 512],
        "guidance_scale": 7.5
    }
)
task = response.json()

# Check status
status = requests.get(f"http://localhost:8000/task/{task['task_id']}")
print(status.json())

# Download image when ready
if status.json()["status"] == "completed":
    image = requests.get(f"http://localhost:8000/image/{task['task_id']}")
    with open("generated.png", "wb") as f:
        f.write(image.content)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Generate image
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "resolution": [512, 512]}'

# Check task status
curl http://localhost:8000/task/task_123456789

# Download generated image
curl -o generated.png http://localhost:8000/image/task_123456789
```

## Support and Maintenance

For production deployments:
- Monitor resource usage and scale accordingly
- Regular model updates and optimizations
- Security updates and dependency management
- Backup and disaster recovery planning
- Performance monitoring and alerting

## License

This deployment system is part of the Aetherist project and follows the same licensing terms.