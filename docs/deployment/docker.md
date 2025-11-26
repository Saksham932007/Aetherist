# 🐋 Docker Deployment Guide

This guide covers deploying Aetherist using Docker containers for development, staging, and production environments.

## 🎯 Quick Start

### Single Container Deployment

```bash
# Pull and run the latest image
docker run -d \
  --name aetherist \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  aetherist/aetherist:latest
```

### Docker Compose Deployment

```bash
# Clone the repository
git clone https://github.com/username/aetherist.git
cd aetherist

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f aetherist
```

## 🏗️ Docker Image Details

### Pre-built Images

Available on Docker Hub:

- `aetherist/aetherist:latest` - Latest stable release
- `aetherist/aetherist:1.0` - Specific version
- `aetherist/aetherist:1.0-gpu` - GPU-optimized image
- `aetherist/aetherist:1.0-cpu` - CPU-only image

### Image Specifications

**Base Image**: `nvidia/cuda:11.8-devel-ubuntu22.04`

**Installed Components**:
- Python 3.10
- PyTorch 2.0+ with CUDA support
- Aetherist and all dependencies
- NGINX (for production)
- Redis (for caching)

**Image Size**: ~8GB (GPU), ~4GB (CPU)

## 📦 Building from Source

### Basic Build

```bash
# Build the image
docker build -t aetherist:local .

# Multi-stage build with specific target
docker build --target production -t aetherist:prod .

# Build with build arguments
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg PYTORCH_VERSION=2.0.1 \
  -t aetherist:custom .
```

### Advanced Build Options

```bash
# Build for different architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t aetherist:multi-arch \
  --push .

# Build with cache optimization
docker build \
  --cache-from aetherist:cache \
  --target builder \
  -t aetherist:cache .
```

### Custom Dockerfile

```dockerfile
# Dockerfile.custom
FROM aetherist/aetherist:latest

# Add custom models
COPY ./custom_models /app/models/custom/

# Install additional packages
RUN pip install your-custom-package

# Custom configuration
COPY ./custom_configs /app/configs/

# Override entrypoint
COPY ./custom_entrypoint.sh /app/
RUN chmod +x /app/custom_entrypoint.sh

ENTRYPOINT ["/app/custom_entrypoint.sh"]
```

## 🔧 Configuration

### Environment Variables

```bash
# Core settings
AETHERIST_MODEL_PATH=/app/models
AETHERIST_DATA_PATH=/app/data
AETHERIST_OUTPUT_PATH=/app/outputs
AETHERIST_LOG_LEVEL=INFO

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_MAX_REQUEST_SIZE=100MB

# GPU settings
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Cache settings
REDIS_URL=redis://redis:6379/0
CACHE_DIR=/app/cache
CACHE_SIZE=10GB

# Security settings
API_KEY=your-secure-api-key
JWT_SECRET=your-jwt-secret
CORS_ORIGINS=https://yourdomain.com

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_FORMAT=json
```

### Volume Mounts

```bash
# Essential mounts
-v ./models:/app/models:ro          # Model files (read-only)
-v ./outputs:/app/outputs           # Generated outputs
-v ./logs:/app/logs                 # Application logs
-v ./cache:/app/cache               # Cache directory

# Configuration mounts
-v ./configs:/app/configs:ro        # Custom configurations
-v ./ssl:/app/ssl:ro                # SSL certificates

# Development mounts
-v ./src:/app/src                   # Source code (development)
-v ./tests:/app/tests               # Test files
```

## 🐙 Docker Compose Configurations

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  aetherist:
    build:
      context: .
      target: development
    ports:
      - "8000:8000"
      - "8888:8888"  # Jupyter
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - AETHERIST_ENV=development
      - LOG_LEVEL=DEBUG
      - RELOAD=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  jupyter:
    build:
      context: .
      target: development
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./src:/app/src
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser
    environment:
      - JUPYTER_ENABLE_LAB=yes

volumes:
  redis_data:
```

### Production Environment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - aetherist
    restart: unless-stopped

  aetherist:
    image: aetherist/aetherist:latest
    expose:
      - "8000"
    volumes:
      - models:/app/models:ro
      - outputs:/app/outputs
      - logs:/app/logs
      - ./configs:/app/configs:ro
    environment:
      - AETHERIST_ENV=production
      - API_WORKERS=8
      - LOG_LEVEL=WARNING
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  models:
  outputs:
  logs:
  redis_data:
  prometheus_data:
  grafana_data:
```

### High Availability Setup

```yaml
# docker-compose.ha.yml
version: '3.8'

services:
  haproxy:
    image: haproxy:2.8
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"  # Stats
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    depends_on:
      - aetherist-1
      - aetherist-2
      - aetherist-3

  aetherist-1:
    image: aetherist/aetherist:latest
    hostname: aetherist-1
    environment:
      - NODE_ID=1
    volumes:
      - shared_models:/app/models:ro
      - node1_outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  aetherist-2:
    image: aetherist/aetherist:latest
    hostname: aetherist-2
    environment:
      - NODE_ID=2
    volumes:
      - shared_models:/app/models:ro
      - node2_outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  aetherist-3:
    image: aetherist/aetherist:latest
    hostname: aetherist-3
    environment:
      - NODE_ID=3
    volumes:
      - shared_models:/app/models:ro
      - node3_outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']
              capabilities: [gpu]

  redis-cluster:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes
    ports:
      - "7000-7005:7000-7005"
    volumes:
      - redis_cluster:/data

volumes:
  shared_models:
    driver: local
    driver_opts:
      type: nfs
      o: addr=nfs-server,rw
      device: ":/shared/models"
  node1_outputs:
  node2_outputs:
  node3_outputs:
  redis_cluster:
```

## ⚙️ NGINX Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream aetherist_backend {
        least_conn;
        server aetherist-1:8000 weight=1 max_fails=3 fail_timeout=30s;
        server aetherist-2:8000 weight=1 max_fails=3 fail_timeout=30s;
        server aetherist-3:8000 weight=1 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name aetherist.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name aetherist.yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        client_max_body_size 100M;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;

        location / {
            proxy_pass http://aetherist_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://aetherist_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        location /health {
            proxy_pass http://aetherist_backend/health;
            access_log off;
        }

        location /static {
            alias /app/static;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## 📊 Monitoring with Docker

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aetherist'
    static_configs:
      - targets: ['aetherist:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9445']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
```

### GPU Monitoring

```yaml
# Add to docker-compose.yml
  nvidia-gpu-exporter:
    image: mindprince/nvidia_gpu_prometheus_exporter:0.1
    ports:
      - "9445:9445"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 🔒 Security Considerations

### Container Security

```bash
# Run with non-root user
docker run --user 1000:1000 aetherist:latest

# Read-only root filesystem
docker run --read-only --tmpfs /tmp aetherist:latest

# Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE aetherist:latest

# Security options
docker run --security-opt=no-new-privileges:true aetherist:latest
```

### Network Security

```yaml
# docker-compose.yml
services:
  aetherist:
    networks:
      - backend
  
  nginx:
    networks:
      - frontend
      - backend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
```

### Secrets Management

```yaml
# docker-compose.yml
secrets:
  api_key:
    file: ./secrets/api_key.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  ssl_cert:
    file: ./secrets/ssl_cert.pem
  ssl_key:
    file: ./secrets/ssl_key.pem

services:
  aetherist:
    secrets:
      - api_key
      - jwt_secret
    environment:
      - API_KEY_FILE=/run/secrets/api_key
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
```

## 📈 Scaling and Performance

### Horizontal Scaling

```bash
# Scale up services
docker-compose up -d --scale aetherist=5

# Auto-scaling with Docker Swarm
docker service update --replicas 10 aetherist_aetherist
```

### Resource Limits

```yaml
# docker-compose.yml
services:
  aetherist:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
          reservations:
            devices:
              - driver: nvidia
                device_ids: ['0', '1']
                capabilities: [gpu]
        reservations:
          cpus: '2.0'
          memory: 8G
```

### Health Checks

```yaml
# docker-compose.yml
services:
  aetherist:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## 🚀 Deployment Commands

### Basic Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart aetherist

# View logs
docker-compose logs -f aetherist

# Execute commands
docker-compose exec aetherist python scripts/benchmark.py
```

### Updates and Maintenance

```bash
# Pull latest images
docker-compose pull

# Recreate containers with new image
docker-compose up -d --force-recreate

# Backup data
docker run --rm -v aetherist_models:/data -v $(pwd):/backup alpine tar czf /backup/models_backup.tar.gz /data

# Restore data
docker run --rm -v aetherist_models:/data -v $(pwd):/backup alpine tar xzf /backup/models_backup.tar.gz -C /
```

### Cleanup

```bash
# Remove stopped containers
docker-compose down --remove-orphans

# Clean up images
docker image prune -f

# Remove all data (destructive!)
docker-compose down -v --remove-orphans
```

## 🔧 Troubleshooting Docker Issues

### Container Won't Start

```bash
# Check container logs
docker logs aetherist

# Check resource usage
docker stats

# Inspect container
docker inspect aetherist

# Check GPU access
docker exec aetherist nvidia-smi
```

### Performance Issues

```bash
# Monitor resource usage
docker exec aetherist htop

# Check GPU utilization
docker exec aetherist nvidia-smi -l 1

# Profile application
docker exec aetherist python -m cProfile scripts/benchmark.py
```

### Network Issues

```bash
# Check network connectivity
docker exec aetherist ping google.com

# Check service discovery
docker exec aetherist nslookup redis

# Debug port mapping
docker port aetherist
```

---

This Docker deployment guide provides comprehensive instructions for deploying Aetherist in containerized environments. For Kubernetes deployment, see the [Kubernetes Deployment Guide](kubernetes.md).