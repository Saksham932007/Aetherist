#!/bin/bash
# Aetherist Docker Entrypoint Script
# Handles initialization and configuration for containerized deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Default values
ENVIRONMENT=${ENVIRONMENT:-production}
LOG_LEVEL=${LOG_LEVEL:-info}
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}
WORKERS=${WORKERS:-4}

log "Starting Aetherist container..."
log "Environment: $ENVIRONMENT"
log "Log Level: $LOG_LEVEL"
log "API Host: $API_HOST"
log "API Port: $API_PORT"

# Check if we're running in GPU mode
if command -v nvidia-smi &> /dev/null; then
    log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    log "Running in CPU mode"
    export CUDA_VISIBLE_DEVICES=""
fi

# Create necessary directories
log "Creating required directories..."
mkdir -p /app/logs
mkdir -p /app/uploads
mkdir -p /app/cache
mkdir -p /app/models

# Set proper permissions
chown -R $(whoami):$(whoami) /app/logs /app/uploads /app/cache 2>/dev/null || true

# Check for model files
log "Checking for model files..."
if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models)" ]; then
    warning "No model files found in /app/models"
    warning "Please mount your model directory or download models"
    warning "The application may not work correctly without models"
else
    success "Model files found"
    ls -la /app/models/ | head -5
fi

# Check configuration files
log "Checking configuration files..."
if [ ! -f "/app/configs/model_config.yaml" ]; then
    warning "Default model configuration not found"
    log "Creating default configuration..."
    
    cat > /app/configs/model_config.yaml << 'EOF'
# Default Aetherist Model Configuration
model:
  name: "aetherist_v1"
  type: "triplane_generator"
  
  # Model architecture
  latent_dim: 512
  triplane_dim: 256
  triplane_res: 256
  resolution: 512
  
  # Paths
  weights_path: "models/aetherist_v1.pth"
  discriminator_path: "models/discriminator.pth"
  
  # Generation settings
  truncation_psi: 0.7
  noise_mode: "const"
  
  # Performance
  fp16: true
  compile_model: false

training:
  batch_size: 8
  learning_rate: 0.002
  beta1: 0.0
  beta2: 0.99
  
inference:
  batch_size: 4
  enable_caching: true
  cache_size: 100
EOF
    
    success "Created default model configuration"
fi

# Check API configuration
if [ ! -f "/app/configs/api_config.yaml" ]; then
    log "Creating default API configuration..."
    
    cat > /app/configs/api_config.yaml << 'EOF'
# Aetherist API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  log_level: "info"
  
  # CORS settings
  cors_origins:
    - "*"
  cors_credentials: true
  cors_methods:
    - "*"
  cors_headers:
    - "*"

# Rate limiting
rate_limiting:
  enabled: true
  requests_per_minute: 60
  burst_size: 10

# Caching
cache:
  enabled: true
  backend: "redis"
  ttl: 3600  # 1 hour
  max_size: 1000

# File upload limits
uploads:
  max_file_size: 50  # MB
  allowed_extensions:
    - ".jpg"
    - ".jpeg" 
    - ".png"
    - ".webp"
  upload_path: "/app/uploads"

# Security
security:
  require_api_key: false
  api_key: "${API_KEY:-}"
  enable_https: false
  
# Monitoring
monitoring:
  enable_metrics: true
  metrics_path: "/metrics"
  health_check_path: "/health"
EOF
    
    success "Created default API configuration"
fi

# Database migration (if applicable)
if [ -n "$DATABASE_URL" ]; then
    log "Database URL configured, checking connection..."
    
    # Wait for database to be ready
    timeout=30
    while [ $timeout -gt 0 ]; do
        if python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
            success "Database connection established"
            break
        else
            timeout=$((timeout-1))
            if [ $timeout -eq 0 ]; then
                error "Failed to connect to database after 30 seconds"
                exit 1
            fi
            log "Waiting for database... ($timeout seconds remaining)"
            sleep 1
        fi
    done
    
    # Run migrations if they exist
    if [ -f "/app/migrations/migrate.py" ]; then
        log "Running database migrations..."
        python /app/migrations/migrate.py || warning "Migration failed, continuing..."
    fi
fi

# Redis connection check (if applicable)
if [ -n "$REDIS_URL" ]; then
    log "Redis URL configured, checking connection..."
    
    timeout=15
    while [ $timeout -gt 0 ]; do
        if python -c "
import redis
import os
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis connection successful')
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
            success "Redis connection established"
            break
        else
            timeout=$((timeout-1))
            if [ $timeout -eq 0 ]; then
                warning "Failed to connect to Redis, caching will be disabled"
                break
            fi
            sleep 1
        fi
    done
fi

# Model verification
log "Verifying model installation..."
if python -c "
import sys
sys.path.insert(0, '/app')
try:
    from aetherist.models import load_model
    print('Model loading functions available')
    exit(0)
except Exception as e:
    print(f'Model verification failed: {e}')
    exit(1)
" 2>/dev/null; then
    success "Model verification passed"
else
    warning "Model verification failed, some features may not work"
fi

# Start application based on command
log "Starting application with command: $@"

# Handle different startup modes
if [ "$1" = "api" ] || [ "$1" = "uvicorn" ] || [[ "$1" =~ "uvicorn" ]]; then
    # API server mode
    log "Starting API server..."
    
    # Determine number of workers based on environment
    if [ "$ENVIRONMENT" = "development" ]; then
        WORKERS=1
        RELOAD_FLAG="--reload"
    else
        WORKERS=${WORKERS:-$(nproc)}
        RELOAD_FLAG=""
    fi
    
    exec python -m uvicorn aetherist.api.main:app \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        $RELOAD_FLAG
        
elif [ "$1" = "web" ] || [ "$1" = "gradio" ]; then
    # Web interface mode
    log "Starting Gradio web interface..."
    exec python -m aetherist.web.gradio_app
    
elif [ "$1" = "worker" ]; then
    # Background worker mode
    log "Starting background worker..."
    exec python -m aetherist.worker.main
    
elif [ "$1" = "train" ]; then
    # Training mode
    log "Starting training..."
    shift  # Remove 'train' from arguments
    exec python -m aetherist.training.train "$@"
    
elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    # Interactive shell mode
    log "Starting interactive shell..."
    exec "$@"
    
else
    # Default: run the provided command
    log "Running custom command: $@"
    exec "$@"
fi