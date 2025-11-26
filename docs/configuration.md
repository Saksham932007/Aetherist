# ⚙️ Configuration Guide

This guide covers all configuration options for Aetherist, from basic settings to advanced production deployments.

## 📁 Configuration Files Overview

Aetherist uses hierarchical YAML configuration files:

```
configs/
├── model_config.yaml          # Model architecture settings
├── train_config.yaml          # Training parameters
├── api_config.yaml            # API server configuration  
├── deploy_config.yaml         # Deployment settings
├── examples/                  # Example configurations
│   ├── quick_start.yaml       # Minimal working config
│   ├── high_quality.yaml      # High quality generation
│   ├── fast_inference.yaml    # Speed-optimized config
│   └── production.yaml        # Production deployment
└── environments/              # Environment-specific configs
    ├── development.yaml       # Development overrides
    ├── staging.yaml           # Staging environment
    └── production.yaml        # Production environment
```

## 🏗️ Model Configuration

### Basic Model Settings

```yaml
# configs/model_config.yaml
model:
  # Core architecture parameters
  architecture:
    name: "aetherist_v1"
    type: "triplane_generator"
    
    generator:
      # Latent space configuration
      latent_dim: 512                    # Latent code dimensionality
      latent_std: 1.0                    # Standard deviation for sampling
      
      # Triplane configuration
      triplane_dim: 256                  # Feature channels per plane
      triplane_res: 64                   # Spatial resolution of triplanes
      triplane_layers: 8                 # Synthesis network depth
      
      # Neural renderer configuration
      neural_renderer:
        type: "mlp_renderer"             # Renderer type
        hidden_dim: 256                  # Hidden layer size
        num_layers: 8                    # Number of MLP layers
        skip_connections: [4]            # Skip connection positions
        activation: "relu"               # Activation function
        output_activation: "tanh"        # Final output activation
        
      # Super-resolution configuration
      super_resolution:
        enabled: true                    # Enable SR module
        scale_factor: 2                  # Upsampling factor
        num_layers: 2                    # Number of SR layers
        channels: 128                    # SR feature channels
        
    discriminator:
      type: "multi_scale"                # Discriminator type
      image_channels: 3                  # Input image channels
      base_channels: 64                  # Base feature channels
      max_channels: 512                  # Maximum feature channels
      num_scales: 4                      # Number of scales
      num_blocks: 4                      # Blocks per scale
      use_spectral_norm: true            # Spectral normalization
      
  # Generation settings
  generation:
    default_resolution: 512              # Default output resolution
    max_resolution: 1024                # Maximum allowed resolution
    min_resolution: 64                  # Minimum allowed resolution
    default_num_views: 1                # Default number of views
    max_num_views: 16                   # Maximum views per request
    default_truncation_psi: 0.8         # Default truncation strength
    
  # Camera configuration
  camera:
    default_radius: 2.5                 # Default camera distance
    radius_range: [1.5, 3.5]           # Valid radius range
    elevation_range: [-0.3, 0.3]       # Elevation angle range (radians)
    azimuth_range: [0, 6.283185]       # Azimuth angle range (0 to 2π)
    fov: 0.78539816                     # Field of view (45 degrees)
    near_plane: 0.1                     # Near clipping plane
    far_plane: 100.0                    # Far clipping plane
    
  # Style control
  style_control:
    enabled: true                       # Enable style control
    attribute_vectors_path: "vectors/"  # Path to attribute vectors
    available_attributes:               # List of controllable attributes
      - age
      - gender
      - expression
      - ethnicity
      - hair_color
      - hair_style
      - facial_hair
      - accessories
      
  # Model paths
  paths:
    pretrained_generator: "models/generator.pth"
    pretrained_discriminator: "models/discriminator.pth"
    attribute_vectors: "models/attribute_vectors/"
    cache_dir: "cache/"
    
  # Device configuration
  device:
    type: "auto"                        # "auto", "cpu", "cuda", "mps"
    gpu_ids: [0]                        # List of GPU IDs to use
    mixed_precision: true               # Enable automatic mixed precision
    compile_model: false                # Enable torch.compile (PyTorch 2.0+)
```

### Advanced Model Settings

```yaml
# Advanced architecture options
model:
  architecture:
    generator:
      # Advanced triplane options
      triplane_config:
        initialization: "xavier_normal"   # Weight initialization
        use_noise_injection: true        # StyleGAN-style noise
        noise_strength: 0.1              # Noise injection strength
        use_progressive_growing: false   # Progressive growing training
        
      # Advanced neural renderer options
      neural_renderer:
        positional_encoding:
          enabled: true                  # Use positional encoding
          num_frequencies: 10            # Number of frequency bands
          include_input: true            # Include raw coordinates
        
        view_direction_encoding:
          enabled: true                  # Encode view direction
          num_frequencies: 4             # Frequency bands for direction
          
        density_activation: "softplus"   # Density activation function
        color_activation: "sigmoid"      # Color activation function
        
        hierarchical_sampling:
          enabled: true                  # Use hierarchical sampling
          num_coarse: 64                # Coarse samples per ray
          num_fine: 128                 # Fine samples per ray
          
      # Style injection configuration
      style_injection:
        enabled: true                    # Enable style modulation
        injection_layers: [0, 2, 4, 6]  # Which layers to inject style
        style_dim: 512                  # Style vector dimensionality
        
    discriminator:
      # Progressive discriminator options
      progressive_config:
        fade_in_epochs: 10               # Epochs to fade in new scale
        stable_epochs: 20                # Epochs at each scale
        
      # Conditioning options
      conditioning:
        use_camera_conditioning: true    # Condition on camera pose
        camera_embedding_dim: 128        # Camera embedding size
        use_latent_conditioning: false   # Condition on latent codes
        
  # Advanced generation options
  generation:
    sampling:
      method: "ddim"                     # Sampling method
      num_steps: 50                     # Number of sampling steps
      eta: 0.0                          # DDIM eta parameter
      
    guidance:
      classifier_free_guidance: false   # Enable CFG
      guidance_scale: 7.5               # CFG scale
      
    post_processing:
      enabled: true                     # Enable post-processing
      upsampling_method: "esrgan"       # Upsampling algorithm
      face_enhancement: true            # Enhance facial features
      background_removal: false         # Remove background
```

## 🏋️ Training Configuration

### Basic Training Settings

```yaml
# configs/train_config.yaml
training:
  # Basic parameters
  experiment_name: "aetherist_v1"       # Experiment identifier
  resume_from_checkpoint: null          # Path to resume from
  seed: 42                             # Random seed for reproducibility
  
  # Training schedule
  schedule:
    max_epochs: 100                     # Maximum training epochs
    max_steps: null                     # Maximum steps (overrides epochs)
    warmup_steps: 1000                  # Learning rate warmup steps
    
    # Learning rate schedule
    lr_schedule:
      type: "cosine_annealing"          # LR schedule type
      base_lr: 0.0002                   # Base learning rate
      min_lr: 1e-6                      # Minimum learning rate
      warmup_lr: 1e-5                   # Warmup learning rate
      decay_steps: [50000, 80000]      # Decay milestones
      decay_factor: 0.5                 # Decay factor
      
  # Batch configuration
  batch:
    size: 8                             # Batch size per GPU
    accumulation_steps: 1               # Gradient accumulation
    max_batch_size: 32                  # Maximum effective batch size
    
  # Data loading
  data:
    num_workers: 4                      # DataLoader workers
    pin_memory: true                    # Pin memory for GPU transfer
    persistent_workers: true            # Keep workers alive
    prefetch_factor: 2                  # Prefetch batches
    
  # Optimization
  optimizer:
    generator:
      type: "Adam"                      # Optimizer type
      lr: 0.0002                        # Learning rate
      betas: [0.5, 0.999]              # Adam parameters
      weight_decay: 0.0001              # L2 regularization
      eps: 1e-8                         # Numerical stability
      
    discriminator:
      type: "Adam"
      lr: 0.0002
      betas: [0.5, 0.999]
      weight_decay: 0.0001
      eps: 1e-8
      
  # Loss configuration
  losses:
    adversarial:
      type: "hinge"                     # "hinge", "wgan", "non_saturating"
      weight: 1.0                       # Loss weight
      
    reconstruction:
      l1_weight: 10.0                   # L1 reconstruction weight
      l2_weight: 0.0                    # L2 reconstruction weight
      
    perceptual:
      enabled: true                     # Enable perceptual loss
      network: "vgg19"                  # Feature network
      layers: ["relu_1_2", "relu_2_2", "relu_3_4", "relu_4_4", "relu_5_4"]
      weight: 1.0                       # Perceptual loss weight
      
    identity:
      enabled: true                     # Enable identity preservation
      network: "arcface"                # Identity network
      weight: 0.5                       # Identity loss weight
      
    lpips:
      enabled: true                     # Enable LPIPS loss
      network: "alex"                   # LPIPS network type
      weight: 0.1                       # LPIPS weight
      
  # Regularization
  regularization:
    gradient_penalty:
      enabled: true                     # Enable gradient penalty
      weight: 10.0                      # GP weight
      lambda_gp: 10.0                   # GP lambda parameter
      
    r1_penalty:
      enabled: true                     # Enable R1 regularization
      weight: 10.0                      # R1 weight
      interval: 16                      # Apply every N steps
      
    path_length_penalty:
      enabled: true                     # Enable path length regularization
      weight: 2.0                       # PLR weight
      decay: 0.01                      # PLR decay factor
      
  # Model updates
  updates:
    generator_steps: 1                  # Generator updates per iteration
    discriminator_steps: 1              # Discriminator updates per iteration
    ema_beta: 0.999                     # EMA momentum for generator
    ema_start_step: 10000              # When to start EMA updates
    
  # Validation
  validation:
    interval: 1000                      # Validation interval (steps)
    num_samples: 16                     # Number of validation samples
    save_images: true                   # Save validation images
    compute_metrics: true               # Compute validation metrics
    
    metrics:
      - "fid"                          # Fréchet Inception Distance
      - "lpips"                        # Learned Perceptual Image Patch Similarity
      - "psnr"                         # Peak Signal-to-Noise Ratio
      - "ssim"                         # Structural Similarity Index
      
  # Checkpointing
  checkpointing:
    interval: 1000                      # Checkpoint interval (steps)
    max_checkpoints: 5                  # Maximum checkpoints to keep
    save_best: true                     # Save best model based on metrics
    save_latest: true                   # Always save latest model
    
  # Logging
  logging:
    interval: 50                        # Log interval (steps)
    log_images: true                    # Log training images
    log_gradients: false                # Log gradient statistics
    log_weights: false                  # Log weight statistics
    
    wandb:
      enabled: false                    # Enable Weights & Biases
      project: "aetherist"              # WandB project name
      entity: "your-entity"             # WandB entity
      
    tensorboard:
      enabled: true                     # Enable TensorBoard
      log_dir: "logs/"                  # TensorBoard log directory
```

### Advanced Training Settings

```yaml
# Advanced training configurations
training:
  # Progressive training
  progressive:
    enabled: false                      # Enable progressive training
    initial_resolution: 64              # Starting resolution
    final_resolution: 512               # Target resolution
    growth_schedule: "linear"           # "linear", "exponential"
    epochs_per_stage: 20               # Epochs per resolution stage
    
  # Mixed precision training
  mixed_precision:
    enabled: true                       # Enable automatic mixed precision
    loss_scale: "dynamic"              # "dynamic" or fixed value
    growth_factor: 2.0                 # Loss scale growth factor
    backoff_factor: 0.5                # Loss scale backoff factor
    growth_interval: 2000              # Steps between scale updates
    
  # Gradient management
  gradients:
    clipping:
      enabled: true                     # Enable gradient clipping
      max_norm: 1.0                    # Maximum gradient norm
      norm_type: 2                     # Norm type for clipping
      
    accumulation:
      enabled: false                    # Enable gradient accumulation
      steps: 4                         # Accumulation steps
      sync_bn: true                    # Sync batch norm across accumulation
      
  # Data augmentation
  augmentation:
    enabled: true                       # Enable data augmentation
    
    geometric:
      rotation: 0.1                    # Random rotation range (radians)
      translation: 0.05                # Random translation fraction
      scaling: 0.1                     # Random scaling factor
      
    photometric:
      brightness: 0.1                  # Brightness adjustment range
      contrast: 0.1                    # Contrast adjustment range
      saturation: 0.1                  # Saturation adjustment range
      hue: 0.05                        # Hue adjustment range
      
    advanced:
      cutout:
        enabled: false                 # Enable random cutout
        ratio: 0.25                   # Cutout area ratio
        
      mixup:
        enabled: false                 # Enable mixup augmentation
        alpha: 0.2                    # Mixup alpha parameter
        
  # Curriculum learning
  curriculum:
    enabled: false                      # Enable curriculum learning
    
    camera_curriculum:
      start_radius: [2.0, 3.0]         # Initial camera radius range
      end_radius: [1.5, 3.5]          # Final camera radius range
      start_elevation: [-0.1, 0.1]     # Initial elevation range
      end_elevation: [-0.3, 0.3]      # Final elevation range
      curriculum_steps: 50000          # Steps to complete curriculum
```

## 🌐 API Configuration

### Basic API Settings

```yaml
# configs/api_config.yaml
api:
  # Server configuration
  server:
    host: "0.0.0.0"                    # Bind host
    port: 8000                         # Bind port
    workers: 4                         # Number of worker processes
    worker_class: "uvicorn.workers.UvicornWorker"  # Worker class
    worker_connections: 1000           # Connections per worker
    max_requests: 1000                 # Requests before worker restart
    max_requests_jitter: 50            # Random jitter for max_requests
    timeout: 300                       # Request timeout (seconds)
    keepalive: 5                       # Keep-alive timeout
    
  # Request handling
  requests:
    max_request_size: 100MB            # Maximum request size
    max_batch_size: 16                 # Maximum batch processing size
    default_timeout: 120               # Default generation timeout
    queue_size: 100                    # Request queue size
    
    # File upload limits
    upload:
      max_file_size: 50MB              # Maximum uploaded file size
      allowed_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
      max_files_per_request: 10        # Maximum files per request
      
  # CORS configuration
  cors:
    allow_origins: ["*"]               # Allowed origins
    allow_credentials: true            # Allow credentials
    allow_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: ["*"]               # Allowed headers
    expose_headers: ["*"]              # Exposed headers
    max_age: 600                      # Preflight cache time
    
  # Rate limiting
  rate_limiting:
    enabled: true                      # Enable rate limiting
    storage: "memory"                  # "memory", "redis"
    
    # Rate limit tiers
    tiers:
      default:
        requests_per_minute: 60        # Requests per minute
        requests_per_hour: 1000        # Requests per hour
        requests_per_day: 10000        # Requests per day
        burst_limit: 100               # Burst request limit
        
      premium:
        requests_per_minute: 600
        requests_per_hour: 10000
        requests_per_day: 100000
        burst_limit: 1000
        
    # Rate limit by endpoint
    endpoints:
      "/generate/avatar":
        requests_per_minute: 30        # Lower limit for expensive endpoint
      "/generate/batch":
        requests_per_minute: 10        # Very low limit for batch processing
        
  # Authentication
  authentication:
    enabled: false                     # Enable authentication
    type: "api_key"                   # "api_key", "jwt", "oauth"
    
    api_key:
      header_name: "X-API-Key"         # API key header name
      query_param: "api_key"           # Query parameter name
      keys:                           # Valid API keys
        - "your-api-key-here"
        
    jwt:
      secret_key: "your-jwt-secret"    # JWT secret key
      algorithm: "HS256"               # JWT algorithm
      expiry: 3600                     # Token expiry (seconds)
      refresh_expiry: 86400            # Refresh token expiry
      
  # Response configuration
  responses:
    default_format: "json"             # Default response format
    include_request_id: true           # Include request ID in responses
    include_timing: true               # Include timing information
    
    compression:
      enabled: true                    # Enable response compression
      threshold: 1024                  # Compression threshold (bytes)
      level: 6                        # Compression level (1-9)
      
    caching:
      enabled: false                   # Enable response caching
      max_age: 3600                   # Cache max age (seconds)
      
  # WebSocket configuration
  websocket:
    enabled: true                      # Enable WebSocket support
    max_connections: 100               # Maximum concurrent connections
    heartbeat_interval: 30             # Heartbeat interval (seconds)
    
  # Static files
  static:
    enabled: true                      # Serve static files
    directory: "static/"               # Static files directory
    max_age: 86400                    # Cache max age for static files
```

### Production API Settings

```yaml
# Production-specific API configuration
api:
  # Production server settings
  server:
    workers: 8                         # More workers for production
    worker_class: "gunicorn.workers.sync.SyncWorker"
    preload_app: true                  # Preload application
    max_worker_memory: 2048            # Max worker memory (MB)
    worker_tmp_dir: "/dev/shm"         # Use shared memory for temp files
    
  # Security settings
  security:
    enable_https: true                 # Force HTTPS
    ssl_cert_path: "/path/to/cert.pem"
    ssl_key_path: "/path/to/key.pem"
    
    headers:
      # Security headers
      strict_transport_security: "max-age=31536000; includeSubDomains"
      content_type_options: "nosniff"
      frame_options: "DENY"
      xss_protection: "1; mode=block"
      
  # Monitoring and health checks
  monitoring:
    enabled: true                      # Enable monitoring
    
    health_check:
      endpoint: "/health"              # Health check endpoint
      timeout: 30                     # Health check timeout
      
    metrics:
      endpoint: "/metrics"             # Prometheus metrics endpoint
      include_default_metrics: true    # Include default system metrics
      
    logging:
      level: "INFO"                   # Log level
      format: "json"                  # Log format
      include_request_id: true        # Include request ID
      
      # Log sampling (reduce log volume)
      sampling:
        enabled: true
        rate: 0.1                     # Log 10% of requests
        
  # Resource limits
  limits:
    max_concurrent_generations: 8      # Max concurrent generations
    max_gpu_memory_per_request: 4096   # Max GPU memory (MB)
    max_cpu_time_per_request: 300     # Max CPU time (seconds)
    
  # Caching configuration
  caching:
    enabled: true                      # Enable caching
    backend: "redis"                   # Cache backend
    
    redis:
      host: "redis"                   # Redis host
      port: 6379                      # Redis port
      db: 0                          # Redis database
      password: null                  # Redis password
      
    # Cache settings
    default_ttl: 3600                 # Default TTL (seconds)
    max_memory: "1GB"                 # Maximum cache memory
    
    # Cache keys configuration
    keys:
      generation: "gen:{hash}"         # Generation cache key pattern
      model: "model:{version}"         # Model cache key pattern
```

## 🚀 Deployment Configuration

### Docker Deployment

```yaml
# configs/deploy_config.yaml
deployment:
  type: "docker"                       # Deployment type
  
  docker:
    # Image configuration
    image:
      name: "aetherist"                # Image name
      tag: "latest"                    # Image tag
      registry: "docker.io"           # Container registry
      
    # Container configuration
    container:
      name: "aetherist"                # Container name
      restart_policy: "unless-stopped" # Restart policy
      
      # Resource limits
      resources:
        memory: "16GB"                 # Memory limit
        cpus: "8"                      # CPU limit
        
        gpu:
          enabled: true                # Enable GPU access
          device_ids: [0, 1]          # GPU device IDs
          
      # Environment variables
      environment:
        AETHERIST_ENV: "production"
        LOG_LEVEL: "INFO"
        
      # Volume mounts
      volumes:
        - "./models:/app/models:ro"    # Mount models (read-only)
        - "./outputs:/app/outputs"     # Mount outputs
        - "./logs:/app/logs"          # Mount logs
        - "./cache:/app/cache"        # Mount cache
        
      # Port mapping
      ports:
        - "8000:8000"                 # API port
        - "9090:9090"                 # Metrics port
        
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
      
  # Load balancer configuration
  load_balancer:
    enabled: true                      # Enable load balancing
    type: "nginx"                     # Load balancer type
    
    nginx:
      config_path: "nginx.conf"       # NGINX config file
      upstream_servers:               # Backend servers
        - "aetherist-1:8000"
        - "aetherist-2:8000"
        - "aetherist-3:8000"
        
      # SSL configuration
      ssl:
        enabled: true
        cert_path: "/etc/ssl/certs/cert.pem"
        key_path: "/etc/ssl/private/key.pem"
```

### Kubernetes Deployment

```yaml
# Kubernetes deployment configuration
deployment:
  type: "kubernetes"
  
  kubernetes:
    # Namespace configuration
    namespace: "aetherist"
    
    # Deployment specification
    deployment:
      name: "aetherist"
      replicas: 3                     # Number of replicas
      
      # Update strategy
      strategy:
        type: "RollingUpdate"
        max_unavailable: 1
        max_surge: 1
        
      # Pod specification
      pod:
        # Container specification
        container:
          image: "aetherist:latest"
          
          # Resource requests and limits
          resources:
            requests:
              memory: "8Gi"
              cpu: "4"
              nvidia.com/gpu: 1
            limits:
              memory: "16Gi"
              cpu: "8"
              nvidia.com/gpu: 1
              
          # Environment variables
          env:
            - name: "AETHERIST_ENV"
              value: "production"
            - name: "LOG_LEVEL"
              value: "INFO"
              
          # Volume mounts
          volume_mounts:
            - name: "models"
              mount_path: "/app/models"
              read_only: true
            - name: "cache"
              mount_path: "/app/cache"
              
        # Volumes
        volumes:
          - name: "models"
            persistent_volume_claim:
              claim_name: "aetherist-models"
          - name: "cache"
            empty_dir: {}
            
    # Service configuration
    service:
      name: "aetherist-service"
      type: "ClusterIP"               # Service type
      ports:
        - name: "api"
          port: 8000
          target_port: 8000
          
    # Horizontal Pod Autoscaler
    hpa:
      enabled: true                   # Enable auto-scaling
      min_replicas: 3                # Minimum replicas
      max_replicas: 10               # Maximum replicas
      
      # Scaling metrics
      metrics:
        - type: "Resource"
          resource:
            name: "cpu"
            target_average_utilization: 70
        - type: "Resource"
          resource:
            name: "memory"
            target_average_utilization: 80
            
    # Ingress configuration
    ingress:
      enabled: true
      class: "nginx"
      
      # TLS configuration
      tls:
        enabled: true
        secret_name: "aetherist-tls"
        
      # Rules
      rules:
        - host: "api.aetherist.com"
          paths:
            - path: "/"
              service: "aetherist-service"
              port: 8000
```

## 🔧 Environment-Specific Configurations

### Development Environment

```yaml
# configs/environments/development.yaml
# Override base configuration for development

# Use smaller models for faster iteration
model:
  architecture:
    generator:
      triplane_res: 32               # Smaller triplane for speed
      neural_renderer:
        num_layers: 4                # Fewer layers for speed
        
  generation:
    default_resolution: 256          # Lower resolution for speed

# Faster training settings
training:
  batch:
    size: 2                         # Smaller batches for debugging
    
  validation:
    interval: 100                   # More frequent validation
    
  logging:
    interval: 10                    # More frequent logging
    log_gradients: true             # Enable gradient logging for debugging

# Development API settings
api:
  server:
    workers: 1                      # Single worker for debugging
    reload: true                    # Auto-reload on code changes
    
  rate_limiting:
    enabled: false                  # Disable rate limiting for testing
    
  authentication:
    enabled: false                  # Disable auth for development
```

### Production Environment

```yaml
# configs/environments/production.yaml
# Production overrides

# Optimized model settings
model:
  device:
    mixed_precision: true           # Enable mixed precision
    compile_model: true             # Enable model compilation
    
# Production training settings (if training in production)
training:
  batch:
    size: 16                        # Larger batches for efficiency
    
  mixed_precision:
    enabled: true                   # Enable mixed precision training
    
  logging:
    interval: 500                   # Less frequent logging
    log_gradients: false            # Disable gradient logging
    
    wandb:
      enabled: true                 # Enable WandB for production monitoring

# Production API settings
api:
  server:
    workers: 8                      # More workers for production
    
  rate_limiting:
    enabled: true                   # Enable rate limiting
    
  authentication:
    enabled: true                   # Enable authentication
    
  monitoring:
    enabled: true                   # Enable comprehensive monitoring
    
  security:
    enable_https: true              # Force HTTPS
```

## 📊 Configuration Validation

### Configuration Schema

Aetherist validates all configuration files against predefined schemas:

```python
# Example configuration validation
from aetherist.config import validate_config, load_config

# Load and validate configuration
try:
    config = load_config("configs/model_config.yaml")
    validate_config(config)
    print("✅ Configuration is valid")
except ConfigurationError as e:
    print(f"❌ Configuration error: {e}")
```

### Common Configuration Errors

1. **Invalid Parameter Ranges**
   ```yaml
   # ❌ Invalid - batch size too large
   training:
     batch:
       size: 1000  # Will cause OOM errors
   
   # ✅ Valid - reasonable batch size
   training:
     batch:
       size: 8
   ```

2. **Incompatible Settings**
   ```yaml
   # ❌ Invalid - conflicting settings
   model:
     device:
       mixed_precision: true
       compile_model: true      # May not work together
   
   # ✅ Valid - compatible settings
   model:
     device:
       mixed_precision: true
       compile_model: false
   ```

3. **Missing Required Paths**
   ```yaml
   # ❌ Invalid - missing model path
   model:
     paths:
       pretrained_generator: ""  # Empty path
   
   # ✅ Valid - proper model path
   model:
     paths:
       pretrained_generator: "models/generator.pth"
   ```

---

This configuration guide provides comprehensive documentation for all Aetherist settings. Start with the basic configurations and gradually customize advanced options based on your specific needs.