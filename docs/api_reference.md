# 📚 API Reference

Complete reference for Aetherist's Python API and REST endpoints.

## 🐍 Python API

### Core Classes

#### AetheristGenerator

The main generator class for 3D avatar generation.

```python
class AetheristGenerator(torch.nn.Module):
    """
    3D-aware avatar generator using triplane neural rendering.
    
    Args:
        config (AetheristConfig): Model configuration
        pretrained_path (str, optional): Path to pretrained weights
    """
    
    def __init__(self, config: AetheristConfig, pretrained_path: Optional[str] = None):
        super().__init__()
        # Implementation details...
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'AetheristGenerator':
        """
        Load generator from pretrained checkpoint.
        
        Args:
            path (str): Path to checkpoint file
            
        Returns:
            AetheristGenerator: Loaded generator
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint.get('config', AetheristConfig())
        generator = cls(config)
        generator.load_state_dict(checkpoint['generator'])
        return generator
    
    def forward(self, 
                latent_codes: torch.Tensor, 
                camera_params: torch.Tensor,
                truncation_psi: float = 1.0,
                noise_mode: str = 'const') -> torch.Tensor:
        """
        Generate avatar images.
        
        Args:
            latent_codes (torch.Tensor): [batch_size, latent_dim]
            camera_params (torch.Tensor): [batch_size, 25] camera parameters
            truncation_psi (float): Truncation strength (0-1)
            noise_mode (str): 'const', 'random', or 'none'
            
        Returns:
            torch.Tensor: Generated images [batch_size, 3, height, width]
        """
    
    def generate_front_view(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Generate front-facing view with default camera."""
    
    def generate_multi_view(self, 
                           latent_codes: torch.Tensor,
                           num_views: int = 8,
                           radius: float = 2.5) -> List[torch.Tensor]:
        """Generate multiple views of avatar."""
    
    def interpolate(self, 
                   latent_a: torch.Tensor,
                   latent_b: torch.Tensor,
                   steps: int = 10) -> List[torch.Tensor]:
        """Interpolate between two latent codes."""
```

#### AetheristConfig

Configuration class for model parameters.

```python
@dataclass
class AetheristConfig:
    """Configuration for Aetherist model."""
    
    # Model architecture
    latent_dim: int = 512
    triplane_dim: int = 256
    triplane_res: int = 64
    neural_renderer_layers: int = 8
    super_resolution_layers: int = 2
    
    # Generation settings
    resolution: int = 512
    batch_size: int = 8
    num_views: int = 1
    
    # Camera settings
    camera_radius: float = 2.5
    elevation_range: Tuple[float, float] = (-0.25, 0.25)
    azimuth_range: Tuple[float, float] = (0, 2 * np.pi)
    
    # Training settings
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0001
    
    # Optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    
    # Loss weights
    adversarial_weight: float = 1.0
    reconstruction_weight: float = 10.0
    perceptual_weight: float = 1.0
    identity_weight: float = 0.5
    
    def save(self, path: str):
        """Save configuration to YAML file."""
    
    @classmethod
    def load(cls, path: str) -> 'AetheristConfig':
        """Load configuration from YAML file."""
```

#### CameraConfig

Camera pose configuration utilities.

```python
class CameraConfig:
    """Camera pose configuration and utilities."""
    
    @staticmethod
    def create_front_view(device: torch.device = None) -> torch.Tensor:
        """Create front-facing camera parameters."""
    
    @staticmethod
    def create_orbit(radius: float = 2.5,
                    elevation: float = 0.0,
                    num_views: int = 8,
                    full_rotation: bool = True,
                    device: torch.device = None) -> torch.Tensor:
        """Create orbital camera trajectory."""
    
    @staticmethod
    def create_random(batch_size: int,
                     radius_range: Tuple[float, float] = (2.0, 3.0),
                     elevation_range: Tuple[float, float] = (-0.3, 0.3),
                     device: torch.device = None) -> torch.Tensor:
        """Create random camera poses."""
    
    @staticmethod
    def interpolate_poses(pose_a: torch.Tensor,
                         pose_b: torch.Tensor,
                         steps: int = 10) -> torch.Tensor:
        """Interpolate between two camera poses."""
```

### Style and Control

#### StyleController

Control avatar attributes and style.

```python
class StyleController:
    """Control avatar generation with semantic attributes."""
    
    def __init__(self, generator: AetheristGenerator):
        self.generator = generator
        self.attribute_vectors = self._load_attribute_vectors()
    
    def encode_attributes(self, attributes: Dict[str, Any]) -> torch.Tensor:
        """
        Encode semantic attributes to latent code.
        
        Args:
            attributes (dict): Attribute specifications
                age: 'young', 'middle', 'elderly'
                gender: 'male', 'female', 'neutral'
                expression: 'neutral', 'smile', 'serious'
                ethnicity: Various options
                hair_color: Color specifications
                
        Returns:
            torch.Tensor: Latent code with encoded attributes
        """
    
    def edit_attribute(self, 
                      latent_code: torch.Tensor,
                      attribute: str,
                      direction: str,
                      strength: float = 1.0) -> torch.Tensor:
        """Edit specific attribute in latent code."""
    
    def mix_attributes(self, 
                      base_latent: torch.Tensor,
                      attribute_edits: List[Dict]) -> torch.Tensor:
        """Apply multiple attribute edits."""
```

#### StyleTransfer

Apply artistic styles to generated avatars.

```python
class StyleTransfer:
    """Apply artistic styles to avatar generation."""
    
    def __init__(self, generator: AetheristGenerator):
        self.generator = generator
        self.style_encoder = StyleEncoder()
    
    def transfer(self,
                latent_code: torch.Tensor,
                style_image: PIL.Image.Image,
                camera_params: torch.Tensor,
                style_weight: float = 1.0,
                preserve_identity: bool = True) -> torch.Tensor:
        """
        Apply style transfer to generated avatar.
        
        Args:
            latent_code: Base latent code
            style_image: Reference style image
            camera_params: Camera parameters
            style_weight: Strength of style transfer
            preserve_identity: Whether to preserve identity features
            
        Returns:
            torch.Tensor: Stylized avatar image
        """
    
    def encode_style(self, image: PIL.Image.Image) -> torch.Tensor:
        """Encode style from reference image."""
    
    def apply_style_vector(self,
                          latent_code: torch.Tensor,
                          style_vector: torch.Tensor,
                          weight: float = 1.0) -> torch.Tensor:
        """Apply pre-encoded style vector."""
```

### Batch Processing

#### BatchProcessor

Efficient batch processing for large-scale generation.

```python
class BatchProcessor:
    """Efficient batch processing for avatar generation."""
    
    def __init__(self,
                 generator: AetheristGenerator,
                 batch_size: int = 8,
                 device: torch.device = None,
                 num_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            generator: Aetherist generator model
            batch_size: Batch size for processing
            device: Device for computation
            num_workers: Number of worker processes
        """
    
    def process_latent_batch(self,
                           latent_codes: torch.Tensor,
                           camera_params: torch.Tensor,
                           **kwargs) -> List[torch.Tensor]:
        """Process batch of latent codes."""
    
    def process_attribute_batch(self,
                              attributes_list: List[Dict],
                              camera_params: torch.Tensor) -> List[torch.Tensor]:
        """Process batch of attribute specifications."""
    
    def process_style_transfer_batch(self,
                                   latent_codes: torch.Tensor,
                                   style_images: List[PIL.Image.Image],
                                   camera_params: torch.Tensor) -> List[torch.Tensor]:
        """Process batch of style transfers."""
    
    def generate_dataset(self,
                        num_samples: int,
                        output_dir: str,
                        attributes_config: Optional[Dict] = None,
                        camera_config: Optional[Dict] = None):
        """Generate large dataset of avatars."""
```

### Utilities

#### Image Utilities

```python
def save_image(tensor: torch.Tensor, 
               path: str,
               normalize: bool = True,
               format: str = 'PNG') -> None:
    """Save tensor as image file."""

def save_image_grid(tensors: List[torch.Tensor],
                   path: str,
                   nrow: int = 8,
                   padding: int = 2) -> None:
    """Save batch of images as grid."""

def tensor_to_pil(tensor: torch.Tensor) -> PIL.Image.Image:
    """Convert tensor to PIL Image."""

def pil_to_tensor(image: PIL.Image.Image,
                 device: torch.device = None) -> torch.Tensor:
    """Convert PIL Image to tensor."""

def resize_tensor(tensor: torch.Tensor,
                 size: Tuple[int, int],
                 mode: str = 'bilinear') -> torch.Tensor:
    """Resize image tensor."""
```

#### Camera Utilities

```python
def generate_camera_poses(num_views: int = 8,
                         radius: float = 2.5,
                         elevation_range: Tuple[float, float] = (-0.25, 0.25),
                         azimuth_range: Tuple[float, float] = (0, 2*np.pi),
                         device: torch.device = None) -> torch.Tensor:
    """Generate camera poses for multi-view generation."""

def create_orbit_camera_path(radius: float = 2.5,
                           elevation: float = 0.0,
                           num_frames: int = 60,
                           device: torch.device = None) -> torch.Tensor:
    """Create smooth orbital camera path for animation."""

def look_at_matrix(eye: torch.Tensor,
                  target: torch.Tensor,
                  up: torch.Tensor) -> torch.Tensor:
    """Create look-at transformation matrix."""

def perspective_projection(fov: float,
                         aspect: float,
                         near: float = 0.1,
                         far: float = 100.0) -> torch.Tensor:
    """Create perspective projection matrix."""
```

#### Latent Space Utilities

```python
def interpolate_latents(latent_a: torch.Tensor,
                       latent_b: torch.Tensor,
                       steps: int = 10,
                       mode: str = 'slerp') -> torch.Tensor:
    """Interpolate between latent codes."""

def sample_latent_codes(num_samples: int,
                       latent_dim: int = 512,
                       truncation_psi: float = 1.0,
                       device: torch.device = None) -> torch.Tensor:
    """Sample latent codes from prior distribution."""

def normalize_latent_codes(latent_codes: torch.Tensor,
                         method: str = 'clip') -> torch.Tensor:
    """Normalize latent codes to valid range."""

def find_latent_directions(attribute_vectors: Dict[str, torch.Tensor],
                         latent_codes: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Find semantic directions in latent space."""
```

## 🌐 REST API

### Base URL
```
http://localhost:8000
```

### Authentication

```http
# API Key authentication
Authorization: Bearer YOUR_API_KEY

# JWT authentication
Authorization: Bearer YOUR_JWT_TOKEN
```

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "gpu_available": true,
    "model_loaded": true
}
```

#### Generate Avatar

```http
POST /generate/avatar
Content-Type: application/json
```

**Request Body:**
```json
{
    "num_views": 4,
    "resolution": 512,
    "seed": 42,
    "truncation_psi": 0.8,
    "camera_config": {
        "radius": 2.5,
        "elevation_range": [-0.2, 0.4],
        "azimuth_range": [0, 6.28]
    },
    "style_config": {
        "age": "young",
        "gender": "female",
        "expression": "smile",
        "hair_color": "brown"
    },
    "output_format": "png"
}
```

**Response:**
```json
{
    "request_id": "req_12345",
    "status": "success",
    "generation_time": 2.34,
    "images": [
        "base64_encoded_image_1",
        "base64_encoded_image_2",
        "base64_encoded_image_3",
        "base64_encoded_image_4"
    ],
    "metadata": {
        "resolution": 512,
        "num_views": 4,
        "model_version": "1.0.0"
    }
}
```

#### Generate with Custom Latent

```http
POST /generate/custom
Content-Type: application/json
```

**Request Body:**
```json
{
    "latent_code": [/* 512 float values */],
    "camera_params": [/* 25 float values per view */],
    "num_views": 1,
    "resolution": 512
}
```

#### Style Transfer

```http
POST /generate/style-transfer
Content-Type: multipart/form-data
```

**Form Data:**
- `style_image`: Image file
- `config`: JSON configuration

**JSON Config:**
```json
{
    "style_weight": 0.7,
    "preserve_identity": true,
    "num_views": 1,
    "resolution": 512,
    "base_attributes": {
        "age": "young",
        "gender": "male"
    }
}
```

#### Batch Generation

```http
POST /generate/batch
Content-Type: application/json
```

**Request Body:**
```json
{
    "requests": [
        {
            "style_config": {"age": "young", "gender": "female"},
            "num_views": 2
        },
        {
            "style_config": {"age": "elderly", "gender": "male"},
            "num_views": 4
        }
    ],
    "resolution": 512,
    "output_format": "png"
}
```

#### Get Generation Status

```http
GET /status/{request_id}
```

**Response:**
```json
{
    "request_id": "req_12345",
    "status": "completed",
    "progress": 100,
    "estimated_remaining": 0,
    "result_url": "/results/req_12345"
}
```

#### Download Results

```http
GET /results/{request_id}
Accept: application/zip
```

Returns ZIP file with generated images.

### Error Responses

```json
{
    "error": "validation_error",
    "message": "Invalid resolution. Must be between 64 and 1024.",
    "details": {
        "field": "resolution",
        "provided": 2048,
        "valid_range": [64, 1024]
    }
}
```

### Rate Limits

- **Free Tier**: 10 requests/minute, 100 requests/day
- **Pro Tier**: 100 requests/minute, 10,000 requests/day
- **Enterprise**: Custom limits

Headers include rate limit information:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

## 🔧 Configuration Schema

### Model Configuration (YAML)

```yaml
# configs/model_config.yaml
model:
  architecture:
    generator:
      latent_dim: 512
      triplane_dim: 256
      triplane_res: 64
      neural_renderer:
        layers: 8
        hidden_dim: 256
        output_activation: "tanh"
      super_resolution:
        enabled: true
        layers: 2
        scale_factor: 2
    
    discriminator:
      image_channels: 3
      base_channels: 64
      max_channels: 512
      num_scales: 3
      use_spectral_norm: true
  
  training:
    optimizer:
      type: "Adam"
      lr: 0.0002
      betas: [0.5, 0.999]
      weight_decay: 0.0001
    
    losses:
      adversarial:
        weight: 1.0
        type: "hinge"
      reconstruction:
        weight: 10.0
        type: "l1"
      perceptual:
        weight: 1.0
        network: "vgg19"
      identity:
        weight: 0.5
        network: "arcface"
    
    regularization:
      gradient_penalty: 10.0
      path_length_penalty: 2.0
      r1_penalty: 10.0
    
    schedule:
      batch_size: 8
      max_epochs: 100
      warmup_steps: 1000
      lr_decay_steps: [50000, 80000]
      lr_decay_gamma: 0.5

generation:
  default_resolution: 512
  default_num_views: 1
  default_truncation_psi: 0.8
  
  camera:
    default_radius: 2.5
    elevation_range: [-0.25, 0.25]
    azimuth_range: [0, 6.283185]
    fov: 0.78539816  # 45 degrees in radians
  
  optimization:
    mixed_precision: true
    compile_model: false
    gradient_checkpointing: false
    memory_efficient: true

paths:
  models_dir: "./models"
  cache_dir: "./cache"
  output_dir: "./outputs"
  logs_dir: "./logs"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "aetherist.log"
  max_size: 100MB
  backup_count: 5
```

### API Configuration (YAML)

```yaml
# configs/api_config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  worker_class: "uvicorn.workers.UvicornWorker"
  max_request_size: 100MB
  timeout: 300
  
  cors:
    allow_origins: ["*"]
    allow_credentials: true
    allow_methods: ["*"]
    allow_headers: ["*"]
  
  rate_limiting:
    enabled: true
    default_limit: "60/minute"
    burst_limit: "100/minute"
    storage: "memory"  # or "redis"
  
  authentication:
    enabled: false
    type: "api_key"  # or "jwt"
    api_keys:
      - "your-api-key-here"
    jwt_secret: "your-jwt-secret"
    jwt_algorithm: "HS256"
    jwt_expiry: 3600  # seconds

generation:
  max_batch_size: 16
  max_resolution: 1024
  max_num_views: 16
  default_timeout: 120
  queue_size: 100
  
  gpu:
    enable_multi_gpu: false
    gpu_ids: [0]
    memory_fraction: 0.8

storage:
  backend: "local"  # or "s3", "gcs"
  local:
    base_path: "./api_storage"
    max_file_size: 50MB
    cleanup_after: 3600  # seconds
  
  s3:
    bucket: "aetherist-storage"
    region: "us-west-2"
    access_key_id: "${AWS_ACCESS_KEY_ID}"
    secret_access_key: "${AWS_SECRET_ACCESS_KEY}"

monitoring:
  enabled: true
  metrics:
    - "request_count"
    - "request_duration"
    - "generation_time"
    - "gpu_memory_usage"
  
  logging:
    level: "INFO"
    format: "json"
    include_request_id: true
    
  health_check:
    endpoint: "/health"
    timeout: 30
    
  prometheus:
    enabled: false
    endpoint: "/metrics"
```

---

This API reference provides comprehensive documentation for all Aetherist functionality. For more examples and tutorials, see the [Getting Started Guide](getting_started.md).