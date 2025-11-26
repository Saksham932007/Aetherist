#!/usr/bin/env python3
"""Generate comprehensive API documentation for Aetherist.

Generates OpenAPI/Swagger documentation, code examples, and API guides.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_openapi_spec() -> Dict[str, Any]:
    """Generate OpenAPI specification for Aetherist API."""
    
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Aetherist API",
            "description": "Advanced 3D-aware image generation using GANs with attention mechanisms",
            "version": "1.0.0",
            "contact": {
                "name": "Aetherist Team",
                "url": "https://github.com/Saksham932007/Aetherist"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.aetherist.ai",
                "description": "Production server"
            }
        ],
        "paths": {
            "/generate": {
                "post": {
                    "summary": "Generate images",
                    "description": "Generate high-quality artistic images using the Aetherist model",
                    "tags": ["Generation"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/GenerateRequest"
                                },
                                "examples": {
                                    "basic": {
                                        "summary": "Basic generation",
                                        "value": {
                                            "num_samples": 4,
                                            "seed": 42,
                                            "resolution": 512,
                                            "style": "artistic"
                                        }
                                    },
                                    "advanced": {
                                        "summary": "Advanced generation with camera control",
                                        "value": {
                                            "num_samples": 1,
                                            "seed": 123,
                                            "resolution": 1024,
                                            "camera_angles": [[0, 0], [30, 45]],
                                            "style": "photorealistic",
                                            "guidance_scale": 7.5
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successfully generated images",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/GenerateResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid request parameters",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Check API health and model status",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "System healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models": {
                "get": {
                    "summary": "List available models",
                    "description": "Get list of available models and their configurations",
                    "tags": ["Models"],
                    "responses": {
                        "200": {
                            "description": "List of models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ModelsResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models/{model_id}/analyze": {
                "post": {
                    "summary": "Analyze model",
                    "description": "Perform comprehensive model analysis",
                    "tags": ["Models", "Analysis"],
                    "parameters": [
                        {
                            "name": "model_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "Model identifier"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/AnalyzeRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Analysis results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AnalyzeResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/batch/jobs": {
                "post": {
                    "summary": "Submit batch job",
                    "description": "Submit a batch processing job",
                    "tags": ["Batch"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/BatchJobRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": {
                            "description": "Job accepted",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/BatchJobResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/batch/jobs/{job_id}": {
                "get": {
                    "summary": "Get batch job status",
                    "description": "Get the status of a batch processing job",
                    "tags": ["Batch"],
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "Job identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Job status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/BatchJobStatusResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/monitoring/metrics": {
                "get": {
                    "summary": "Get system metrics",
                    "description": "Get current system monitoring metrics",
                    "tags": ["Monitoring"],
                    "responses": {
                        "200": {
                            "description": "System metrics",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MetricsResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "GenerateRequest": {
                    "type": "object",
                    "required": ["num_samples"],
                    "properties": {
                        "num_samples": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 16,
                            "description": "Number of images to generate"
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducible generation"
                        },
                        "resolution": {
                            "type": "integer",
                            "enum": [256, 512, 1024],
                            "default": 512,
                            "description": "Output image resolution"
                        },
                        "camera_angles": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "Camera angles as [pitch, yaw] pairs"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["artistic", "photorealistic", "abstract"],
                            "default": "artistic",
                            "description": "Generation style"
                        },
                        "guidance_scale": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 20.0,
                            "default": 7.5,
                            "description": "Guidance scale for generation"
                        }
                    }
                },
                "GenerateResponse": {
                    "type": "object",
                    "required": ["images", "metadata"],
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "format": "base64",
                                "description": "Base64 encoded image data"
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "generation_time": {
                                    "type": "number",
                                    "description": "Generation time in seconds"
                                },
                                "model_version": {
                                    "type": "string",
                                    "description": "Model version used"
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Generation parameters used"
                                }
                            }
                        }
                    }
                },
                "HealthResponse": {
                    "type": "object",
                    "required": ["status", "timestamp"],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "unhealthy"]
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "model_loaded": {
                            "type": "boolean"
                        },
                        "gpu_available": {
                            "type": "boolean"
                        },
                        "memory_usage": {
                            "type": "object",
                            "properties": {
                                "used_gb": {"type": "number"},
                                "total_gb": {"type": "number"},
                                "percentage": {"type": "number"}
                            }
                        }
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "required": ["error", "message"],
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error type"
                        },
                        "message": {
                            "type": "string",
                            "description": "Human-readable error message"
                        },
                        "details": {
                            "type": "object",
                            "description": "Additional error details"
                        }
                    }
                }
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        },
        "tags": [
            {
                "name": "Generation",
                "description": "Image generation endpoints"
            },
            {
                "name": "Models",
                "description": "Model management and analysis"
            },
            {
                "name": "Batch",
                "description": "Batch processing operations"
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and metrics"
            },
            {
                "name": "System",
                "description": "System administration"
            }
        ]
    }

def generate_api_docs_markdown() -> str:
    """Generate comprehensive API documentation in Markdown format."""
    
    return """
# Aetherist API Documentation

## Overview

The Aetherist API provides access to advanced 3D-aware image generation capabilities using state-of-the-art GANs with attention mechanisms. This API allows you to generate high-quality artistic images with fine-grained control over style, camera angles, and generation parameters.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.aetherist.ai`

## Authentication

The API supports two authentication methods:

1. **API Key** (Recommended for server-to-server communication)
   ```
   X-API-Key: your_api_key_here
   ```

2. **Bearer Token** (For user-facing applications)
   ```
   Authorization: Bearer your_token_here
   ```

## Rate Limiting

- **Free Tier**: 100 requests per hour, 10 images per request max
- **Pro Tier**: 1000 requests per hour, 16 images per request max
- **Enterprise**: Custom limits based on agreement

## Quick Start

### 1. Basic Image Generation

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "num_samples": 4,
    "seed": 42,
    "resolution": 512,
    "style": "artistic"
  }'
```

### 2. Advanced Generation with Camera Control

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "num_samples": 2,
    "seed": 123,
    "resolution": 1024,
    "camera_angles": [[0, 0], [30, 45]],
    "style": "photorealistic",
    "guidance_scale": 7.5
  }'
```

### 3. Python SDK Example

```python
import requests
import base64
from PIL import Image
from io import BytesIO

class AetheristClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})
    
    def generate(self, **kwargs):
        response = self.session.post(
            f"{self.base_url}/generate",
            json=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def decode_images(self, response):
        images = []
        for img_data in response["images"]:
            img_bytes = base64.b64decode(img_data)
            images.append(Image.open(BytesIO(img_bytes)))
        return images

# Usage
client = AetheristClient("your_api_key")
result = client.generate(
    num_samples=2,
    resolution=512,
    style="artistic"
)
images = client.decode_images(result)
for i, img in enumerate(images):
    img.save(f"generated_{i}.png")
```

## API Endpoints

### Image Generation

#### POST /generate

Generate high-quality artistic images.

**Parameters:**
- `num_samples` (required): Number of images (1-16)
- `seed` (optional): Random seed for reproducibility
- `resolution` (optional): Output resolution (256, 512, 1024)
- `camera_angles` (optional): Array of [pitch, yaw] camera angles
- `style` (optional): Generation style ("artistic", "photorealistic", "abstract")
- `guidance_scale` (optional): Guidance scale (1.0-20.0)

**Response:**
```json
{
  "images": ["base64_encoded_image_data", ...],
  "metadata": {
    "generation_time": 2.34,
    "model_version": "v1.0.0",
    "parameters": {...}
  }
}
```

### System Monitoring

#### GET /health

Check API health and system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "model_loaded": true,
  "gpu_available": true,
  "memory_usage": {
    "used_gb": 8.5,
    "total_gb": 16.0,
    "percentage": 53.1
  }
}
```

#### GET /monitoring/metrics

Get detailed system metrics.

**Response:**
```json
{
  "cpu": {
    "usage_percent": 45.2,
    "temperature": 65.0
  },
  "memory": {
    "used_gb": 12.3,
    "total_gb": 32.0,
    "percentage": 38.4
  },
  "gpu": {
    "usage_percent": 78.5,
    "memory_used_gb": 6.2,
    "memory_total_gb": 8.0,
    "temperature": 72.0
  },
  "performance": {
    "avg_generation_time": 1.85,
    "requests_per_minute": 12.5,
    "total_generations": 1247
  }
}
```

### Batch Processing

#### POST /batch/jobs

Submit a batch processing job for large-scale generation.

**Parameters:**
```json
{
  "job_name": "large_generation",
  "num_samples": 1000,
  "batch_size": 32,
  "parameters": {
    "resolution": 512,
    "style": "artistic"
  },
  "output_format": "png",
  "webhook_url": "https://your-app.com/webhook"
}
```

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "estimated_completion": "2024-01-01T14:30:00Z"
}
```

#### GET /batch/jobs/{job_id}

Get batch job status and progress.

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "running",
  "progress": {
    "completed": 245,
    "total": 1000,
    "percentage": 24.5
  },
  "estimated_completion": "2024-01-01T14:15:00Z",
  "results_url": "https://storage.aetherist.ai/job_123456/"
}
```

### Model Analysis

#### GET /models

List available models and their configurations.

#### POST /models/{model_id}/analyze

Perform comprehensive model analysis including architecture, performance, and quality metrics.

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

```json
{
  "error": "validation_error",
  "message": "Invalid parameter: resolution must be 256, 512, or 1024",
  "details": {
    "field": "resolution",
    "value": 128,
    "allowed_values": [256, 512, 1024]
  }
}
```

### Common Error Codes

- **400**: Bad Request - Invalid parameters
- **401**: Unauthorized - Invalid or missing API key
- **429**: Rate Limited - Too many requests
- **500**: Internal Server Error - Server-side error
- **503**: Service Unavailable - Model loading or maintenance

## SDK and Libraries

### Official SDKs

- **Python**: `pip install aetherist-python`
- **JavaScript/Node.js**: `npm install aetherist-js`
- **Go**: `go get github.com/aetherist/go-sdk`

### Community Libraries

- **R**: Available on CRAN
- **Ruby**: Available as gem
- **PHP**: Available via Composer

## Webhooks

For batch jobs and long-running operations, you can specify a webhook URL to receive notifications:

```json
{
  "event": "job.completed",
  "job_id": "job_123456",
  "status": "completed",
  "results": {
    "total_generated": 1000,
    "success_count": 998,
    "error_count": 2,
    "download_url": "https://storage.aetherist.ai/job_123456.zip"
  },
  "timestamp": "2024-01-01T14:30:00Z"
}
```

## Best Practices

1. **Use appropriate batch sizes**: For single requests, use 1-4 images. For batch jobs, use 16-32.

2. **Handle rate limits gracefully**: Implement exponential backoff for retries.

3. **Cache results**: Generated images are deterministic with the same seed.

4. **Monitor usage**: Use the `/monitoring/metrics` endpoint to track performance.

5. **Use webhooks for batch jobs**: Don't poll for status on long-running jobs.

## Support

- **Documentation**: https://docs.aetherist.ai
- **GitHub**: https://github.com/Saksham932007/Aetherist
- **Discord**: https://discord.gg/aetherist
- **Email**: support@aetherist.ai

## Changelog

### v1.0.0 (2024-01-01)
- Initial API release
- Basic image generation
- Batch processing
- System monitoring

---

*This documentation is automatically generated from the OpenAPI specification.*
"""

def generate_code_examples() -> Dict[str, str]:
    """Generate code examples in multiple languages."""
    
    return {
        "python": """
# Python Example - Aetherist API Client
import requests
import base64
from PIL import Image
from io import BytesIO
import time

class AetheristClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        })
    
    def generate(self, **kwargs):
        """Generate images with given parameters."""
        response = self.session.post(f"{self.base_url}/generate", json=kwargs)
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def decode_images(self, response):
        """Decode base64 images to PIL Images."""
        images = []
        for img_data in response["images"]:
            img_bytes = base64.b64decode(img_data)
            images.append(Image.open(BytesIO(img_bytes)))
        return images
    
    def submit_batch_job(self, **kwargs):
        """Submit a batch processing job."""
        response = self.session.post(f"{self.base_url}/batch/jobs", json=kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_batch_status(self, job_id):
        """Get batch job status."""
        response = self.session.get(f"{self.base_url}/batch/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_batch(self, job_id, timeout=3600, poll_interval=30):
        """Wait for batch job to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_batch_status(job_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Batch job {job_id} did not complete within {timeout} seconds")

# Usage Examples
if __name__ == "__main__":
    client = AetheristClient("your_api_key_here")
    
    # Health check
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Basic generation
    result = client.generate(
        num_samples=2,
        seed=42,
        resolution=512,
        style="artistic"
    )
    
    # Save generated images
    images = client.decode_images(result)
    for i, img in enumerate(images):
        img.save(f"generated_{i}.png")
    
    # Advanced generation with camera control
    advanced_result = client.generate(
        num_samples=1,
        seed=123,
        resolution=1024,
        camera_angles=[[0, 0]],
        style="photorealistic",
        guidance_scale=7.5
    )
    
    # Batch job example
    batch_job = client.submit_batch_job(
        job_name="large_generation",
        num_samples=100,
        batch_size=16,
        parameters={
            "resolution": 512,
            "style": "artistic"
        }
    )
    
    print(f"Submitted batch job: {batch_job['job_id']}")
    
    # Wait for completion (optional)
    final_status = client.wait_for_batch(batch_job['job_id'])
    print(f"Batch job completed: {final_status}")
""",
        
        "javascript": """
// JavaScript Example - Aetherist API Client
class AetheristClient {
    constructor(apiKey, baseUrl = 'http://localhost:8000') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }
    
    async generate(params) {
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error(`Generation failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async submitBatchJob(params) {
        const response = await fetch(`${this.baseUrl}/batch/jobs`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error(`Batch job submission failed: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async getBatchStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/batch/jobs/${jobId}`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`Failed to get batch status: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    decodeImage(base64Data) {
        // In browser environment
        return `data:image/png;base64,${base64Data}`;
    }
}

// Usage Examples
async function main() {
    const client = new AetheristClient('your_api_key_here');
    
    try {
        // Health check
        const health = await client.healthCheck();
        console.log('API Status:', health.status);
        
        // Basic generation
        const result = await client.generate({
            num_samples: 2,
            seed: 42,
            resolution: 512,
            style: 'artistic'
        });
        
        // Display images in browser
        result.images.forEach((imgData, index) => {
            const img = document.createElement('img');
            img.src = client.decodeImage(imgData);
            img.alt = `Generated image ${index + 1}`;
            document.body.appendChild(img);
        });
        
        // Submit batch job
        const batchJob = await client.submitBatchJob({
            job_name: 'web_generation',
            num_samples: 50,
            batch_size: 8,
            parameters: {
                resolution: 512,
                style: 'artistic'
            }
        });
        
        console.log('Submitted batch job:', batchJob.job_id);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Run if in Node.js environment
if (typeof window === 'undefined') {
    main();
}
""",
        
        "curl": """
# cURL Examples - Aetherist API

# Health Check
curl -X GET "http://localhost:8000/health" \
  -H "X-API-Key: your_api_key_here"

# Basic Image Generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "num_samples": 4,
    "seed": 42,
    "resolution": 512,
    "style": "artistic"
  }'

# Advanced Generation with Camera Control
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "num_samples": 2,
    "seed": 123,
    "resolution": 1024,
    "camera_angles": [[0, 0], [30, 45]],
    "style": "photorealistic",
    "guidance_scale": 7.5
  }'

# Submit Batch Job
curl -X POST "http://localhost:8000/batch/jobs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "job_name": "large_generation",
    "num_samples": 1000,
    "batch_size": 32,
    "parameters": {
      "resolution": 512,
      "style": "artistic"
    },
    "output_format": "png"
  }'

# Get Batch Job Status
curl -X GET "http://localhost:8000/batch/jobs/job_123456" \
  -H "X-API-Key: your_api_key_here"

# Get System Metrics
curl -X GET "http://localhost:8000/monitoring/metrics" \
  -H "X-API-Key: your_api_key_here"

# List Available Models
curl -X GET "http://localhost:8000/models" \
  -H "X-API-Key: your_api_key_here"

# Analyze Model
curl -X POST "http://localhost:8000/models/aetherist-v1/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "analysis_types": ["architecture", "performance", "quality"],
    "batch_sizes": [1, 4, 8, 16]
  }'
"""
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Aetherist API documentation")
    parser.add_argument("--output-dir", type=str, default="docs/api",
                       help="Output directory for documentation")
    parser.add_argument("--format", type=str, choices=["openapi", "markdown", "all"],
                       default="all", help="Documentation format to generate")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating API documentation in {output_dir}")
    
    if args.format in ["openapi", "all"]:
        # Generate OpenAPI specification
        openapi_spec = generate_openapi_spec()
        with open(output_dir / "openapi.json", "w") as f:
            json.dump(openapi_spec, f, indent=2)
        print("✓ Generated OpenAPI specification (openapi.json)")
    
    if args.format in ["markdown", "all"]:
        # Generate Markdown documentation
        markdown_docs = generate_api_docs_markdown()
        with open(output_dir / "README.md", "w") as f:
            f.write(markdown_docs)
        print("✓ Generated Markdown documentation (README.md)")
        
        # Generate code examples
        examples = generate_code_examples()
        examples_dir = output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        for lang, code in examples.items():
            ext = {"python": "py", "javascript": "js", "curl": "sh"}[lang]
            with open(examples_dir / f"example.{ext}", "w") as f:
                f.write(code)
        print("✓ Generated code examples")
    
    print("\n🎉 API documentation generation complete!")
    print(f"\n📁 Documentation available in: {output_dir}")
    print("\n📖 Files generated:")
    if args.format in ["openapi", "all"]:
        print("  • openapi.json - OpenAPI 3.0 specification")
    if args.format in ["markdown", "all"]:
        print("  • README.md - Comprehensive API guide")
        print("  • examples/ - Code examples in multiple languages")
    
if __name__ == "__main__":
    main()
