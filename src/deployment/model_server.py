import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import redis
from contextlib import asynccontextmanager

from ..config.config_manager import ConfigManager
from ..inference.inference_pipeline import AetheristInferencePipeline
from ..utils.camera_utils import CameraPoseGenerator

# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for image generation."""
    latent_code: Optional[List[float]] = None
    seed: Optional[int] = None
    camera_pose: Optional[Dict[str, float]] = None
    resolution: Tuple[int, int] = (512, 512)
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    
class GenerationResponse(BaseModel):
    """Response model for image generation."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    image_url: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
class BatchGenerationRequest(BaseModel):
    """Request model for batch image generation."""
    requests: List[GenerationRequest]
    batch_size: int = 4
    
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime: float
    version: str
    
@dataclass
class ServerConfig:
    """Configuration for model server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 4
    max_queue_size: int = 100
    redis_url: Optional[str] = None
    enable_auth: bool = False
    api_key: Optional[str] = None
    cors_origins: List[str] = None
    log_level: str = "INFO"
    model_config_path: str = "configs/base_config.yaml"
    checkpoint_path: Optional[str] = None
    
class ModelServerState:
    """Global state for the model server."""
    def __init__(self):
        self.inference_pipeline = None
        self.camera_generator = None
        self.config = None
        self.redis_client = None
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.start_time = time.time()
        self.tasks = {}  # task_id -> task_info
        
server_state = ModelServerState()

class AetheristModelServer:
    """Production model server for Aetherist."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.app = self._create_app()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
            
        app = FastAPI(
            title="Aetherist Model Server",
            description="Production API for Aetherist 3D-aware image generation",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        if self.config.cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
        # Add routes
        self._add_routes(app)
        
        return app
        
    async def _startup(self):
        """Initialize server components."""
        self.logger.info("Starting Aetherist Model Server...")
        
        # Load model configuration
        config_manager = ConfigManager()
        server_state.config = config_manager.load_config(self.config.model_config_path)
        
        # Initialize inference pipeline
        try:
            server_state.inference_pipeline = AetheristInferencePipeline(server_state.config)
            if self.config.checkpoint_path:
                server_state.inference_pipeline.load_checkpoint(self.config.checkpoint_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
        # Initialize camera generator
        server_state.camera_generator = CameraPoseGenerator()
        
        # Initialize Redis if configured
        if self.config.redis_url:
            try:
                server_state.redis_client = redis.from_url(self.config.redis_url)
                server_state.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                
        # Start background task processor
        asyncio.create_task(self._process_tasks())
        
        self.logger.info("Server startup complete")
        
    async def _shutdown(self):
        """Cleanup server components."""
        self.logger.info("Shutting down server...")
        
        if server_state.executor:
            server_state.executor.shutdown(wait=True)
            
        if server_state.redis_client:
            server_state.redis_client.close()
            
    def _add_routes(self, app: FastAPI):
        """Add API routes to the application."""
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                gpu_available = torch.cuda.is_available()
                memory_usage = {}
                
                if gpu_available:
                    memory_usage["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3
                    memory_usage["gpu_cached"] = torch.cuda.memory_reserved() / 1024**3
                    
                uptime = time.time() - server_state.start_time
                
                return HealthResponse(
                    status="healthy",
                    model_loaded=server_state.inference_pipeline is not None,
                    gpu_available=gpu_available,
                    memory_usage=memory_usage,
                    uptime=uptime,
                    version="1.0.0"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.post("/generate", response_model=GenerationResponse)
        async def generate_image(request: GenerationRequest, background_tasks: BackgroundTasks):
            """Generate a single image."""
            try:
                task_id = f"task_{int(time.time() * 1000)}_{id(request)}"
                
                # Add task to queue
                task_info = {
                    "id": task_id,
                    "request": request,
                    "status": "pending",
                    "created_at": time.time()
                }
                
                server_state.tasks[task_id] = task_info
                await server_state.task_queue.put(task_info)
                
                return GenerationResponse(task_id=task_id, status="pending")
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.post("/generate_batch", response_model=List[GenerationResponse])
        async def generate_batch(request: BatchGenerationRequest):
            """Generate multiple images in batch."""
            try:
                responses = []
                
                for i, gen_request in enumerate(request.requests):
                    task_id = f"batch_task_{int(time.time() * 1000)}_{i}"
                    
                    task_info = {
                        "id": task_id,
                        "request": gen_request,
                        "status": "pending",
                        "created_at": time.time(),
                        "batch_id": f"batch_{int(time.time() * 1000)}"
                    }
                    
                    server_state.tasks[task_id] = task_info
                    await server_state.task_queue.put(task_info)
                    
                    responses.append(GenerationResponse(task_id=task_id, status="pending"))
                    
                return responses
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.get("/task/{task_id}", response_model=GenerationResponse)
        async def get_task_status(task_id: str):
            """Get task status and result."""
            if task_id not in server_state.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
                
            task_info = server_state.tasks[task_id]
            
            return GenerationResponse(
                task_id=task_id,
                status=task_info["status"],
                image_url=task_info.get("image_url"),
                error_message=task_info.get("error_message"),
                processing_time=task_info.get("processing_time"),
                metadata=task_info.get("metadata")
            )
            
        @app.get("/image/{task_id}")
        async def get_generated_image(task_id: str):
            """Download generated image."""
            if task_id not in server_state.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
                
            task_info = server_state.tasks[task_id]
            
            if task_info["status"] != "completed":
                raise HTTPException(status_code=400, detail="Task not completed")
                
            if "image_data" not in task_info:
                raise HTTPException(status_code=404, detail="Image not found")
                
            image_bytes = io.BytesIO(task_info["image_data"])
            
            return StreamingResponse(
                io.BytesIO(task_info["image_data"]),
                media_type="image/png",
                headers={"Content-Disposition": f"attachment; filename=generated_{task_id}.png"}
            )
            
    async def _process_tasks(self):
        """Background task processor."""
        self.logger.info("Starting task processor")
        
        while True:
            try:
                # Get task from queue
                task_info = await server_state.task_queue.get()
                
                # Update task status
                task_info["status"] = "processing"
                task_info["started_at"] = time.time()
                
                # Process task in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    server_state.executor,
                    self._process_generation_task,
                    task_info
                )
                
                # Update task with result
                task_info.update(result)
                task_info["completed_at"] = time.time()
                task_info["processing_time"] = task_info["completed_at"] - task_info["started_at"]
                
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
                if 'task_info' in locals():
                    task_info["status"] = "failed"
                    task_info["error_message"] = str(e)
                    
    def _process_generation_task(self, task_info: Dict) -> Dict:
        """Process a single generation task."""
        try:
            request = task_info["request"]
            
            # Generate latent code if not provided
            if request.latent_code is None:
                if request.seed is not None:
                    torch.manual_seed(request.seed)
                latent = torch.randn(1, 512)  # Standard latent dimension
            else:
                latent = torch.tensor([request.latent_code])
                
            # Generate camera pose if not provided
            if request.camera_pose is None:
                camera_params = server_state.camera_generator.sample_random_pose()
            else:
                camera_params = request.camera_pose
                
            # Generate image
            with torch.no_grad():
                generated_image = server_state.inference_pipeline.generate(
                    latent_codes=latent,
                    camera_params=camera_params,
                    output_size=request.resolution
                )
                
            # Convert to PIL Image and save as bytes
            if isinstance(generated_image, torch.Tensor):
                generated_image = generated_image.squeeze().cpu().numpy()
                generated_image = (generated_image * 255).astype(np.uint8)
                
            if generated_image.ndim == 3 and generated_image.shape[0] == 3:
                generated_image = generated_image.transpose(1, 2, 0)
                
            pil_image = Image.fromarray(generated_image)
            
            # Save image as bytes
            image_bytes = io.BytesIO()
            pil_image.save(image_bytes, format='PNG')
            image_data = image_bytes.getvalue()
            
            return {
                "status": "completed",
                "image_data": image_data,
                "image_url": f"/image/{task_info['id']}",
                "metadata": {
                    "resolution": request.resolution,
                    "camera_params": camera_params,
                    "latent_shape": list(latent.shape)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                "status": "failed",
                "error_message": str(e)
            }
            
    def run(self):
        """Run the server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level=self.config.log_level.lower()
        )
        
def create_server(config_path: Optional[str] = None) -> AetheristModelServer:
    """Factory function to create model server."""
    if config_path:
        # Load server config from file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ServerConfig(**config_dict)
    else:
        config = ServerConfig()
        
    return AetheristModelServer(config)
