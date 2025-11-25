"""Batch processing utilities for Aetherist.

Provides efficient batch processing for large-scale image generation,
analysis, and data processing tasks.
"""

import time
import json
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import logging
from queue import Queue
from threading import Thread, Event
import uuid

try:
    import torch
    import numpy as np
    from PIL import Image
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    job_type: str
    input_data: Any
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None
    progress: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
            
@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 4
    batch_size: int = 8
    max_queue_size: int = 100
    use_multiprocessing: bool = False
    checkpoint_interval: int = 10  # Save progress every N jobs
    retry_failed: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    output_dir: str = "outputs/batch_results"
    save_intermediate: bool = True
    
class BatchProcessor:
    """High-performance batch processor for image generation and analysis."""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.job_queue: Queue = Queue(maxsize=config.max_queue_size)
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.failed_jobs: Dict[str, BatchJob] = {}
        self.processing = False
        self.stop_event = Event()
        self.workers: List[Thread] = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing functions registry
        self.processors: Dict[str, Callable] = {}
        
    def register_processor(self, job_type: str, processor_func: Callable):
        """Register a processing function for a job type."""
        self.processors[job_type] = processor_func
        self.logger.info(f"Registered processor for job type: {job_type}")
        
    def add_job(self, 
                job_type: str,
                input_data: Any,
                parameters: Optional[Dict[str, Any]] = None,
                job_id: Optional[str] = None) -> str:
        """Add a job to the processing queue."""
        if job_type not in self.processors:
            raise ValueError(f"No processor registered for job type: {job_type}")
            
        if job_id is None:
            job_id = str(uuid.uuid4())
            
        if parameters is None:
            parameters = {}
            
        job = BatchJob(
            job_id=job_id,
            job_type=job_type,
            input_data=input_data,
            parameters=parameters
        )
        
        try:
            self.job_queue.put(job, timeout=5.0)
            self.logger.info(f"Added job {job_id} of type {job_type}")
            return job_id
        except:
            raise RuntimeError("Job queue is full")
            
    def add_batch_jobs(self, 
                      job_type: str,
                      input_data_list: List[Any],
                      parameters_list: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add multiple jobs as a batch."""
        if parameters_list is None:
            parameters_list = [{}] * len(input_data_list)
            
        if len(input_data_list) != len(parameters_list):
            raise ValueError("input_data_list and parameters_list must have same length")
            
        job_ids = []
        for input_data, parameters in zip(input_data_list, parameters_list):
            job_id = self.add_job(job_type, input_data, parameters)
            job_ids.append(job_id)
            
        self.logger.info(f"Added {len(job_ids)} batch jobs of type {job_type}")
        return job_ids
        
    def start_processing(self) -> None:
        """Start batch processing."""
        if self.processing:
            self.logger.warning("Processing already started")
            return
            
        self.processing = True
        self.stop_event.clear()
        
        # Start worker threads
        for i in range(self.config.max_workers):
            worker = Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
            
        self.logger.info(f"Started batch processing with {self.config.max_workers} workers")
        
    def stop_processing(self, wait: bool = True) -> None:
        """Stop batch processing."""
        if not self.processing:
            return
            
        self.processing = False
        self.stop_event.set()
        
        if wait:
            for worker in self.workers:
                worker.join(timeout=10.0)
                
        self.workers.clear()
        self.logger.info("Stopped batch processing")
        
    def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop."""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.processing and not self.stop_event.is_set():
            try:
                # Get job from queue with timeout
                job = self.job_queue.get(timeout=1.0)
                
                # Process job
                self._process_job(job, worker_id)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except:
                # Timeout or queue empty, continue
                continue
                
        self.logger.info(f"Worker {worker_id} stopped")
        
    def _process_job(self, job: BatchJob, worker_id: int) -> None:
        """Process a single job."""
        job.status = "processing"
        job.started_at = time.time()
        
        try:
            self.logger.info(f"Worker {worker_id} processing job {job.job_id}")
            
            # Get processor function
            processor = self.processors[job.job_type]
            
            # Execute with timeout if configured
            if self.config.timeout_seconds:
                result = self._execute_with_timeout(
                    processor, job.input_data, job.parameters, 
                    self.config.timeout_seconds
                )
            else:
                result = processor(job.input_data, **job.parameters)
                
            # Store result
            job.result = result
            job.status = "completed"
            job.completed_at = time.time()
            job.progress = 1.0
            
            self.completed_jobs[job.job_id] = job
            
            # Save intermediate results if configured
            if self.config.save_intermediate:
                self._save_job_result(job)
                
            self.logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = time.time()
            
            self.failed_jobs[job.job_id] = job
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            
            # Retry if configured
            if self.config.retry_failed and job.parameters.get("retry_count", 0) < self.config.max_retries:
                retry_job = BatchJob(
                    job_id=f"{job.job_id}_retry_{job.parameters.get('retry_count', 0) + 1}",
                    job_type=job.job_type,
                    input_data=job.input_data,
                    parameters={**job.parameters, "retry_count": job.parameters.get("retry_count", 0) + 1}
                )
                
                try:
                    self.job_queue.put(retry_job, timeout=1.0)
                    self.logger.info(f"Retrying job {job.job_id}")
                except:
                    self.logger.warning(f"Failed to queue retry for job {job.job_id}")
                    
    def _execute_with_timeout(self, func: Callable, input_data: Any, parameters: Dict, timeout: int) -> Any:
        """Execute function with timeout."""
        # This is a simplified version - in production, you might want to use
        # more sophisticated timeout handling
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution exceeded {timeout} seconds")
            
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func(input_data, **parameters)
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    def _save_job_result(self, job: BatchJob) -> None:
        """Save job result to file."""
        try:
            result_dir = self.output_dir / job.job_type
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata = {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "processing_time": job.completed_at - job.started_at if job.completed_at and job.started_at else None,
                "parameters": job.parameters,
                "error_message": job.error_message
            }
            
            with open(result_dir / f"{job.job_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save result data if it's a tensor or array
            if hasattr(job.result, 'numpy'):
                # PyTorch tensor
                np.save(result_dir / f"{job.job_id}_result.npy", job.result.detach().cpu().numpy())
            elif isinstance(job.result, np.ndarray):
                np.save(result_dir / f"{job.job_id}_result.npy", job.result)
            elif isinstance(job.result, (list, dict)):
                with open(result_dir / f"{job.job_id}_result.json", 'w') as f:
                    json.dump(job.result, f, indent=2)
                    
        except Exception as e:
            self.logger.warning(f"Failed to save job result: {e}")
            
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "processing_time": job.completed_at - job.started_at if job.completed_at and job.started_at else None
            }
            
        # Check failed jobs
        if job_id in self.failed_jobs:
            job = self.failed_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "error_message": job.error_message,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at
            }
            
        return None
        
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing status."""
        total_completed = len(self.completed_jobs)
        total_failed = len(self.failed_jobs)
        queue_size = self.job_queue.qsize()
        
        # Calculate average processing time
        avg_processing_time = 0.0
        if self.completed_jobs:
            processing_times = [
                job.completed_at - job.started_at 
                for job in self.completed_jobs.values() 
                if job.completed_at and job.started_at
            ]
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
                
        return {
            "total_completed": total_completed,
            "total_failed": total_failed,
            "queue_size": queue_size,
            "avg_processing_time": avg_processing_time,
            "is_processing": self.processing,
            "active_workers": len(self.workers),
            "success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0.0
        }
        
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all jobs to complete."""
        start_time = time.time()
        
        while self.processing:
            if self.job_queue.empty():
                # Give workers a moment to finish
                time.sleep(1.0)
                if self.job_queue.empty():
                    break
                    
            if timeout and (time.time() - start_time) > timeout:
                return False
                
            time.sleep(0.1)
            
        return True
        
    def export_results(self, output_path: Path, format_type: str = "json") -> None:
        """Export processing results."""
        results_summary = {
            "processing_summary": self.get_processing_summary(),
            "completed_jobs": {
                job_id: {
                    "job_type": job.job_type,
                    "status": job.status,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "processing_time": job.completed_at - job.started_at if job.completed_at and job.started_at else None,
                    "parameters": job.parameters
                }
                for job_id, job in self.completed_jobs.items()
            },
            "failed_jobs": {
                job_id: {
                    "job_type": job.job_type,
                    "status": job.status,
                    "error_message": job.error_message,
                    "created_at": job.created_at,
                    "parameters": job.parameters
                }
                for job_id, job in self.failed_jobs.items()
            },
            "export_timestamp": time.time()
        }
        
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        self.logger.info(f"Results exported to {output_path}")
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_processing(wait=True)
        self.completed_jobs.clear()
        self.failed_jobs.clear()
        
        # Clear queue
        while not self.job_queue.empty():
            try:
                self.job_queue.get_nowait()
            except:
                break
                
        self.logger.info("Batch processor cleanup completed")
        
    def __enter__(self):
        self.start_processing()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Built-in processors for common tasks
def dummy_processor(input_data: Any, **kwargs) -> Dict[str, Any]:
    """Dummy processor for testing."""
    time.sleep(kwargs.get("sleep_time", 0.1))
    return {
        "input": str(input_data),
        "processed_at": time.time(),
        "parameters": kwargs
    }

def image_resize_processor(image_path: Union[str, Path], 
                         target_size: tuple = (256, 256), 
                         **kwargs) -> Dict[str, Any]:
    """Resize image processor."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for image processing")
        
    try:
        image = Image.open(image_path)
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save resized image
        output_path = kwargs.get("output_path")
        if output_path:
            resized_image.save(output_path)
            
        return {
            "original_size": image.size,
            "target_size": target_size,
            "output_path": output_path,
            "processed_at": time.time()
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to process image {image_path}: {e}")

def tensor_computation_processor(tensor_data: Any, 
                               operation: str = "mean", 
                               **kwargs) -> Dict[str, Any]:
    """Generic tensor computation processor."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for tensor processing")
        
    if not isinstance(tensor_data, torch.Tensor):
        tensor_data = torch.tensor(tensor_data)
        
    if operation == "mean":
        result = torch.mean(tensor_data)
    elif operation == "std":
        result = torch.std(tensor_data)
    elif operation == "sum":
        result = torch.sum(tensor_data)
    elif operation == "norm":
        result = torch.norm(tensor_data)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
        
    return {
        "input_shape": list(tensor_data.shape),
        "operation": operation,
        "result": float(result),
        "processed_at": time.time()
    }
