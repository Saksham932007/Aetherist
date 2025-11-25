"""System monitoring utilities for Aetherist.

Provides real-time monitoring of model performance, resource usage,
and training/inference metrics.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from collections import defaultdict, deque
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_metrics: Optional[Dict[str, Any]] = None
    
@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: float
    batch_size: int
    inference_time: float
    throughput: float  # samples per second
    memory_allocated: Optional[float] = None  # GB
    memory_cached: Optional[float] = None  # GB
    model_name: Optional[str] = None
    
@dataclass
class TrainingMetrics:
    """Training progress metrics."""
    timestamp: float
    epoch: int
    step: int
    generator_loss: float
    discriminator_loss: float
    learning_rate: float
    batch_time: float
    data_time: float
    memory_usage: Optional[float] = None
    
class MetricsCollector:
    """Collects and stores various types of metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.model_metrics: deque = deque(maxlen=max_history)
        self.training_metrics: deque = deque(maxlen=max_history)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
    def add_system_metrics(self, metrics: SystemMetrics) -> None:
        """Add system metrics."""
        self.system_metrics.append(metrics)
        
    def add_model_metrics(self, metrics: ModelMetrics) -> None:
        """Add model performance metrics."""
        self.model_metrics.append(metrics)
        
    def add_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Add training metrics."""
        self.training_metrics.append(metrics)
        
    def add_custom_metric(self, name: str, value: Any, timestamp: Optional[float] = None) -> None:
        """Add custom metric."""
        if timestamp is None:
            timestamp = time.time()
        self.custom_metrics[name].append({"timestamp": timestamp, "value": value})
        
    def get_recent_metrics(self, metric_type: str, count: int = 10) -> List[Any]:
        """Get recent metrics of specified type."""
        if metric_type == "system":
            return list(self.system_metrics)[-count:]
        elif metric_type == "model":
            return list(self.model_metrics)[-count:]
        elif metric_type == "training":
            return list(self.training_metrics)[-count:]
        elif metric_type in self.custom_metrics:
            return list(self.custom_metrics[metric_type])[-count:]
        else:
            return []
            
    def clear_metrics(self, metric_type: Optional[str] = None) -> None:
        """Clear metrics."""
        if metric_type is None:
            self.system_metrics.clear()
            self.model_metrics.clear()
            self.training_metrics.clear()
            self.custom_metrics.clear()
        elif metric_type == "system":
            self.system_metrics.clear()
        elif metric_type == "model":
            self.model_metrics.clear()
        elif metric_type == "training":
            self.training_metrics.clear()
        elif metric_type in self.custom_metrics:
            self.custom_metrics[metric_type].clear()
            
class SystemMonitor:
    """Real-time system monitoring."""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 enable_gpu_monitoring: bool = True,
                 max_history: int = 1000):
        self.collection_interval = collection_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPUTIL_AVAILABLE
        self.metrics_collector = MetricsCollector(max_history)
        self.logger = logging.getLogger(__name__)
        
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring:
            self.logger.warning("Monitoring already started")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_collector.add_system_metrics(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_metrics = None
        if self.enable_gpu_monitoring:
            gpu_metrics = self._collect_gpu_metrics()
            
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_metrics=gpu_metrics
        )
        
    def _collect_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect GPU metrics."""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
                
            gpu_data = []
            for gpu in gpus:
                gpu_info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Convert to percentage
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "temperature": gpu.temperature
                }
                gpu_data.append(gpu_info)
                
            # Add PyTorch GPU metrics if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch_gpu_info = {
                    "torch_allocated": torch.cuda.memory_allocated() / (1024**3),
                    "torch_cached": torch.cuda.memory_reserved() / (1024**3),
                    "torch_device_count": torch.cuda.device_count()
                }
                
            return {
                "gpus": gpu_data,
                "torch_info": torch_gpu_info if 'torch_gpu_info' in locals() else None
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to collect GPU metrics: {e}")
            return None
            
    @contextmanager
    def monitor_inference(self, batch_size: int, model_name: Optional[str] = None):
        """Context manager for monitoring inference."""
        start_time = time.time()
        start_memory = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
            
        try:
            yield
        finally:
            end_time = time.time()
            inference_time = end_time - start_time
            throughput = batch_size / inference_time if inference_time > 0 else 0
            
            # Memory metrics
            memory_allocated = None
            memory_cached = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_cached = torch.cuda.memory_reserved() / (1024**3)
                
            metrics = ModelMetrics(
                timestamp=end_time,
                batch_size=batch_size,
                inference_time=inference_time,
                throughput=throughput,
                memory_allocated=memory_allocated,
                memory_cached=memory_cached,
                model_name=model_name
            )
            
            self.metrics_collector.add_model_metrics(metrics)
            
    def get_system_summary(self) -> Dict[str, Any]:
        """Get current system summary."""
        recent_metrics = self.metrics_collector.get_recent_metrics("system", 1)
        if not recent_metrics:
            return {"status": "no_data"}
            
        metrics = recent_metrics[0]
        
        summary = {
            "timestamp": metrics.timestamp,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_gb": metrics.memory_used_gb,
            "memory_total_gb": metrics.memory_total_gb,
            "disk_usage_percent": metrics.disk_usage_percent,
            "status": "healthy" if metrics.cpu_percent < 90 and metrics.memory_percent < 90 else "warning"
        }
        
        if metrics.gpu_metrics:
            summary["gpu_metrics"] = metrics.gpu_metrics
            
        return summary
        
    def get_performance_summary(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary over time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        # Filter recent metrics
        recent_model = [m for m in self.metrics_collector.model_metrics if m.timestamp > cutoff_time]
        recent_system = [m for m in self.metrics_collector.system_metrics if m.timestamp > cutoff_time]
        
        summary = {
            "window_minutes": window_minutes,
            "model_metrics": {},
            "system_metrics": {},
            "timestamp": time.time()
        }
        
        # Model performance summary
        if recent_model:
            throughputs = [m.throughput for m in recent_model]
            inference_times = [m.inference_time for m in recent_model]
            
            summary["model_metrics"] = {
                "total_inferences": len(recent_model),
                "avg_throughput": sum(throughputs) / len(throughputs),
                "max_throughput": max(throughputs),
                "min_throughput": min(throughputs),
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "max_inference_time": max(inference_times),
                "min_inference_time": min(inference_times)
            }
            
        # System resource summary
        if recent_system:
            cpu_values = [m.cpu_percent for m in recent_system]
            memory_values = [m.memory_percent for m in recent_system]
            
            summary["system_metrics"] = {
                "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                "max_cpu_percent": max(cpu_values),
                "avg_memory_percent": sum(memory_values) / len(memory_values),
                "max_memory_percent": max(memory_values),
                "sample_count": len(recent_system)
            }
            
        return summary
        
    def export_metrics(self, 
                      filepath: Path, 
                      format_type: str = "json",
                      window_hours: Optional[int] = None) -> None:
        """Export metrics to file."""
        cutoff_time = None
        if window_hours:
            cutoff_time = time.time() - (window_hours * 3600)
            
        # Collect metrics
        system_metrics = [
            {
                "timestamp": m.timestamp,
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "memory_used_gb": m.memory_used_gb,
                "gpu_metrics": m.gpu_metrics
            }
            for m in self.metrics_collector.system_metrics
            if cutoff_time is None or m.timestamp > cutoff_time
        ]
        
        model_metrics = [
            {
                "timestamp": m.timestamp,
                "batch_size": m.batch_size,
                "inference_time": m.inference_time,
                "throughput": m.throughput,
                "memory_allocated": m.memory_allocated,
                "model_name": m.model_name
            }
            for m in self.metrics_collector.model_metrics
            if cutoff_time is None or m.timestamp > cutoff_time
        ]
        
        export_data = {
            "export_timestamp": time.time(),
            "window_hours": window_hours,
            "system_metrics": system_metrics,
            "model_metrics": model_metrics
        }
        
        if format_type == "json":
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        self.logger.info(f"Metrics exported to {filepath}")
        
    def __enter__(self):
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()

class TrainingMonitor:
    """Specialized monitor for training processes."""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self.system_monitor = system_monitor or SystemMonitor()
        self.logger = logging.getLogger(__name__)
        self.training_start_time = None
        self.current_epoch = 0
        self.current_step = 0
        
    def start_training_monitoring(self) -> None:
        """Start monitoring for training session."""
        self.training_start_time = time.time()
        self.system_monitor.start_monitoring()
        self.logger.info("Training monitoring started")
        
    def stop_training_monitoring(self) -> None:
        """Stop training monitoring."""
        self.system_monitor.stop_monitoring()
        
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            
    def log_training_step(self,
                         epoch: int,
                         step: int,
                         generator_loss: float,
                         discriminator_loss: float,
                         learning_rate: float,
                         batch_time: float,
                         data_time: float) -> None:
        """Log training step metrics."""
        self.current_epoch = epoch
        self.current_step = step
        
        # Get memory usage
        memory_usage = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024**3)
            
        metrics = TrainingMetrics(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            learning_rate=learning_rate,
            batch_time=batch_time,
            data_time=data_time,
            memory_usage=memory_usage
        )
        
        self.system_monitor.metrics_collector.add_training_metrics(metrics)
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training session summary."""
        if not self.training_start_time:
            return {"status": "not_started"}
            
        elapsed_time = time.time() - self.training_start_time
        
        # Get recent training metrics
        recent_training = self.system_monitor.metrics_collector.get_recent_metrics("training", 10)
        
        summary = {
            "elapsed_time": elapsed_time,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "status": "running" if self.system_monitor._monitoring else "stopped"
        }
        
        if recent_training:
            recent_losses_g = [m.generator_loss for m in recent_training]
            recent_losses_d = [m.discriminator_loss for m in recent_training]
            recent_batch_times = [m.batch_time for m in recent_training]
            
            summary.update({
                "recent_generator_loss": {
                    "avg": sum(recent_losses_g) / len(recent_losses_g),
                    "min": min(recent_losses_g),
                    "max": max(recent_losses_g)
                },
                "recent_discriminator_loss": {
                    "avg": sum(recent_losses_d) / len(recent_losses_d),
                    "min": min(recent_losses_d),
                    "max": max(recent_losses_d)
                },
                "recent_batch_time": {
                    "avg": sum(recent_batch_times) / len(recent_batch_times),
                    "min": min(recent_batch_times),
                    "max": max(recent_batch_times)
                }
            })
            
        return summary
