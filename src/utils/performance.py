#!/usr/bin/env python3
"""Performance optimization utilities for Aetherist.

Provides mixed precision training, memory optimization, and performance tuning.
"""

import logging
import time
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
from functools import wraps

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    import torch.nn.utils as nn_utils
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Comprehensive performance optimization manager."""
    
    def __init__(self, 
                 mixed_precision: bool = True,
                 gradient_clipping: float = 1.0,
                 memory_efficient: bool = True,
                 profile_memory: bool = False):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for performance optimization")
            
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.gradient_clipping = gradient_clipping
        self.memory_efficient = memory_efficient
        self.profile_memory = profile_memory
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Performance tracking
        self.timing_stats = {}
        self.memory_stats = {}
        
        logger.info(f"Performance optimizer initialized:")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        logger.info(f"  Gradient clipping: {self.gradient_clipping}")
        logger.info(f"  Memory efficient: {self.memory_efficient}")
        
    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision forward pass."""
        if self.mixed_precision:
            with autocast():
                yield
        else:
            yield
            
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.mixed_precision:
            return self.scaler.scale(loss)
        return loss
        
    def backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                         model: nn.Module) -> Dict[str, float]:
        """Perform backward pass and optimizer step with optimizations."""
        stats = {}
        
        if self.mixed_precision:
            # Scaled backward
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                grad_norm = nn_utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clipping
                )
                stats["grad_norm"] = grad_norm.item()
                
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            
        else:
            # Standard backward
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                grad_norm = nn_utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clipping
                )
                stats["grad_norm"] = grad_norm.item()
                
            # Optimizer step
            optimizer.step()
            
        optimizer.zero_grad()
        return stats
        
    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference performance."""
        logger.info("Optimizing model for inference...")
        
        # Set to eval mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
            
        # Fuse operations if possible
        try:
            if hasattr(torch.jit, 'optimize_for_inference'):
                model = torch.jit.optimize_for_inference(
                    torch.jit.script(model)
                )
                logger.info("Applied TorchScript optimization")
        except Exception as e:
            logger.warning(f"Could not apply TorchScript optimization: {e}")
            
        # Memory optimization
        if self.memory_efficient:
            self._apply_memory_optimizations(model)
            
        return model
        
    def _apply_memory_optimizations(self, model: nn.Module):
        """Apply memory optimizations to model."""
        # Convert to half precision if using mixed precision
        if self.mixed_precision:
            model.half()
            logger.info("Converted model to half precision")
            
        # Optimize memory layout
        if hasattr(model, 'to_memory_efficient'):
            model.to_memory_efficient()
            
    @contextmanager
    def memory_profiling(self, operation_name: str):
        """Context manager for memory profiling."""
        if not self.profile_memory or not torch.cuda.is_available():
            yield
            return
            
        # Clear cache and measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            # Measure final memory
            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory
            
            self.memory_stats[operation_name] = {
                "memory_used_mb": memory_used / 1024 / 1024,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "initial_memory_mb": initial_memory / 1024 / 1024,
                "final_memory_mb": final_memory / 1024 / 1024
            }
            
    @contextmanager
    def timing_context(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.timing_stats:
                self.timing_stats[operation_name] = []
            self.timing_stats[operation_name].append(duration)
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "timing_stats": {},
            "memory_stats": self.memory_stats.copy()
        }
        
        # Compute timing statistics
        for op_name, timings in self.timing_stats.items():
            stats["timing_stats"][op_name] = {
                "count": len(timings),
                "mean": sum(timings) / len(timings),
                "min": min(timings),
                "max": max(timings),
                "total": sum(timings)
            }
            
        return stats
        
    def clear_stats(self):
        """Clear performance statistics."""
        self.timing_stats.clear()
        self.memory_stats.clear()
        
    def optimize_dataloader(self, dataloader) -> None:
        """Optimize dataloader for performance."""
        # Set optimal number of workers
        import multiprocessing
        optimal_workers = min(multiprocessing.cpu_count(), 8)
        
        if hasattr(dataloader, 'num_workers'):
            if dataloader.num_workers == 0:
                logger.info(f"Recommend setting num_workers={optimal_workers} for better performance")
                
        # Enable pin_memory for GPU training
        if torch.cuda.is_available() and hasattr(dataloader, 'pin_memory'):
            if not dataloader.pin_memory:
                logger.info("Recommend enabling pin_memory=True for GPU training")
                
        # Suggest prefetch_factor for faster loading
        if hasattr(dataloader, 'prefetch_factor'):
            logger.info("Consider increasing prefetch_factor for faster data loading")

def timing_decorator(optimizer: PerformanceOptimizer, operation_name: str):
    """Decorator for automatic timing of functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with optimizer.timing_context(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def memory_profiling_decorator(optimizer: PerformanceOptimizer, operation_name: str):
    """Decorator for automatic memory profiling of functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with optimizer.memory_profiling(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class GradientAccumulator:
    """Helper for gradient accumulation to simulate larger batch sizes."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps
        
class MemoryManager:
    """Memory management utilities."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {}
            
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "cached_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "max_cached_gb": torch.cuda.max_memory_reserved() / 1024**3
        }
        
    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage."""
        # Clear cache
        MemoryManager.clear_cache()
        
        # Set memory fraction (if needed)
        if torch.cuda.is_available():
            # Optionally set memory fraction to prevent OOM
            # torch.cuda.set_per_process_memory_fraction(0.8)
            pass
            
class ModelCompiler:
    """Model compilation utilities for performance."""
    
    @staticmethod
    def compile_model(model: nn.Module, **compile_kwargs) -> nn.Module:
        """Compile model for better performance."""
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, skipping compilation")
            return model
            
        try:
            # Default compilation options
            default_options = {
                "mode": "default",  # or "reduce-overhead", "max-autotune"
                "fullgraph": False,
                "dynamic": True
            }
            default_options.update(compile_kwargs)
            
            compiled_model = torch.compile(model, **default_options)
            logger.info(f"Model compiled with options: {default_options}")
            return compiled_model
            
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            return model
            
class PerformanceProfiler:
    """Comprehensive performance profiler."""
    
    def __init__(self):
        self.profiles = {}
        
    @contextmanager
    def profile(self, operation_name: str, enable_trace: bool = False):
        """Profile an operation."""
        if enable_trace and torch.cuda.is_available():
            # Use CUDA profiler for detailed analysis
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, 
                           torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                start_time = time.time()
                yield
                end_time = time.time()
                
            self.profiles[operation_name] = {
                "duration": end_time - start_time,
                "profiler_trace": prof.key_averages().table(sort_by="cuda_time_total"),
                "memory_profile": prof.key_averages(group_by_input_shape=True).table()
            }
        else:
            # Simple timing
            start_time = time.time()
            yield
            end_time = time.time()
            
            self.profiles[operation_name] = {
                "duration": end_time - start_time
            }
            
    def get_profile_summary(self) -> str:
        """Get summary of all profiles."""
        summary = "\n=== Performance Profile Summary ===\n"
        
        for op_name, profile_data in self.profiles.items():
            summary += f"\n{op_name}:\n"
            summary += f"  Duration: {profile_data['duration']:.4f}s\n"
            
            if "profiler_trace" in profile_data:
                summary += "  Detailed trace:\n"
                summary += profile_data["profiler_trace"]
                summary += "\n"
                
        return summary

# Convenience functions
def optimize_for_training(model: nn.Module, 
                         mixed_precision: bool = True,
                         gradient_clipping: float = 1.0) -> PerformanceOptimizer:
    """Quick setup for training optimization."""
    return PerformanceOptimizer(
        mixed_precision=mixed_precision,
        gradient_clipping=gradient_clipping,
        memory_efficient=True,
        profile_memory=False
    )
    
def optimize_for_inference(model: nn.Module, 
                          compile_model: bool = True) -> nn.Module:
    """Quick setup for inference optimization."""
    optimizer = PerformanceOptimizer(
        mixed_precision=False,  # Usually disabled for inference
        memory_efficient=True
    )
    
    optimized_model = optimizer.optimize_model_for_inference(model)
    
    if compile_model:
        optimized_model = ModelCompiler.compile_model(optimized_model)
        
    return optimized_model

def benchmark_model(model: nn.Module, 
                   input_tensors: List[torch.Tensor],
                   num_iterations: int = 100,
                   warmup_iterations: int = 10) -> Dict[str, float]:
    """Benchmark model performance."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(*input_tensors)
            
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(*input_tensors)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "total_time": total_time,
        "avg_time_per_iteration": total_time / num_iterations,
        "iterations_per_second": num_iterations / total_time,
        "num_iterations": num_iterations
    }
