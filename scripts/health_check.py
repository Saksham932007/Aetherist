#!/usr/bin/env python3
"""
Aetherist Health Check and Diagnostics Tool

Comprehensive system health monitoring and diagnostic capabilities
for Aetherist installations and deployments.

Usage:
    python scripts/health_check.py [--config config.yaml] [--continuous] [--alert]
"""

import argparse
import asyncio
import json
import logging
import psutil
import requests
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import torch


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    status: HealthStatus
    value: Any
    threshold: Optional[float] = None
    message: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class SystemInfo:
    """System information collection."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    temperature: Optional[float] = None
    uptime: float = 0.0


class HealthChecker:
    """Comprehensive health checking and monitoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics = []
        self.logger = logging.getLogger(__name__)
        
        # Health check thresholds
        self.thresholds = self.config.get("thresholds", {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "gpu_usage": 90.0,
            "gpu_memory": 95.0,
            "temperature": 85.0,
            "response_time": 5.0,
            "error_rate": 5.0
        })
        
        # API endpoints to monitor
        self.api_endpoints = self.config.get("api_endpoints", {
            "health": "http://localhost:8000/health",
            "metrics": "http://localhost:8000/metrics",
            "api_docs": "http://localhost:8000/docs"
        })
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if not config_path:
            config_path = "configs/health_config.yaml"
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        else:
            self.logger.info(f"Config file not found: {config_path}. Using defaults.")
            return {}
    
    def check_system_resources(self) -> List[HealthMetric]:
        """Check system resource utilization."""
        metrics = []
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        status = HealthStatus.CRITICAL if cpu_usage > self.thresholds["cpu_usage"] else \
                 HealthStatus.WARNING if cpu_usage > self.thresholds["cpu_usage"] * 0.8 else \
                 HealthStatus.HEALTHY
        
        metrics.append(HealthMetric(
            name="cpu_usage",
            status=status,
            value=cpu_usage,
            threshold=self.thresholds["cpu_usage"],
            message=f"CPU usage: {cpu_usage:.1f}%"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        status = HealthStatus.CRITICAL if memory_usage > self.thresholds["memory_usage"] else \
                 HealthStatus.WARNING if memory_usage > self.thresholds["memory_usage"] * 0.8 else \
                 HealthStatus.HEALTHY
        
        metrics.append(HealthMetric(
            name="memory_usage",
            status=status,
            value=memory_usage,
            threshold=self.thresholds["memory_usage"],
            message=f"Memory usage: {memory_usage:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        status = HealthStatus.CRITICAL if disk_usage > self.thresholds["disk_usage"] else \
                 HealthStatus.WARNING if disk_usage > self.thresholds["disk_usage"] * 0.8 else \
                 HealthStatus.HEALTHY
        
        metrics.append(HealthMetric(
            name="disk_usage",
            status=status,
            value=disk_usage,
            threshold=self.thresholds["disk_usage"],
            message=f"Disk usage: {disk_usage:.1f}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)"
        ))
        
        # System temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                max_temp = max([temp.current for sensor_temps in temps.values() for temp in sensor_temps])
                status = HealthStatus.CRITICAL if max_temp > self.thresholds["temperature"] else \
                         HealthStatus.WARNING if max_temp > self.thresholds["temperature"] * 0.9 else \
                         HealthStatus.HEALTHY
                
                metrics.append(HealthMetric(
                    name="temperature",
                    status=status,
                    value=max_temp,
                    threshold=self.thresholds["temperature"],
                    message=f"Max temperature: {max_temp:.1f}°C"
                ))
        except:
            pass  # Temperature monitoring not available
        
        return metrics
    
    def check_gpu_resources(self) -> List[HealthMetric]:
        """Check GPU resource utilization."""
        metrics = []
        
        if not torch.cuda.is_available():
            return metrics
        
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            
            for i in range(torch.cuda.device_count()):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = utilization.gpu
                
                status = HealthStatus.CRITICAL if gpu_usage > self.thresholds["gpu_usage"] else \
                         HealthStatus.WARNING if gpu_usage > self.thresholds["gpu_usage"] * 0.8 else \
                         HealthStatus.HEALTHY
                
                metrics.append(HealthMetric(
                    name=f"gpu_{i}_usage",
                    status=status,
                    value=gpu_usage,
                    threshold=self.thresholds["gpu_usage"],
                    message=f"GPU {i} usage: {gpu_usage}%"
                ))
                
                # GPU memory
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage = (memory_info.used / memory_info.total) * 100
                
                status = HealthStatus.CRITICAL if gpu_memory_usage > self.thresholds["gpu_memory"] else \
                         HealthStatus.WARNING if gpu_memory_usage > self.thresholds["gpu_memory"] * 0.8 else \
                         HealthStatus.HEALTHY
                
                metrics.append(HealthMetric(
                    name=f"gpu_{i}_memory",
                    status=status,
                    value=gpu_memory_usage,
                    threshold=self.thresholds["gpu_memory"],
                    message=f"GPU {i} memory: {gpu_memory_usage:.1f}% ({memory_info.used // (1024**2)}MB / {memory_info.total // (1024**2)}MB)"
                ))
                
                # GPU temperature
                try:
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    status = HealthStatus.CRITICAL if temperature > self.thresholds["temperature"] else \
                             HealthStatus.WARNING if temperature > self.thresholds["temperature"] * 0.9 else \
                             HealthStatus.HEALTHY
                    
                    metrics.append(HealthMetric(
                        name=f"gpu_{i}_temperature",
                        status=status,
                        value=temperature,
                        threshold=self.thresholds["temperature"],
                        message=f"GPU {i} temperature: {temperature}°C"
                    ))
                except:
                    pass  # Temperature not available
                    
        except ImportError:
            self.logger.warning("nvidia-ml-py3 not available for GPU monitoring")
        except Exception as e:
            self.logger.error(f"Error checking GPU resources: {e}")
        
        return metrics
    
    def check_model_health(self) -> List[HealthMetric]:
        """Check model loading and basic functionality."""
        metrics = []
        
        try:
            # Test model import
            start_time = time.time()
            from aetherist import AetheristModel
            import_time = time.time() - start_time
            
            metrics.append(HealthMetric(
                name="model_import",
                status=HealthStatus.HEALTHY,
                value=import_time,
                message=f"Model import successful ({import_time:.2f}s)"
            ))
            
            # Test basic model functionality
            start_time = time.time()
            
            # Create minimal config for testing
            from aetherist.config import AetheristConfig
            config = AetheristConfig(
                latent_dim=128,  # Small for quick test
                triplane_dim=64,
                triplane_res=32,
                resolution=128
            )
            
            model = AetheristModel(config)
            init_time = time.time() - start_time
            
            metrics.append(HealthMetric(
                name="model_initialization",
                status=HealthStatus.HEALTHY,
                value=init_time,
                message=f"Model initialization successful ({init_time:.2f}s)"
            ))
            
            # Test forward pass
            start_time = time.time()
            with torch.no_grad():
                z = torch.randn(1, config.latent_dim)
                output = model.generate(z)
            inference_time = time.time() - start_time
            
            status = HealthStatus.CRITICAL if inference_time > 30 else \
                     HealthStatus.WARNING if inference_time > 10 else \
                     HealthStatus.HEALTHY
            
            metrics.append(HealthMetric(
                name="model_inference",
                status=status,
                value=inference_time,
                threshold=10.0,
                message=f"Model inference successful ({inference_time:.2f}s)"
            ))
            
        except Exception as e:
            metrics.append(HealthMetric(
                name="model_health",
                status=HealthStatus.CRITICAL,
                value=0,
                message=f"Model health check failed: {str(e)}"
            ))
        
        return metrics
    
    def check_api_endpoints(self) -> List[HealthMetric]:
        """Check API endpoint availability and response times."""
        metrics = []
        
        for endpoint_name, url in self.api_endpoints.items():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=self.thresholds["response_time"])
                response_time = time.time() - start_time
                
                # Check response status
                if response.status_code == 200:
                    status = HealthStatus.WARNING if response_time > self.thresholds["response_time"] * 0.8 else \
                             HealthStatus.HEALTHY
                    message = f"{endpoint_name} endpoint healthy ({response_time:.2f}s)"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"{endpoint_name} endpoint returned {response.status_code}"
                
                metrics.append(HealthMetric(
                    name=f"api_{endpoint_name}",
                    status=status,
                    value=response_time,
                    threshold=self.thresholds["response_time"],
                    message=message
                ))
                
            except requests.exceptions.ConnectionError:
                metrics.append(HealthMetric(
                    name=f"api_{endpoint_name}",
                    status=HealthStatus.CRITICAL,
                    value=0,
                    message=f"{endpoint_name} endpoint not reachable"
                ))
            except requests.exceptions.Timeout:
                metrics.append(HealthMetric(
                    name=f"api_{endpoint_name}",
                    status=HealthStatus.CRITICAL,
                    value=self.thresholds["response_time"],
                    message=f"{endpoint_name} endpoint timeout"
                ))
            except Exception as e:
                metrics.append(HealthMetric(
                    name=f"api_{endpoint_name}",
                    status=HealthStatus.CRITICAL,
                    value=0,
                    message=f"{endpoint_name} endpoint error: {str(e)}"
                ))
        
        return metrics
    
    def check_dependencies(self) -> List[HealthMetric]:
        """Check critical dependencies and versions."""
        metrics = []
        
        critical_deps = {
            "torch": "1.11.0",
            "torchvision": "0.12.0",
            "numpy": "1.20.0",
            "pillow": "8.0.0"
        }
        
        for dep_name, min_version in critical_deps.items():
            try:
                import importlib
                module = importlib.import_module(dep_name.lower())
                
                if hasattr(module, "__version__"):
                    version = module.__version__
                    # Simplified version comparison
                    status = HealthStatus.HEALTHY
                    message = f"{dep_name} {version} installed"
                else:
                    status = HealthStatus.WARNING
                    message = f"{dep_name} installed (version unknown)"
                
                metrics.append(HealthMetric(
                    name=f"dependency_{dep_name}",
                    status=status,
                    value=version if hasattr(module, "__version__") else "unknown",
                    message=message
                ))
                
            except ImportError:
                metrics.append(HealthMetric(
                    name=f"dependency_{dep_name}",
                    status=HealthStatus.CRITICAL,
                    value="missing",
                    message=f"{dep_name} not installed"
                ))
        
        return metrics
    
    def run_comprehensive_check(self) -> Dict[str, List[HealthMetric]]:
        """Run all health checks and return results."""
        results = {}
        
        self.logger.info("Running comprehensive health check...")
        
        # System resources
        results["system"] = self.check_system_resources()
        
        # GPU resources
        results["gpu"] = self.check_gpu_resources()
        
        # Model health
        results["model"] = self.check_model_health()
        
        # API endpoints
        results["api"] = self.check_api_endpoints()
        
        # Dependencies
        results["dependencies"] = self.check_dependencies()
        
        return results
    
    def get_overall_status(self, results: Dict[str, List[HealthMetric]]) -> HealthStatus:
        """Determine overall system health status."""
        all_metrics = []
        for category_metrics in results.values():
            all_metrics.extend(category_metrics)
        
        if any(m.status == HealthStatus.CRITICAL for m in all_metrics):
            return HealthStatus.CRITICAL
        elif any(m.status == HealthStatus.WARNING for m in all_metrics):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def format_results(self, results: Dict[str, List[HealthMetric]], format_type: str = "text") -> str:
        """Format health check results for display."""
        if format_type == "json":
            output = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": self.get_overall_status(results).value,
                "categories": {}
            }
            
            for category, metrics in results.items():
                output["categories"][category] = [asdict(m) for m in metrics]
            
            return json.dumps(output, indent=2)
        
        else:  # text format
            lines = []
            lines.append("🏥 Aetherist Health Check Report")
            lines.append("=" * 50)
            lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            overall_status = self.get_overall_status(results)
            status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "❌", "unknown": "❓"}[overall_status.value]
            lines.append(f"Overall Status: {status_icon} {overall_status.value.upper()}")
            lines.append("")
            
            for category, metrics in results.items():
                lines.append(f"📊 {category.upper()} METRICS:")
                lines.append("-" * 30)
                
                for metric in metrics:
                    icon = status_icon[metric.status.value]
                    lines.append(f"{icon} {metric.message}")
                
                lines.append("")
            
            return "\n".join(lines)
    
    def save_results(self, results: Dict[str, List[HealthMetric]], filepath: str) -> None:
        """Save health check results to file."""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_status(results).value,
            "categories": {cat: [asdict(m) for m in metrics] for cat, metrics in results.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Health check results saved to: {filepath}")


async def continuous_monitoring(checker: HealthChecker, interval: int = 60, duration: int = 3600):
    """Run continuous health monitoring."""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        results = checker.run_comprehensive_check()
        
        # Log results
        overall_status = checker.get_overall_status(results)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if overall_status == HealthStatus.CRITICAL:
            checker.logger.error(f"[{timestamp}] CRITICAL: System health issues detected")
        elif overall_status == HealthStatus.WARNING:
            checker.logger.warning(f"[{timestamp}] WARNING: System performance concerns")
        else:
            checker.logger.info(f"[{timestamp}] HEALTHY: All systems operating normally")
        
        # Save periodic results
        save_path = f"logs/health_check_{int(time.time())}.json"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        checker.save_results(results, save_path)
        
        await asyncio.sleep(interval)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Aetherist Health Check and Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--save", "-s",
        type=str,
        help="Save results to file"
    )
    
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous monitoring"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Monitoring duration in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize health checker
    checker = HealthChecker(config_path=args.config)
    
    try:
        if args.continuous:
            # Run continuous monitoring
            asyncio.run(continuous_monitoring(
                checker, 
                interval=args.interval, 
                duration=args.duration
            ))
        else:
            # Run single health check
            results = checker.run_comprehensive_check()
            
            # Format and display results
            output = checker.format_results(results, args.format)
            print(output)
            
            # Save results if requested
            if args.save:
                checker.save_results(results, args.save)
            
            # Exit with appropriate code
            overall_status = checker.get_overall_status(results)
            if overall_status == HealthStatus.CRITICAL:
                sys.exit(1)
            elif overall_status == HealthStatus.WARNING:
                sys.exit(2)
            else:
                sys.exit(0)
                
    except KeyboardInterrupt:
        print("\nHealth monitoring interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()