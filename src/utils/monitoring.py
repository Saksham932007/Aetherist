#!/usr/bin/env python3
"""Advanced monitoring and observability for Aetherist.

Provides distributed tracing, custom metrics, alerting, and system monitoring.
"""

import os
import time
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import queue
from pathlib import Path

try:
    import torch
    import psutil
    import numpy as np
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    status: str = "ok"  # ok, error, timeout
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []
            
    def finish(self, status: str = "ok"):
        """Finish the span."""
        self.end_time = time.time()
        self.status = status
        
    def add_tag(self, key: str, value: Any):
        """Add tag to span."""
        self.tags[key] = value
        
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add log entry to span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
        
    def duration(self) -> Optional[float]:
        """Get span duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
        
class Tracer:
    """Distributed tracer for request tracing."""
    
    def __init__(self, service_name: str = "aetherist"):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: List[Span] = []
        self.current_span: Optional[Span] = None
        self.lock = threading.Lock()
        
    def start_span(self, operation_name: str, 
                   parent_span: Optional[Span] = None) -> Span:
        """Start a new tracing span."""
        with self.lock:
            # Generate IDs
            if parent_span:
                trace_id = parent_span.trace_id
                parent_span_id = parent_span.span_id
            else:
                trace_id = str(uuid.uuid4())
                parent_span_id = None
                
            span_id = str(uuid.uuid4())
            
            # Create span
            span = Span(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=time.time()
            )
            
            # Add service tag
            span.add_tag("service.name", self.service_name)
            
            # Track active span
            self.active_spans[span_id] = span
            
            return span
            
    def finish_span(self, span: Span, status: str = "ok"):
        """Finish a span."""
        with self.lock:
            span.finish(status)
            
            # Move from active to completed
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            self.completed_spans.append(span)
            
    @contextmanager
    def trace(self, operation_name: str, 
              parent_span: Optional[Span] = None):
        """Context manager for tracing."""
        span = self.start_span(operation_name, parent_span)
        old_current = self.current_span
        self.current_span = span
        
        try:
            yield span
            self.finish_span(span, "ok")
        except Exception as e:
            span.add_log(f"Error: {str(e)}", "error")
            self.finish_span(span, "error")
            raise
        finally:
            self.current_span = old_current
            
    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self.current_span
        
    def get_trace_data(self) -> List[Dict[str, Any]]:
        """Get all trace data."""
        with self.lock:
            return [asdict(span) for span in self.completed_spans]
            
    def clear_traces(self):
        """Clear completed traces."""
        with self.lock:
            self.completed_spans.clear()
            
def trace_operation(tracer: Tracer, operation_name: str = None):
    """Decorator for automatic operation tracing."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            parent_span = tracer.get_current_span()
            with tracer.trace(op_name, parent_span) as span:
                # Add function metadata
                span.add_tag("function.name", func.__name__)
                span.add_tag("function.module", func.__module__)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Add result metadata
                if hasattr(result, 'shape'):
                    span.add_tag("result.shape", str(result.shape))
                    
                return result
        return wrapper
    return decorator

class MetricsCollector:
    """Custom metrics collection and aggregation."""
    
    def __init__(self, export_interval: int = 60):
        self.export_interval = export_interval
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        self.lock = threading.Lock()
        self.last_export = time.time()
        
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment counter metric."""
        metric_name = self._format_metric_name(name, tags)
        with self.lock:
            self.counters[metric_name] += value
            
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set gauge metric."""
        metric_name = self._format_metric_name(name, tags)
        with self.lock:
            self.gauges[metric_name] = value
            
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record histogram value."""
        metric_name = self._format_metric_name(name, tags)
        with self.lock:
            self.histograms[metric_name].append(value)
            
    def record_timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record timing metric."""
        self.record_histogram(f"{name}.duration", duration, tags)
        
    def _format_metric_name(self, name: str, tags: Dict[str, str] = None) -> str:
        """Format metric name with tags."""
        if not tags:
            return name
            
        tag_string = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_string}"
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            summary = {
                "timestamp": time.time(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {}
            }
            
            # Compute histogram statistics
            for name, values in self.histograms.items():
                if values:
                    summary["histograms"][name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99),
                        "min": min(values),
                        "max": max(values)
                    }
                    
            return summary
            
    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            
class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: int = 30):
        """Start system monitoring."""
        if not MONITORING_AVAILABLE:
            logger.warning("System monitoring dependencies not available")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
        
    def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def _collect_system_metrics(self):
        """Collect CPU, memory, and disk metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("system.cpu.usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system.memory.usage_percent", memory.percent)
            self.metrics.set_gauge("system.memory.available_gb", memory.available / 1024**3)
            self.metrics.set_gauge("system.memory.used_gb", memory.used / 1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("system.disk.usage_percent", 
                                 (disk.used / disk.total) * 100)
            self.metrics.set_gauge("system.disk.free_gb", disk.free / 1024**3)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.metrics.set_gauge("system.network.bytes_sent", net_io.bytes_sent)
            self.metrics.set_gauge("system.network.bytes_recv", net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Memory metrics
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                    
                    tags = {"device": str(i)}
                    self.metrics.set_gauge("gpu.memory.allocated_gb", 
                                         memory_allocated, tags)
                    self.metrics.set_gauge("gpu.memory.cached_gb", 
                                         memory_cached, tags)
                    
                    # Utilization (if available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, 
                                                             pynvml.NVML_TEMPERATURE_GPU)
                        
                        self.metrics.set_gauge("gpu.utilization_percent", 
                                             util.gpu, tags)
                        self.metrics.set_gauge("gpu.memory_utilization_percent", 
                                             util.memory, tags)
                        self.metrics.set_gauge("gpu.temperature_celsius", temp, tags)
                        
                    except ImportError:
                        pass  # pynvml not available
                        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            
class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable] = []
        
    def add_alert_rule(self, name: str, metric_name: str, 
                      threshold: float, condition: str = ">",
                      duration: int = 60, severity: str = "warning"):
        """Add alert rule."""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "threshold": threshold,
            "condition": condition,
            "duration": duration,
            "severity": severity,
            "triggered_at": None,
            "last_check": None
        }
        self.alert_rules.append(rule)
        
    def add_notification_handler(self, handler: Callable):
        """Add notification handler."""
        self.notification_handlers.append(handler)
        
    def check_alerts(self):
        """Check all alert rules."""
        current_time = time.time()
        metrics_summary = self.metrics.get_metrics_summary()
        
        for rule in self.alert_rules:
            metric_value = self._get_metric_value(metrics_summary, rule["metric_name"])
            
            if metric_value is None:
                continue
                
            # Check condition
            triggered = self._evaluate_condition(
                metric_value, rule["threshold"], rule["condition"]
            )
            
            if triggered:
                if rule["triggered_at"] is None:
                    rule["triggered_at"] = current_time
                elif current_time - rule["triggered_at"] >= rule["duration"]:
                    # Alert should fire
                    self._fire_alert(rule, metric_value)
            else:
                # Reset trigger time
                rule["triggered_at"] = None
                
            rule["last_check"] = current_time
            
    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Get metric value from metrics summary."""
        # Check gauges
        if metric_name in metrics["gauges"]:
            return metrics["gauges"][metric_name]
            
        # Check counters
        if metric_name in metrics["counters"]:
            return float(metrics["counters"][metric_name])
            
        # Check histograms (use mean)
        if metric_name in metrics["histograms"]:
            return metrics["histograms"][metric_name]["mean"]
            
        return None
        
    def _evaluate_condition(self, value: float, threshold: float, condition: str) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        else:
            return False
            
    def _fire_alert(self, rule: Dict[str, Any], current_value: float):
        """Fire alert and send notifications."""
        alert = {
            "timestamp": time.time(),
            "rule_name": rule["name"],
            "metric_name": rule["metric_name"],
            "current_value": current_value,
            "threshold": rule["threshold"],
            "severity": rule["severity"],
            "message": f"Alert: {rule['name']} - {rule['metric_name']} = {current_value:.2f} {rule['condition']} {rule['threshold']}"
        }
        
        self.alert_history.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
                
class LogAggregator:
    """Log aggregation and analysis."""
    
    def __init__(self, log_file: str = "aetherist.log", 
                 max_lines: int = 10000):
        self.log_file = log_file
        self.max_lines = max_lines
        self.log_buffer = deque(maxlen=max_lines)
        self.error_patterns = [
            "ERROR", "Exception", "Traceback", "CRITICAL"
        ]
        
    def add_log_entry(self, level: str, message: str, **kwargs):
        """Add log entry to buffer."""
        entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.log_buffer.append(entry)
        
    def get_error_logs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get error logs from last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        error_logs = []
        for entry in self.log_buffer:
            if entry["timestamp"] >= cutoff_time:
                if any(pattern in entry["message"] for pattern in self.error_patterns):
                    error_logs.append(entry)
                    
        return error_logs
        
    def get_log_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        level_counts = defaultdict(int)
        total_logs = len(self.log_buffer)
        
        for entry in self.log_buffer:
            level_counts[entry["level"]] += 1
            
        return {
            "total_logs": total_logs,
            "level_counts": dict(level_counts),
            "error_rate": level_counts.get("ERROR", 0) / max(total_logs, 1)
        }
        
class MonitoringDashboard:
    """Monitoring dashboard and reporting."""
    
    def __init__(self, tracer: Tracer, metrics: MetricsCollector,
                 alerts: AlertManager, logs: LogAggregator):
        self.tracer = tracer
        self.metrics = metrics
        self.alerts = alerts
        self.logs = logs
        
    def generate_report(self, output_file: str = "monitoring_report.json"):
        """Generate comprehensive monitoring report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.get_metrics_summary(),
            "traces": self.tracer.get_trace_data()[-100:],  # Last 100 traces
            "alerts": self.alerts.alert_history[-50:],  # Last 50 alerts
            "logs": {
                "stats": self.logs.get_log_stats(),
                "errors": self.logs.get_error_logs(24)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        metrics_summary = self.metrics.get_metrics_summary()
        log_stats = self.logs.get_log_stats()
        
        # Determine health based on metrics
        health = "healthy"
        issues = []
        
        # Check error rate
        if log_stats["error_rate"] > 0.1:  # More than 10% errors
            health = "degraded"
            issues.append(f"High error rate: {log_stats['error_rate']:.2%}")
            
        # Check recent alerts
        recent_alerts = [a for a in self.alerts.alert_history 
                        if time.time() - a["timestamp"] < 3600]
        if recent_alerts:
            health = "degraded"
            issues.append(f"{len(recent_alerts)} alerts in last hour")
            
        # Check system resources
        gauges = metrics_summary.get("gauges", {})
        if gauges.get("system.memory.usage_percent", 0) > 90:
            health = "critical"
            issues.append("High memory usage")
            
        if gauges.get("system.cpu.usage_percent", 0) > 95:
            health = "critical" 
            issues.append("High CPU usage")
            
        return {
            "status": health,
            "issues": issues,
            "last_updated": datetime.now().isoformat()
        }

# Notification handlers
def console_notification_handler(alert: Dict[str, Any]):
    """Print alert to console."""
    print(f"[{alert['severity'].upper()}] {alert['message']}")
    
def file_notification_handler(alert: Dict[str, Any], file_path: str = "alerts.log"):
    """Write alert to file."""
    with open(file_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {alert['message']}\n")
        
# Factory functions
def create_monitoring_stack(service_name: str = "aetherist") -> Dict[str, Any]:
    """Create complete monitoring stack."""
    tracer = Tracer(service_name)
    metrics = MetricsCollector()
    alerts = AlertManager(metrics)
    logs = LogAggregator()
    
    # Add default alert rules
    alerts.add_alert_rule(
        "High Memory Usage",
        "system.memory.usage_percent",
        85.0,
        ">",
        duration=300,  # 5 minutes
        severity="warning"
    )
    
    alerts.add_alert_rule(
        "High Error Rate",
        "error_rate",
        0.1,
        ">",
        duration=60,
        severity="critical"
    )
    
    # Add notification handlers
    alerts.add_notification_handler(console_notification_handler)
    
    # Create dashboard
    dashboard = MonitoringDashboard(tracer, metrics, alerts, logs)
    
    # Start system monitoring
    system_monitor = SystemMonitor(metrics)
    system_monitor.start_monitoring()
    
    return {
        "tracer": tracer,
        "metrics": metrics,
        "alerts": alerts,
        "logs": logs,
        "dashboard": dashboard,
        "system_monitor": system_monitor
    }
