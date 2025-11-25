#!/usr/bin/env python3
"""Real-time monitoring dashboard for Aetherist.

Provides a web-based dashboard for monitoring system performance,
model metrics, and training progress.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from src.monitoring.system_monitor import SystemMonitor, TrainingMonitor
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dashboard dependencies not available: {e}")
    DASHBOARD_AVAILABLE = False

class MonitoringDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Aetherist Monitoring Dashboard")
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor(self.system_monitor)
        self.connected_clients = set()
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the main dashboard page."""
            return self._get_dashboard_html()
            
        @self.app.get("/api/system-status")
        async def get_system_status():
            """Get current system status."""
            return self.system_monitor.get_system_summary()
            
        @self.app.get("/api/performance-summary")
        async def get_performance_summary(window_minutes: int = 10):
            """Get performance summary."""
            return self.system_monitor.get_performance_summary(window_minutes)
            
        @self.app.get("/api/training-summary")
        async def get_training_summary():
            """Get training summary."""
            return self.training_monitor.get_training_summary()
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    data = {
                        "timestamp": time.time(),
                        "system_status": self.system_monitor.get_system_summary(),
                        "performance_summary": self.system_monitor.get_performance_summary(5),
                        "training_summary": self.training_monitor.get_training_summary()
                    }
                    
                    await websocket.send_text(json.dumps(data))
                    await asyncio.sleep(2)  # Update every 2 seconds
                    
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
                
        @self.app.post("/api/start-monitoring")
        async def start_monitoring():
            """Start system monitoring."""
            self.system_monitor.start_monitoring()
            return {"status": "monitoring_started"}
            
        @self.app.post("/api/stop-monitoring")
        async def stop_monitoring():
            """Stop system monitoring."""
            self.system_monitor.stop_monitoring()
            return {"status": "monitoring_stopped"}
            
        @self.app.post("/api/export-metrics")
        async def export_metrics(hours: int = 1):
            """Export metrics to file."""
            export_path = Path("outputs/monitoring") / f"metrics_{int(time.time())}.json"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.system_monitor.export_metrics(export_path, window_hours=hours)
            return {"status": "exported", "file": str(export_path)}
            
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Aetherist Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-label { font-weight: bold; }
        .metric-value { color: #007bff; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        .controls {
            margin: 20px 0;
            text-align: center;
        }
        button {
            padding: 10px 20px;
            margin: 0 10px;
            border: none;
            border-radius: 5px;
            background: #007bff;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { background: #0056b3; }
        button.danger { background: #dc3545; }
        button.danger:hover { background: #c82333; }
        .log { 
            height: 200px; 
            overflow-y: scroll; 
            background: #f8f9fa; 
            padding: 10px; 
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Aetherist Monitoring Dashboard</h1>
        <p>Real-time system and model performance monitoring</p>
    </div>
    
    <div class="controls">
        <button onclick="startMonitoring()">Start Monitoring</button>
        <button onclick="stopMonitoring()" class="danger">Stop Monitoring</button>
        <button onclick="exportMetrics()">Export Metrics</button>
        <button onclick="clearCharts()">Clear Charts</button>
    </div>
    
    <div class="dashboard-grid">
        <!-- System Status Card -->
        <div class="card">
            <h3>💻 System Status</h3>
            <div id="system-status">
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span id="system-health" class="metric-value">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CPU Usage:</span>
                    <span id="cpu-usage" class="metric-value">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage:</span>
                    <span id="memory-usage" class="metric-value">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">GPU Memory:</span>
                    <span id="gpu-memory" class="metric-value">N/A</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime:</span>
                    <span id="uptime" class="metric-value">0s</span>
                </div>
            </div>
        </div>
        
        <!-- Performance Metrics Card -->
        <div class="card">
            <h3>⚡ Performance Metrics</h3>
            <div id="performance-metrics">
                <div class="metric">
                    <span class="metric-label">Avg Inference Time:</span>
                    <span id="avg-inference-time" class="metric-value">N/A</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Throughput:</span>
                    <span id="throughput" class="metric-value">N/A</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Inferences:</span>
                    <span id="total-inferences" class="metric-value">0</span>
                </div>
            </div>
        </div>
        
        <!-- Training Status Card -->
        <div class="card">
            <h3>🎯 Training Status</h3>
            <div id="training-status">
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span id="training-health" class="metric-value">Not Started</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Epoch:</span>
                    <span id="current-epoch" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Step:</span>
                    <span id="current-step" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Elapsed Time:</span>
                    <span id="elapsed-time" class="metric-value">0s</span>
                </div>
            </div>
        </div>
        
        <!-- Real-time Charts -->
        <div class="card" style="grid-column: span 2;">
            <h3>📊 Real-time Performance</h3>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>
        
        <!-- Activity Log -->
        <div class="card" style="grid-column: span 2;">
            <h3>📝 Activity Log</h3>
            <div id="activity-log" class="log">
                <div>Dashboard initialized...</div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws = null;
        let chart = null;
        let chartData = {
            labels: [],
            datasets: [
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                },
                {
                    label: 'Memory %',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                },
                {
                    label: 'Throughput (samples/s)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    yAxisID: 'y1',
                }
            ]
        };
        
        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            max: 100,
                            title: {
                                display: true,
                                text: 'Usage %'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Throughput'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    }
                }
            });
        }
        
        // Connect to WebSocket
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                addLog('WebSocket connection closed. Attempting to reconnect...');
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                addLog('WebSocket error: ' + error);
            };
        }
        
        // Update dashboard with new data
        function updateDashboard(data) {
            // Update system status
            const systemStatus = data.system_status || {};
            document.getElementById('system-health').textContent = systemStatus.status || 'Unknown';
            document.getElementById('system-health').className = 
                'metric-value ' + (systemStatus.status === 'healthy' ? 'status-good' : 'status-warning');
            
            document.getElementById('cpu-usage').textContent = 
                (systemStatus.cpu_percent || 0).toFixed(1) + '%';
            document.getElementById('memory-usage').textContent = 
                (systemStatus.memory_percent || 0).toFixed(1) + '%';
            document.getElementById('uptime').textContent = 
                formatTime(systemStatus.uptime || 0);
            
            // Update GPU info if available
            if (systemStatus.gpu_metrics && systemStatus.gpu_metrics.torch_info) {
                const gpuMemory = systemStatus.gpu_metrics.torch_info.torch_allocated || 0;
                document.getElementById('gpu-memory').textContent = gpuMemory.toFixed(2) + ' GB';
            }
            
            // Update performance metrics
            const perfSummary = data.performance_summary || {};
            const modelMetrics = perfSummary.model_metrics || {};
            
            if (modelMetrics.avg_inference_time) {
                document.getElementById('avg-inference-time').textContent = 
                    (modelMetrics.avg_inference_time * 1000).toFixed(1) + ' ms';
            }
            if (modelMetrics.avg_throughput) {
                document.getElementById('throughput').textContent = 
                    modelMetrics.avg_throughput.toFixed(1) + ' samples/s';
            }
            if (modelMetrics.total_inferences) {
                document.getElementById('total-inferences').textContent = 
                    modelMetrics.total_inferences;
            }
            
            // Update training status
            const trainingSummary = data.training_summary || {};
            document.getElementById('training-health').textContent = trainingSummary.status || 'Not Started';
            document.getElementById('current-epoch').textContent = trainingSummary.current_epoch || 0;
            document.getElementById('current-step').textContent = trainingSummary.current_step || 0;
            document.getElementById('elapsed-time').textContent = 
                formatTime(trainingSummary.elapsed_time || 0);
            
            // Update chart
            updateChart(systemStatus, modelMetrics);
        }
        
        // Update performance chart
        function updateChart(systemStatus, modelMetrics) {
            const now = new Date().toLocaleTimeString();
            const maxPoints = 50;
            
            // Add new data point
            chartData.labels.push(now);
            chartData.datasets[0].data.push(systemStatus.cpu_percent || 0);
            chartData.datasets[1].data.push(systemStatus.memory_percent || 0);
            chartData.datasets[2].data.push(modelMetrics.avg_throughput || 0);
            
            // Remove old data points
            if (chartData.labels.length > maxPoints) {
                chartData.labels.shift();
                chartData.datasets.forEach(dataset => dataset.data.shift());
            }
            
            if (chart) {
                chart.update('none');
            }
        }
        
        // Utility functions
        function formatTime(seconds) {
            if (seconds < 60) return seconds.toFixed(0) + 's';
            if (seconds < 3600) return (seconds / 60).toFixed(1) + 'm';
            return (seconds / 3600).toFixed(1) + 'h';
        }
        
        function addLog(message) {
            const log = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.textContent = `[${timestamp}] ${message}`;
            log.appendChild(logEntry);
            log.scrollTop = log.scrollHeight;
            
            // Keep only last 100 log entries
            while (log.children.length > 100) {
                log.removeChild(log.firstChild);
            }
        }
        
        // Control functions
        async function startMonitoring() {
            try {
                const response = await fetch('/api/start-monitoring', {method: 'POST'});
                const result = await response.json();
                addLog('Monitoring started');
            } catch (error) {
                addLog('Failed to start monitoring: ' + error);
            }
        }
        
        async function stopMonitoring() {
            try {
                const response = await fetch('/api/stop-monitoring', {method: 'POST'});
                const result = await response.json();
                addLog('Monitoring stopped');
            } catch (error) {
                addLog('Failed to stop monitoring: ' + error);
            }
        }
        
        async function exportMetrics() {
            try {
                const response = await fetch('/api/export-metrics', {method: 'POST'});
                const result = await response.json();
                addLog('Metrics exported to: ' + result.file);
            } catch (error) {
                addLog('Failed to export metrics: ' + error);
            }
        }
        
        function clearCharts() {
            chartData.labels = [];
            chartData.datasets.forEach(dataset => dataset.data = []);
            if (chart) chart.update();
            addLog('Charts cleared');
        }
        
        // Initialize dashboard
        window.onload = function() {
            initChart();
            connectWebSocket();
            addLog('Dashboard initialized');
        };
    </script>
</body>
</html>
        """
        
    def run(self):
        """Run the monitoring dashboard."""
        self.system_monitor.start_monitoring()
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        finally:
            self.system_monitor.stop_monitoring()

def main():
    parser = argparse.ArgumentParser(description="Run Aetherist monitoring dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind the dashboard to")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port to bind the dashboard to")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    if not DASHBOARD_AVAILABLE:
        logger.error("Dashboard dependencies not available. Install with: pip install fastapi uvicorn")
        sys.exit(1)
        
    try:
        logger.info(f"Starting monitoring dashboard on http://{args.host}:{args.port}")
        dashboard = MonitoringDashboard(args.host, args.port)
        dashboard.run()
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
