#!/usr/bin/env python3
"""Comprehensive test script for Aetherist utilities.

Tests all monitoring, batch processing, and analysis utilities.
"""

import argparse
import logging
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_system_monitor():
    """Test system monitoring functionality."""
    print("\n=== Testing System Monitor ===")
    
    try:
        from src.monitoring.system_monitor import SystemMonitor, SystemMetrics
        
        # Test monitor initialization
        monitor = SystemMonitor(collection_interval=0.1, max_history=10)
        print("✓ System monitor initialization successful")
        
        # Test metric collection without starting background monitoring
        metrics = monitor._collect_system_metrics()
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        print("✓ System metrics collection successful")
        
        # Test summary generation
        summary = monitor.get_system_summary()
        assert "status" in summary
        print("✓ System summary generation successful")
        
        # Test short monitoring session
        monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect a few samples
        monitor.stop_monitoring()
        
        performance_summary = monitor.get_performance_summary(1)
        assert "window_minutes" in performance_summary
        print("✓ Short monitoring session successful")
        
        # Test metric export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = Path(f.name)
            
        monitor.export_metrics(export_path)
        assert export_path.exists()
        export_path.unlink()  # Clean up
        print("✓ Metric export successful")
        
    except Exception as e:
        print(f"✗ System monitor test failed: {e}")
        return False
        
    return True

def test_model_analyzer():
    """Test model analysis functionality."""
    print("\n=== Testing Model Analyzer ===")
    
    try:
        from src.monitoring.model_analyzer import ModelAnalyzer
        
        # Check if torch is available
        try:
            import torch
            import torch.nn as nn
            torch_available = True
        except ImportError:
            print("⚠️ PyTorch not available, skipping model analyzer tests")
            return True
            
        # Create analyzer
        analyzer = ModelAnalyzer(device="cpu")
        print("✓ Model analyzer initialization successful")
        
        # Create dummy model for testing
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 50)
                self.linear2 = nn.Linear(50, 1)
                
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                return self.linear2(x)
                
        dummy_model = DummyModel()
        
        # Test architecture analysis
        arch_analysis = analyzer.analyze_architecture(dummy_model, "DummyModel")
        assert arch_analysis.total_parameters > 0
        assert arch_analysis.model_size_mb > 0
        print("✓ Architecture analysis successful")
        
        # Test performance benchmarking (very small scale)
        perf_results = analyzer.benchmark_inference(
            dummy_model, (10,), "DummyModel",
            num_warmup=2, num_iterations=5, batch_sizes=[1, 2]
        )
        assert len(perf_results) > 0
        print("✓ Performance benchmarking successful")
        
        # Test generation quality analysis
        dummy_images = torch.randn(4, 3, 64, 64)
        quality_metrics = analyzer.analyze_generation_quality(dummy_images)
        assert quality_metrics.batch_size == 4
        assert quality_metrics.resolution == (64, 64)
        print("✓ Generation quality analysis successful")
        
    except Exception as e:
        print(f"✗ Model analyzer test failed: {e}")
        return False
        
    return True

def test_batch_processor():
    """Test batch processing functionality."""
    print("\n=== Testing Batch Processor ===")
    
    try:
        from src.batch.batch_processor import (
            BatchProcessor, BatchConfig, dummy_processor, 
            tensor_computation_processor
        )
        
        # Test batch processor initialization
        config = BatchConfig(
            max_workers=2,
            batch_size=4,
            max_queue_size=10,
            timeout_seconds=5
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            processor = BatchProcessor(config)
            print("✓ Batch processor initialization successful")
            
            # Register test processors
            processor.register_processor("dummy", dummy_processor)
            
            # Test PyTorch tensor processor if available
            try:
                import torch
                processor.register_processor("tensor_compute", tensor_computation_processor)
                torch_available = True
            except ImportError:
                torch_available = False
                
            print("✓ Processor registration successful")
            
            # Test job processing
            with processor:
                # Add dummy jobs
                job_ids = []
                for i in range(5):
                    job_id = processor.add_job(
                        "dummy", 
                        f"test_data_{i}", 
                        {"sleep_time": 0.1}
                    )
                    job_ids.append(job_id)
                    
                print(f"✓ Added {len(job_ids)} test jobs")
                
                # Add tensor computation jobs if PyTorch available
                if torch_available:
                    tensor_jobs = []
                    for i in range(3):
                        test_tensor = torch.randn(10, 10)
                        job_id = processor.add_job(
                            "tensor_compute",
                            test_tensor,
                            {"operation": "mean"}
                        )
                        tensor_jobs.append(job_id)
                        
                    job_ids.extend(tensor_jobs)
                    print(f"✓ Added {len(tensor_jobs)} tensor computation jobs")
                    
                # Wait for completion
                success = processor.wait_for_completion(timeout=10.0)
                assert success, "Jobs did not complete in time"
                
                # Check results
                summary = processor.get_processing_summary()
                print(f"✓ Processing completed: {summary['total_completed']} jobs")
                assert summary["total_completed"] > 0
                
                # Test job status retrieval
                for job_id in job_ids[:2]:  # Check first 2 jobs
                    status = processor.get_job_status(job_id)
                    assert status is not None
                    assert status["status"] == "completed"
                    
                print("✓ Job status retrieval successful")
                
    except Exception as e:
        print(f"✗ Batch processor test failed: {e}")
        return False
        
    return True

def test_monitoring_integration():
    """Test integration between monitoring components."""
    print("\n=== Testing Monitoring Integration ===")
    
    try:
        from src.monitoring.system_monitor import SystemMonitor, TrainingMonitor
        
        # Test training monitor integration
        system_monitor = SystemMonitor(collection_interval=0.1)
        training_monitor = TrainingMonitor(system_monitor)
        
        # Start monitoring
        training_monitor.start_training_monitoring()
        print("✓ Training monitoring started")
        
        # Simulate some training steps
        for epoch in range(2):
            for step in range(3):
                training_monitor.log_training_step(
                    epoch=epoch,
                    step=step,
                    generator_loss=0.5 + 0.1 * step,
                    discriminator_loss=0.3 + 0.05 * step,
                    learning_rate=0.001,
                    batch_time=0.1,
                    data_time=0.02
                )
                time.sleep(0.05)  # Small delay
                
        # Get training summary
        summary = training_monitor.get_training_summary()
        assert summary["current_epoch"] == 1
        assert summary["current_step"] == 2
        print("✓ Training step logging successful")
        
        # Stop monitoring
        training_monitor.stop_training_monitoring()
        print("✓ Training monitoring stopped")
        
    except Exception as e:
        print(f"✗ Monitoring integration test failed: {e}")
        return False
        
    return True

def test_configuration_loading():
    """Test configuration loading for utilities."""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        # Test deployment configuration loading
        config_path = project_root / "configs" / "deployment_config.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            assert "environment" in config_data
            assert "server_config" in config_data
            print("✓ Deployment configuration loading successful")
            
        # Test server configuration loading
        server_config_path = project_root / "configs" / "server_config.json"
        
        if server_config_path.exists():
            import json
            with open(server_config_path, 'r') as f:
                server_data = json.load(f)
                
            assert "host" in server_data
            assert "port" in server_data
            print("✓ Server configuration loading successful")
            
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False
        
    return True

def test_utility_imports():
    """Test that all utility modules can be imported."""
    print("\n=== Testing Utility Imports ===")
    
    try:
        # Test monitoring imports
        from src.monitoring import (
            SystemMonitor, TrainingMonitor, ModelAnalyzer
        )
        print("✓ Monitoring module imports successful")
        
        # Test batch processing imports
        from src.batch import (
            BatchProcessor, BatchConfig, BatchJob
        )
        print("✓ Batch processing module imports successful")
        
        # Test deployment imports
        from src.deployment import (
            ModelOptimizer, AetheristModelServer, DeploymentManager
        )
        print("✓ Deployment module imports successful")
        
    except Exception as e:
        print(f"✗ Utility imports test failed: {e}")
        return False
        
    return True

def test_script_availability():
    """Test that all utility scripts are available."""
    print("\n=== Testing Script Availability ===")
    
    scripts_to_check = [
        "analyze_model.py",
        "batch_generate.py", 
        "monitoring_dashboard.py",
        "deploy_optimize.py",
        "deploy_server.py",
        "deploy_full.py"
    ]
    
    scripts_dir = project_root / "scripts"
    
    for script_name in scripts_to_check:
        script_path = scripts_dir / script_name
        if script_path.exists():
            print(f"✓ {script_name} available")
        else:
            print(f"✗ {script_name} missing")
            return False
            
    return True

def main():
    """Run all utility tests."""
    parser = argparse.ArgumentParser(description="Test Aetherist utilities")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    parser.add_argument("--skip-slow", action="store_true",
                       help="Skip slow tests")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Starting Aetherist Utilities Test Suite")
    print("=" * 50)
    
    tests = [
        test_utility_imports,
        test_script_availability,
        test_configuration_loading,
        test_system_monitor,
        test_model_analyzer,
        test_batch_processor,
        test_monitoring_integration
    ]
    
    if args.skip_slow:
        print("⚠️ Skipping slow tests (model analysis, batch processing)")
        tests = [test_utility_imports, test_script_availability, test_configuration_loading]
        
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
            
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All utility tests passed!")
        print("\nThe advanced system utilities are ready for use.")
        print("\nAvailable features:")
        print("• Real-time system and performance monitoring")
        print("• Comprehensive model analysis and benchmarking")
        print("• High-performance batch processing")
        print("• Production deployment automation")
        print("• Web-based monitoring dashboard")
        print("• Advanced utility scripts for analysis and generation")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
