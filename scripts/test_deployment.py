#!/usr/bin/env python3
"""Test script for Aetherist deployment infrastructure."""

import asyncio
import json
import logging
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_optimizer():
    """Test model optimization functionality."""
    print("\n=== Testing Model Optimizer ===")
    
    try:
        from src.deployment.model_optimizer import ModelOptimizer, OptimizationConfig
        
        # Create test configuration
        config = OptimizationConfig(
            quantize=False,  # Disable quantization for testing
            export_onnx=False,  # Disable ONNX for testing
            export_torchscript=False,  # Disable TorchScript for testing
            optimize_for_inference=True
        )
        
        optimizer = ModelOptimizer(config)
        
        # Test optimization config creation
        assert optimizer.config.optimize_for_inference == True
        assert optimizer.config.batch_size == 1
        
        print("✓ Model optimizer initialization successful")
        
        # Test benchmark report generation (without actual models)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_report.json"
            
            test_optimization_results = {
                "optimized": Path(temp_dir) / "test_model.pth"
            }
            
            test_benchmark_results = {
                "optimized": {
                    "avg_inference_time": 0.1,
                    "throughput": 10.0,
                    "model_size_mb": 450.5
                }
            }
            
            optimizer.save_optimization_report(
                test_optimization_results,
                test_benchmark_results,
                temp_path
            )
            
            # Verify report was created
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                report = json.load(f)
                
            assert "optimization_config" in report
            assert "benchmark_results" in report
            assert "timestamp" in report
            
        print("✓ Optimization report generation successful")
        
    except Exception as e:
        print(f"✗ Model optimizer test failed: {e}")
        return False
        
    return True

def test_server_config():
    """Test server configuration and initialization."""
    print("\n=== Testing Server Configuration ===")
    
    try:
        from src.deployment.model_server import ServerConfig, AetheristModelServer
        
        # Test default configuration
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        
        print("✓ Default server configuration successful")
        
        # Test custom configuration
        custom_config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            workers=2,
            log_level="DEBUG"
        )
        
        assert custom_config.host == "127.0.0.1"
        assert custom_config.port == 9000
        assert custom_config.workers == 2
        assert custom_config.log_level == "DEBUG"
        
        print("✓ Custom server configuration successful")
        
        # Test server creation (without actually starting it)
        server = AetheristModelServer(config)
        assert server.config == config
        assert server.app is not None
        
        print("✓ Server initialization successful")
        
    except Exception as e:
        print(f"✗ Server configuration test failed: {e}")
        return False
        
    return True

def test_deployment_manager():
    """Test deployment manager functionality."""
    print("\n=== Testing Deployment Manager ===")
    
    try:
        from src.deployment.deployment_manager import (
            DeploymentManager, DeploymentConfig, DockerConfig, 
            KubernetesConfig, AWSConfig
        )
        
        # Test configuration creation
        config = DeploymentConfig(
            environment="development",
            deployment_type="local",
            optimize_models=False
        )
        
        assert config.environment == "development"
        assert config.deployment_type == "local"
        assert config.optimize_models == False
        
        print("✓ Deployment configuration creation successful")
        
        # Test deployment manager initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DeploymentManager(config, Path(temp_dir))
            
            assert manager.config == config
            assert manager.workspace_root == Path(temp_dir)
            assert manager.deployment_dir.exists()
            
        print("✓ Deployment manager initialization successful")
        
        # Test Docker configuration
        docker_config = DockerConfig(
            base_image="pytorch/pytorch:latest",
            gpu_support=False
        )
        
        assert docker_config.base_image == "pytorch/pytorch:latest"
        assert docker_config.gpu_support == False
        
        print("✓ Docker configuration successful")
        
        # Test Kubernetes configuration
        k8s_config = KubernetesConfig(
            namespace="test-namespace",
            replicas=3
        )
        
        assert k8s_config.namespace == "test-namespace"
        assert k8s_config.replicas == 3
        assert "cpu" in k8s_config.resource_requests
        
        print("✓ Kubernetes configuration successful")
        
        # Test AWS configuration
        aws_config = AWSConfig(
            region="us-east-1",
            instance_type="p3.2xlarge"
        )
        
        assert aws_config.region == "us-east-1"
        assert aws_config.instance_type == "p3.2xlarge"
        
        print("✓ AWS configuration successful")
        
    except Exception as e:
        print(f"✗ Deployment manager test failed: {e}")
        return False
        
    return True

def test_deployment_artifacts():
    """Test deployment artifact generation."""
    print("\n=== Testing Deployment Artifacts ===")
    
    try:
        from src.deployment.deployment_manager import DeploymentManager, DeploymentConfig
        
        config = DeploymentConfig(
            environment="development",
            deployment_type="docker",
            optimize_models=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal workspace structure
            workspace = Path(temp_dir)
            (workspace / "src").mkdir()
            (workspace / "configs").mkdir()
            (workspace / "requirements.txt").touch()
            
            manager = DeploymentManager(config, workspace)
            
            # Test Dockerfile generation
            dockerfile_content = manager._generate_dockerfile()
            assert "FROM" in dockerfile_content
            assert "WORKDIR" in dockerfile_content
            assert "EXPOSE" in dockerfile_content
            
            print("✓ Dockerfile generation successful")
            
            # Test docker-compose generation
            compose_content = manager._generate_docker_compose()
            assert "version:" in compose_content
            assert "services:" in compose_content
            assert "aetherist-api:" in compose_content
            
            print("✓ Docker-compose generation successful")
            
            # Test local run script generation
            run_script = manager._generate_local_run_script()
            assert "#!/bin/bash" in run_script
            assert "PYTHONPATH" in run_script
            assert "python -c" in run_script
            
            print("✓ Local run script generation successful")
            
            # Test Kubernetes manifests generation
            k8s_manifests = manager._generate_k8s_manifests()
            assert "deployment" in k8s_manifests
            assert "service" in k8s_manifests
            assert "apiVersion" in k8s_manifests["deployment"]
            
            print("✓ Kubernetes manifests generation successful")
            
    except Exception as e:
        print(f"✗ Deployment artifacts test failed: {e}")
        return False
        
    return True

def test_configuration_loading():
    """Test loading of deployment configurations."""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        # Test base deployment config loading
        config_path = project_root / "configs" / "deployment_config.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            assert "environment" in config_data
            assert "deployment_type" in config_data
            assert "optimization_config" in config_data
            
            print("✓ Base deployment config loading successful")
        
        # Test server config loading
        server_config_path = project_root / "configs" / "server_config.json"
        
        if server_config_path.exists():
            with open(server_config_path, 'r') as f:
                server_data = json.load(f)
                
            assert "host" in server_data
            assert "port" in server_data
            assert "workers" in server_data
            
            print("✓ Server config loading successful")
            
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False
        
    return True

def test_imports():
    """Test that all deployment modules can be imported."""
    print("\n=== Testing Module Imports ===")
    
    try:
        # Test individual module imports
        from src.deployment.model_optimizer import ModelOptimizer, OptimizationConfig
        print("✓ Model optimizer import successful")
        
        from src.deployment.model_server import AetheristModelServer, ServerConfig
        print("✓ Model server import successful")
        
        from src.deployment.deployment_manager import DeploymentManager, DeploymentConfig
        print("✓ Deployment manager import successful")
        
        # Test package import
        from src.deployment import (
            ModelOptimizer, AetheristModelServer, DeploymentManager
        )
        print("✓ Package import successful")
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False
        
    return True

def main():
    """Run all deployment tests."""
    print("Starting Aetherist Deployment Infrastructure Tests")
    print("=" * 55)
    
    tests = [
        test_imports,
        test_model_optimizer,
        test_server_config,
        test_deployment_manager,
        test_deployment_artifacts,
        test_configuration_loading
    ]
    
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
            failed += 1
            
    print("\n" + "=" * 55)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All deployment infrastructure tests passed!")
        print("\nThe production deployment tools are ready for use.")
        print("\nNext steps:")
        print("1. Install production dependencies: pip install fastapi uvicorn redis")
        print("2. Run model optimization: python scripts/deploy_optimize.py --help")
        print("3. Start development server: python scripts/deploy_server.py")
        print("4. Deploy to production: python scripts/deploy_full.py")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
