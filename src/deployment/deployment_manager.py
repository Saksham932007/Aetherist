import os
import shutil
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging
import time
import tempfile
from jinja2 import Template

from .model_optimizer import ModelOptimizer, OptimizationConfig
from .model_server import ServerConfig
from ..config.config_manager import ConfigManager

@dataclass
class DockerConfig:
    """Configuration for Docker deployment."""
    base_image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    python_version: str = "3.10"
    port: int = 8000
    workers: int = 1
    memory_limit: str = "8g"
    cpu_limit: str = "4"
    gpu_support: bool = True
    
@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes deployment."""
    namespace: str = "aetherist"
    service_name: str = "aetherist-api"
    replicas: int = 2
    image_tag: str = "latest"
    resource_requests: Dict[str, str] = None
    resource_limits: Dict[str, str] = None
    ingress_host: Optional[str] = None
    
    def __post_init__(self):
        if self.resource_requests is None:
            self.resource_requests = {"cpu": "2", "memory": "4Gi"}
        if self.resource_limits is None:
            self.resource_limits = {"cpu": "4", "memory": "8Gi"}
            
@dataclass
class AWSConfig:
    """Configuration for AWS deployment."""
    region: str = "us-west-2"
    instance_type: str = "g4dn.xlarge"
    min_capacity: int = 1
    max_capacity: int = 5
    target_cpu_utilization: int = 70
    vpc_id: Optional[str] = None
    subnet_ids: Optional[List[str]] = None
    security_group_ids: Optional[List[str]] = None
    
@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    environment: str = "production"  # development, staging, production
    deployment_type: str = "docker"  # docker, kubernetes, aws, local
    model_config_path: str = "configs/base_config.yaml"
    checkpoint_path: Optional[str] = None
    optimize_models: bool = True
    optimization_config: OptimizationConfig = None
    server_config: ServerConfig = None
    docker_config: DockerConfig = None
    kubernetes_config: KubernetesConfig = None
    aws_config: AWSConfig = None
    
    def __post_init__(self):
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if self.server_config is None:
            self.server_config = ServerConfig()
        if self.docker_config is None:
            self.docker_config = DockerConfig()
        if self.kubernetes_config is None:
            self.kubernetes_config = KubernetesConfig()
        if self.aws_config is None:
            self.aws_config = AWSConfig()
            
class DeploymentManager:
    """Manage production deployments of Aetherist models."""
    
    def __init__(self, config: DeploymentConfig, workspace_root: Path):
        self.config = config
        self.workspace_root = Path(workspace_root)
        self.logger = self._setup_logging()
        self.deployment_dir = self.workspace_root / "deployments"
        self.deployment_dir.mkdir(exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def deploy(self) -> Dict[str, Any]:
        """Execute complete deployment process."""
        self.logger.info(f"Starting {self.config.deployment_type} deployment...")
        
        deployment_info = {
            "deployment_type": self.config.deployment_type,
            "environment": self.config.environment,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "started"
        }
        
        try:
            # 1. Optimize models if requested
            if self.config.optimize_models:
                optimization_results = self._optimize_models()
                deployment_info["optimization"] = optimization_results
                
            # 2. Prepare deployment artifacts
            artifact_info = self._prepare_deployment_artifacts()
            deployment_info["artifacts"] = artifact_info
            
            # 3. Execute deployment based on type
            if self.config.deployment_type == "docker":
                deployment_result = self._deploy_docker()
            elif self.config.deployment_type == "kubernetes":
                deployment_result = self._deploy_kubernetes()
            elif self.config.deployment_type == "aws":
                deployment_result = self._deploy_aws()
            elif self.config.deployment_type == "local":
                deployment_result = self._deploy_local()
            else:
                raise ValueError(f"Unsupported deployment type: {self.config.deployment_type}")
                
            deployment_info.update(deployment_result)
            deployment_info["status"] = "completed"
            
            # 4. Save deployment info
            self._save_deployment_info(deployment_info)
            
            self.logger.info("Deployment completed successfully")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            deployment_info["status"] = "failed"
            deployment_info["error"] = str(e)
            self._save_deployment_info(deployment_info)
            raise
            
    def _optimize_models(self) -> Dict[str, Any]:
        """Optimize models for deployment."""
        self.logger.info("Optimizing models...")
        
        optimizer = ModelOptimizer(self.config.optimization_config)
        optimization_dir = self.deployment_dir / "optimized_models"
        optimization_dir.mkdir(exist_ok=True)
        
        # Load models (placeholder - would need actual model loading)
        # For now, create dummy optimization results
        optimization_results = {
            "generator": {
                "optimized": str(optimization_dir / "generator_optimized.pth"),
                "onnx": str(optimization_dir / "generator.onnx"),
                "torchscript": str(optimization_dir / "generator_traced.pt")
            },
            "discriminator": {
                "optimized": str(optimization_dir / "discriminator_optimized.pth"),
                "onnx": str(optimization_dir / "discriminator.onnx"),
                "torchscript": str(optimization_dir / "discriminator_traced.pt")
            }
        }
        
        # Save optimization report
        report_path = optimization_dir / "optimization_report.json"
        optimizer.save_optimization_report(
            optimization_results["generator"],
            {},  # benchmark_results would be filled by actual benchmarking
            report_path
        )
        
        return {
            "optimization_dir": str(optimization_dir),
            "models": optimization_results,
            "report": str(report_path)
        }
        
    def _prepare_deployment_artifacts(self) -> Dict[str, str]:
        """Prepare all deployment artifacts."""
        self.logger.info("Preparing deployment artifacts...")
        
        artifact_dir = self.deployment_dir / f"{self.config.deployment_type}_artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        artifacts = {}
        
        # Copy source code
        src_dir = artifact_dir / "src"
        if src_dir.exists():
            shutil.rmtree(src_dir)
        shutil.copytree(self.workspace_root / "src", src_dir)
        artifacts["source"] = str(src_dir)
        
        # Copy configs
        config_dir = artifact_dir / "configs"
        if config_dir.exists():
            shutil.rmtree(config_dir)
        shutil.copytree(self.workspace_root / "configs", config_dir)
        artifacts["configs"] = str(config_dir)
        
        # Copy requirements
        shutil.copy2(self.workspace_root / "requirements.txt", artifact_dir)
        artifacts["requirements"] = str(artifact_dir / "requirements.txt")
        
        # Generate deployment-specific configs
        server_config_path = artifact_dir / "server_config.json"
        with open(server_config_path, 'w') as f:
            json.dump(asdict(self.config.server_config), f, indent=2)
        artifacts["server_config"] = str(server_config_path)
        
        return artifacts
        
    def _deploy_docker(self) -> Dict[str, Any]:
        """Deploy using Docker."""
        self.logger.info("Starting Docker deployment...")
        
        docker_dir = self.deployment_dir / "docker"
        docker_dir.mkdir(exist_ok=True)
        
        # Generate Dockerfile
        dockerfile_content = self._generate_dockerfile()
        dockerfile_path = docker_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
            
        # Generate docker-compose.yml
        compose_content = self._generate_docker_compose()
        compose_path = docker_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
            
        # Build Docker image
        image_name = f"aetherist-api:{self.config.environment}"
        build_cmd = [
            "docker", "build", 
            "-t", image_name,
            "-f", str(dockerfile_path),
            str(self.workspace_root)
        ]
        
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed: {result.stderr}")
            
        return {
            "image_name": image_name,
            "dockerfile": str(dockerfile_path),
            "compose_file": str(compose_path),
            "build_log": result.stdout
        }
        
    def _deploy_kubernetes(self) -> Dict[str, Any]:
        """Deploy using Kubernetes."""
        self.logger.info("Starting Kubernetes deployment...")
        
        k8s_dir = self.deployment_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        # Generate Kubernetes manifests
        manifests = self._generate_k8s_manifests()
        
        manifest_files = []
        for name, content in manifests.items():
            manifest_path = k8s_dir / f"{name}.yaml"
            with open(manifest_path, 'w') as f:
                f.write(content)
            manifest_files.append(str(manifest_path))
            
        return {
            "namespace": self.config.kubernetes_config.namespace,
            "service_name": self.config.kubernetes_config.service_name,
            "manifests": manifest_files
        }
        
    def _deploy_aws(self) -> Dict[str, Any]:
        """Deploy using AWS services."""
        self.logger.info("Starting AWS deployment...")
        
        aws_dir = self.deployment_dir / "aws"
        aws_dir.mkdir(exist_ok=True)
        
        # Generate CloudFormation template
        cf_template = self._generate_cloudformation_template()
        cf_path = aws_dir / "template.yaml"
        with open(cf_path, 'w') as f:
            f.write(cf_template)
            
        return {
            "region": self.config.aws_config.region,
            "cloudformation_template": str(cf_path),
            "instance_type": self.config.aws_config.instance_type
        }
        
    def _deploy_local(self) -> Dict[str, Any]:
        """Deploy locally for development."""
        self.logger.info("Starting local deployment...")
        
        # Generate local run script
        run_script = self._generate_local_run_script()
        script_path = self.deployment_dir / "run_local.sh"
        with open(script_path, 'w') as f:
            f.write(run_script)
        script_path.chmod(0o755)
        
        return {
            "run_script": str(script_path),
            "server_config": str(self.deployment_dir / "docker_artifacts/server_config.json")
        }
        
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        template = Template(
        """FROM {{ base_image }}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY deployments/docker_artifacts/server_config.json ./

# Copy optimized models if available
COPY deployments/optimized_models/ ./models/ || echo "No optimized models found"

# Expose port
EXPOSE {{ port }}

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_CONFIG_PATH=/app/configs/base_config.yaml
{% if gpu_support %}
ENV NVIDIA_VISIBLE_DEVICES=all
{% endif %}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:{{ port }}/health || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "src.deployment.model_server:app", "--host", "0.0.0.0", "--port", "{{ port }}", "--workers", "{{ workers }}"]
"""
        )
        
        return template.render(
            base_image=self.config.docker_config.base_image,
            port=self.config.docker_config.port,
            workers=self.config.docker_config.workers,
            gpu_support=self.config.docker_config.gpu_support
        )
        
    def _generate_docker_compose(self) -> str:
        """Generate docker-compose.yml content."""
        template = Template(
        """version: '3.8'

services:
  aetherist-api:
    build:
      context: ../../
      dockerfile: deployments/docker/Dockerfile
    ports:
      - "{{ port }}:{{ port }}"
    environment:
      - ENVIRONMENT={{ environment }}
    deploy:
      resources:
        limits:
          memory: {{ memory_limit }}
          cpus: '{{ cpu_limit }}'
{% if gpu_support %}
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
{% endif %}
    volumes:
      - ../optimized_models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{{ port }}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
"""
        )
        
        return template.render(
            port=self.config.docker_config.port,
            environment=self.config.environment,
            memory_limit=self.config.docker_config.memory_limit,
            cpu_limit=self.config.docker_config.cpu_limit,
            gpu_support=self.config.docker_config.gpu_support
        )
        
    def _generate_k8s_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes manifest files."""
        k8s_config = self.config.kubernetes_config
        
        # Deployment manifest
        deployment_template = Template(
        """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ service_name }}
  namespace: {{ namespace }}
spec:
  replicas: {{ replicas }}
  selector:
    matchLabels:
      app: {{ service_name }}
  template:
    metadata:
      labels:
        app: {{ service_name }}
    spec:
      containers:
      - name: {{ service_name }}
        image: aetherist-api:{{ image_tag }}
        ports:
        - containerPort: 8000
        resources:
          requests:
{% for key, value in resource_requests.items() %}
            {{ key }}: {{ value }}
{% endfor %}
          limits:
{% for key, value in resource_limits.items() %}
            {{ key }}: {{ value }}
{% endfor %}
        env:
        - name: ENVIRONMENT
          value: "{{ environment }}"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        )
        
        # Service manifest
        service_template = Template(
        """apiVersion: v1
kind: Service
metadata:
  name: {{ service_name }}
  namespace: {{ namespace }}
spec:
  selector:
    app: {{ service_name }}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
"""
        )
        
        manifests = {
            "deployment": deployment_template.render(
                service_name=k8s_config.service_name,
                namespace=k8s_config.namespace,
                replicas=k8s_config.replicas,
                image_tag=k8s_config.image_tag,
                resource_requests=k8s_config.resource_requests,
                resource_limits=k8s_config.resource_limits,
                environment=self.config.environment
            ),
            "service": service_template.render(
                service_name=k8s_config.service_name,
                namespace=k8s_config.namespace
            )
        }
        
        # Add ingress if configured
        if k8s_config.ingress_host:
            ingress_template = Template(
            """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ service_name }}
  namespace: {{ namespace }}
spec:
  rules:
  - host: {{ ingress_host }}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {{ service_name }}
            port:
              number: 80
"""
            )
            
            manifests["ingress"] = ingress_template.render(
                service_name=k8s_config.service_name,
                namespace=k8s_config.namespace,
                ingress_host=k8s_config.ingress_host
            )
            
        return manifests
        
    def _generate_cloudformation_template(self) -> str:
        """Generate AWS CloudFormation template."""
        # Simplified CloudFormation template for ECS deployment
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Aetherist API deployment on AWS",
            "Parameters": {
                "ImageURI": {
                    "Type": "String",
                    "Description": "Docker image URI"
                }
            },
            "Resources": {
                "AetheristCluster": {
                    "Type": "AWS::ECS::Cluster",
                    "Properties": {
                        "ClusterName": "aetherist-cluster"
                    }
                }
                # Add more AWS resources as needed
            }
        }
        
        return yaml.dump(template, default_flow_style=False)
        
    def _generate_local_run_script(self) -> str:
        """Generate local run script."""
        return f"""#!/bin/bash

# Aetherist Local Development Server

set -e

echo "Starting Aetherist development server..."

# Set environment variables
export PYTHONPATH="{self.workspace_root}"
export MODEL_CONFIG_PATH="{self.workspace_root}/configs/base_config.yaml"
export ENVIRONMENT="{self.config.environment}"

# Activate virtual environment if it exists
if [ -d "{self.workspace_root}/.venv" ]; then
    source "{self.workspace_root}/.venv/bin/activate"
    echo "Activated virtual environment"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r "{self.workspace_root}/requirements.txt"

# Run server
echo "Starting server on port {self.config.server_config.port}..."
cd "{self.workspace_root}"
python -c "from src.deployment.model_server import create_server; create_server().run()"
"""
        
    def _save_deployment_info(self, deployment_info: Dict[str, Any]):
        """Save deployment information to file."""
        info_path = self.deployment_dir / f"{self.config.deployment_type}_deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
            
        self.logger.info(f"Deployment info saved to {info_path}")
        
    def rollback(self, deployment_id: Optional[str] = None):
        """Rollback to previous deployment."""
        self.logger.info("Rolling back deployment...")
        # Implementation would depend on deployment type
        # For now, just log the action
        self.logger.info("Rollback completed")
        
    def cleanup(self):
        """Clean up deployment artifacts."""
        self.logger.info("Cleaning up deployment artifacts...")
        
        # Remove old deployment directories
        cleanup_dirs = ["docker_artifacts", "kubernetes", "aws"]
        for dir_name in cleanup_dirs:
            dir_path = self.deployment_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.logger.info(f"Removed {dir_path}")
                
        self.logger.info("Cleanup completed")
        
def create_deployment_manager(config_path: str, workspace_root: str) -> DeploymentManager:
    """Factory function to create deployment manager."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
            
    config = DeploymentConfig(**config_dict)
    return DeploymentManager(config, Path(workspace_root))
