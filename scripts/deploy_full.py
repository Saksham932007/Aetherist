#!/usr/bin/env python3
"""Script for complete Aetherist deployment process."""

import argparse
import logging
import sys
import yaml
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.deployment_manager import DeploymentManager, DeploymentConfig

def main():
    parser = argparse.ArgumentParser(description="Deploy Aetherist to production")
    parser.add_argument("--config", type=str, default="configs/deployment_config.yaml",
                       help="Path to deployment configuration file")
    parser.add_argument("--deployment-type", type=str, 
                       choices=["docker", "kubernetes", "aws", "local"],
                       help="Deployment type (overrides config file)")
    parser.add_argument("--environment", type=str,
                       choices=["development", "staging", "production"],
                       help="Deployment environment (overrides config file)")
    parser.add_argument("--optimize-models", action="store_true",
                       help="Optimize models for deployment")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Skip model optimization")
    parser.add_argument("--checkpoint-path", type=str,
                       help="Path to model checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show deployment plan without executing")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up previous deployment artifacts")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load deployment configuration
        if Path(args.config).exists():
            logger.info(f"Loading deployment configuration from {args.config}")
            with open(args.config, 'r') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
                    
            deployment_config = DeploymentConfig(**config_dict)
        else:
            logger.info("Using default deployment configuration")
            deployment_config = DeploymentConfig()
            
        # Override with command line arguments
        if args.deployment_type:
            deployment_config.deployment_type = args.deployment_type
        if args.environment:
            deployment_config.environment = args.environment
        if args.checkpoint_path:
            deployment_config.checkpoint_path = args.checkpoint_path
        if args.optimize_models:
            deployment_config.optimize_models = True
        if args.no_optimize:
            deployment_config.optimize_models = False
            
        # Create deployment manager
        deployment_manager = DeploymentManager(deployment_config, project_root)
        
        # Handle cleanup
        if args.cleanup:
            logger.info("Cleaning up previous deployment artifacts...")
            deployment_manager.cleanup()
            logger.info("Cleanup completed")
            return
            
        # Show deployment plan
        logger.info("\n=== Deployment Plan ===")
        logger.info(f"Deployment Type: {deployment_config.deployment_type}")
        logger.info(f"Environment: {deployment_config.environment}")
        logger.info(f"Model Config: {deployment_config.model_config_path}")
        logger.info(f"Optimize Models: {deployment_config.optimize_models}")
        if deployment_config.checkpoint_path:
            logger.info(f"Checkpoint: {deployment_config.checkpoint_path}")
            
        if deployment_config.deployment_type == "docker":
            logger.info(f"Docker Base Image: {deployment_config.docker_config.base_image}")
            logger.info(f"Docker Port: {deployment_config.docker_config.port}")
            logger.info(f"GPU Support: {deployment_config.docker_config.gpu_support}")
        elif deployment_config.deployment_type == "kubernetes":
            logger.info(f"K8s Namespace: {deployment_config.kubernetes_config.namespace}")
            logger.info(f"K8s Service: {deployment_config.kubernetes_config.service_name}")
            logger.info(f"K8s Replicas: {deployment_config.kubernetes_config.replicas}")
        elif deployment_config.deployment_type == "aws":
            logger.info(f"AWS Region: {deployment_config.aws_config.region}")
            logger.info(f"Instance Type: {deployment_config.aws_config.instance_type}")
            
        logger.info("========================\n")
        
        if args.dry_run:
            logger.info("Dry run completed. Use --no-dry-run to execute deployment.")
            return
            
        # Execute deployment
        logger.info("Starting deployment...")
        deployment_result = deployment_manager.deploy()
        
        # Display results
        logger.info("\n=== Deployment Results ===")
        logger.info(f"Status: {deployment_result['status']}")
        logger.info(f"Deployment Type: {deployment_result['deployment_type']}")
        logger.info(f"Environment: {deployment_result['environment']}")
        logger.info(f"Timestamp: {deployment_result['timestamp']}")
        
        if deployment_result['status'] == 'completed':
            if 'optimization' in deployment_result:
                logger.info(f"Model optimization: {deployment_result['optimization']}")
                
            if deployment_config.deployment_type == "docker":
                logger.info(f"Docker image: {deployment_result.get('image_name')}")
                logger.info(f"Dockerfile: {deployment_result.get('dockerfile')}")
                logger.info(f"Compose file: {deployment_result.get('compose_file')}")
                logger.info("\nTo run the container:")
                logger.info(f"cd {project_root}/deployments/docker && docker-compose up")
            elif deployment_config.deployment_type == "kubernetes":
                logger.info(f"K8s manifests: {deployment_result.get('manifests')}")
                logger.info("\nTo deploy to Kubernetes:")
                for manifest in deployment_result.get('manifests', []):
                    logger.info(f"kubectl apply -f {manifest}")
            elif deployment_config.deployment_type == "local":
                logger.info(f"Run script: {deployment_result.get('run_script')}")
                logger.info("\nTo start the server locally:")
                logger.info(f"bash {deployment_result.get('run_script')}")
                
        else:
            logger.error(f"Deployment failed: {deployment_result.get('error')}")
            sys.exit(1)
            
        logger.info("\nDeployment completed successfully!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
