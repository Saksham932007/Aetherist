#!/usr/bin/env python3
"""Script for running Aetherist production server."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.model_server import AetheristModelServer, ServerConfig

def main():
    parser = argparse.ArgumentParser(description="Run Aetherist production server")
    parser.add_argument("--config", type=str, default="configs/server_config.json",
                       help="Path to server configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind server to")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    parser.add_argument("--model-config", type=str, default="configs/base_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--redis-url", type=str,
                       help="Redis URL for task queue")
    parser.add_argument("--cors-origins", type=str, nargs="*",
                       help="CORS allowed origins")
    parser.add_argument("--enable-auth", action="store_true",
                       help="Enable API authentication")
    parser.add_argument("--api-key", type=str,
                       help="API key for authentication")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load server configuration if file exists
        if Path(args.config).exists():
            logger.info(f"Loading server configuration from {args.config}")
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            server_config = ServerConfig(**config_dict)
        else:
            logger.info("Using default server configuration")
            server_config = ServerConfig()
            
        # Override with command line arguments
        if args.host:
            server_config.host = args.host
        if args.port:
            server_config.port = args.port
        if args.workers:
            server_config.workers = args.workers
        if args.model_config:
            server_config.model_config_path = args.model_config
        if args.checkpoint:
            server_config.checkpoint_path = args.checkpoint
        if args.log_level:
            server_config.log_level = args.log_level
        if args.redis_url:
            server_config.redis_url = args.redis_url
        if args.cors_origins:
            server_config.cors_origins = args.cors_origins
        if args.enable_auth:
            server_config.enable_auth = True
        if args.api_key:
            server_config.api_key = args.api_key
            
        # Validate configuration
        if server_config.enable_auth and not server_config.api_key:
            logger.error("API key is required when authentication is enabled")
            sys.exit(1)
            
        # Create and run server
        logger.info(f"Starting Aetherist server on {server_config.host}:{server_config.port}")
        logger.info(f"Model config: {server_config.model_config_path}")
        if server_config.checkpoint_path:
            logger.info(f"Checkpoint: {server_config.checkpoint_path}")
            
        server = AetheristModelServer(server_config)
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
