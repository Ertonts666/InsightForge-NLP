"""
Main entry point for running the InsightForge-NLP API.
This script starts the FastAPI server with the InsightForge-NLP API.
"""

import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from app.common.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the InsightForge-NLP API."""
    parser = argparse.ArgumentParser(description='InsightForge-NLP API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the API server on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the API server on')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration if provided
    if args.config:
        config = Config(args.config)
        # Override with command line arguments
        if args.host:
            config.set('api.host', args.host)
        if args.port:
            config.set('api.port', args.port)
    else:
        # Use command line arguments directly
        host = args.host
        port = args.port
    
    # Log startup information
    logger.info(f"Starting InsightForge-NLP API on {args.host}:{args.port}")
    logger.info(f"API documentation will be available at http://{args.host}:{args.port}/docs")
    
    # Run the API server
    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()
