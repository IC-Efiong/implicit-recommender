"""Main entry point for the recommender system.
Run with: python run.py
"""

import argparse
from app.views import app
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='Implicit Feedback Recommender System')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the recommendation model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the web application')
    run_parser.add_argument('--port', type=int, default=5000,
                          help='Port to run the application on')
    run_parser.add_argument('--host', type=str, default='0.0.0.0',
                          help='Host to run the application on')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        logger.info("Training model...")
        train_model()
    elif args.command == 'evaluate':
        logger.info("Evaluating model...")
        evaluate()
    elif args.command == 'run':
        logger.info(f"Starting web application on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    main()