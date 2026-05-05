"""
============================================================
  Flask Web App Launcher
  MNIST Digit Recognition — Arch Technologies
============================================================
  Run  : python run.py
  Open : http://localhost:5000

  Features:
   ✏  Draw a digit on the canvas
   📁  Upload an image (PNG/JPG)
   🔍  Get real-time prediction + confidence
   📊  See all 10 class probabilities as bar chart
   🏆  See top-3 predictions
============================================================
"""

import os, sys

import config
from utils.logger import setup_logger
from flask_app.app import app, get_model

# Setup logging
logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("\n" + "="*55)
    logger.info("MNIST Digit Recognition — Live Web App")
    logger.info("Arch Technologies Internship")
    logger.info("="*55)
    logger.info("Pre-loading CNN model...")
    get_model()
    logger.info("Model ready")
    logger.info(f"Visit http://localhost:{config.FLASK_PORT}")
    logger.info("="*55 + "\n")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
