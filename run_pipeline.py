"""
CareerPulse Pipeline Launcher
=============================
Executes full data generation, data cleaning, feature engineering,
and multi-model training for tech salary valuation.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.train import run_pipeline

if __name__ == "__main__":
    run_pipeline()
