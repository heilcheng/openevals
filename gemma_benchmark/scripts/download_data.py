#!/usr/bin/env python3
"""
Script to download benchmark datasets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to make imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gemma_benchmark.utils.data_downloader import (
    download_mmlu_data,
    download_gsm8k_data,
    download_humaneval_data
)

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    
    parser.add_argument(
        "--mmlu",
        action="store_true",
        help="Download MMLU (Massive Multitask Language Understanding) dataset"
    )
    
    parser.add_argument(
        "--gsm8k",
        action="store_true",
        help="Download GSM8K (Grade School Math 8K) dataset"
    )
    
    parser.add_argument(
        "--humaneval",
        action="store_true",
        help="Download HumanEval dataset for code generation"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if data already exists"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for storing datasets"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger("gemma_benchmark.scripts.download_data")
    logger.info("Starting data download")
    
    # Create base data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Download selected datasets
    if args.mmlu or args.all:
        mmlu_dir = os.path.join(args.data_dir, "mmlu")
        logger.info(f"Downloading MMLU dataset to {mmlu_dir}")
        download_mmlu_data(mmlu_dir, args.force)
    
    if args.gsm8k or args.all:
        gsm8k_dir = os.path.join(args.data_dir, "gsm8k")
        logger.info(f"Downloading GSM8K dataset to {gsm8k_dir}")
        download_gsm8k_data(gsm8k_dir, args.force)
    
    if args.humaneval or args.all:
        humaneval_dir = os.path.join(args.data_dir, "humaneval")
        logger.info(f"Downloading HumanEval dataset to {humaneval_dir}")
        download_humaneval_data(humaneval_dir, args.force)
    
    # If no dataset was selected, show help
    if not (args.mmlu or args.gsm8k or args.humaneval or args.all):
        logger.warning("No dataset selected. Please specify at least one dataset to download.")
        print("\nUse one of the following options to download datasets:")
        print("  --mmlu       Download MMLU dataset")
        print("  --gsm8k      Download GSM8K dataset")
        print("  --humaneval  Download HumanEval dataset")
        print("  --all        Download all datasets")
        print("\nExample: python -m gemma_benchmark.scripts.download_data --mmlu")
    else:
        logger.info("Data download complete")

if __name__ == "__main__":
    main()
