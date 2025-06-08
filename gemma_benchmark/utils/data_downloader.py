"""
Data downloading utilities using HuggingFace datasets.
"""

import os
import logging
from datasets import load_dataset

def download_mmlu_data(target_dir="data/mmlu", force=False):
    """Download MMLU dataset using HuggingFace datasets."""
    logger = logging.getLogger("gemma_benchmark.utils.data_downloader")
    
    if not force and os.path.exists(os.path.join(target_dir, ".downloaded")):
        logger.info(f"MMLU data already cached. Use force=True to redownload.")
        return
    
    logger.info("Downloading MMLU dataset...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Download and cache the dataset
        dataset = load_dataset("cais/mmlu", "all", cache_dir=target_dir)
        
        # Create marker file
        with open(os.path.join(target_dir, ".downloaded"), "w") as f:
            f.write("MMLU dataset downloaded successfully")
        
        logger.info("MMLU dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download MMLU: {e}")
        raise

def download_gsm8k_data(target_dir="data/gsm8k", force=False):
    """Download GSM8K dataset."""
    logger = logging.getLogger("gemma_benchmark.utils.data_downloader")
    
    if not force and os.path.exists(os.path.join(target_dir, ".downloaded")):
        logger.info(f"GSM8K data already cached.")
        return
    
    logger.info("Downloading GSM8K dataset...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("gsm8k", "main", cache_dir=target_dir)
        
        with open(os.path.join(target_dir, ".downloaded"), "w") as f:
            f.write("GSM8K dataset downloaded successfully")
        
        logger.info("GSM8K dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download GSM8K: {e}")
        raise

def download_humaneval_data(target_dir="data/humaneval", force=False):
    """Download HumanEval dataset."""
    logger = logging.getLogger("gemma_benchmark.utils.data_downloader")
    
    if not force and os.path.exists(os.path.join(target_dir, ".downloaded")):
        logger.info(f"HumanEval data already cached.")
        return
    
    logger.info("Downloading HumanEval dataset...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("openai_humaneval", cache_dir=target_dir)
        
        with open(os.path.join(target_dir, ".downloaded"), "w") as f:
            f.write("HumanEval dataset downloaded successfully")
        
        logger.info("HumanEval dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download HumanEval: {e}")
        raise