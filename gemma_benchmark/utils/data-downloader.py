"""
Data downloading utilities for benchmark datasets.
"""

import os
import requests
import zipfile
import logging
import shutil
import tarfile
from tqdm import tqdm

def download_file(url, destination, description=None):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        description: Description for the progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    desc = description or f"Downloading {os.path.basename(destination)}"
    with open(destination, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
        
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

def download_mmlu_data(target_dir="data/mmlu", force=False):
    """
    Download MMLU (Massive Multitask Language Understanding) dataset.
    
    Args:
        target_dir: Directory to save the dataset
        force: Whether to force download even if data exists
    """
    logger = logging.getLogger("gemma_benchmark.utils.data_downloader")
    
    # Convert to absolute path if relative
    target_dir = os.path.abspath(target_dir)
    
    # Check if data already exists
    if not force and os.path.exists(target_dir) and os.listdir(target_dir):
        logger.info(f"MMLU data already exists in {target_dir}. Use force=True to redownload.")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # MMLU dataset URL (this is just a sample - adjust with the actual URL)
    # In a real implementation, you would use the official MMLU source
    url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    
    # Temporary download location
    temp_dir = os.path.join(target_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    tar_path = os.path.join(temp_dir, "mmlu_data.tar")
    
    try:
        # Download data
        logger.info(f"Downloading MMLU data from {url}")
        download_file(url, tar_path, "Downloading MMLU dataset")
        
        # Extract data
        logger.info(f"Extracting data to {target_dir}")
        with tarfile.open(tar_path) as tar:
            # Filter out problematic filenames
            for member in tqdm(tar.getmembers(), desc="Extracting files"):
                # Extract only files that belong to the MMLU dataset
                if member.name.startswith("data/"):
                    # Remove the initial "data/" from the path
                    member.name = member.name.replace("data/", "", 1)
                    tar.extract(member, target_dir)
        
        logger.info("MMLU data downloaded and extracted successfully")
    except Exception as e:
        logger.error(f"Error downloading MMLU data: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def download_gsm8k_data(target_dir="data/gsm8k", force=False):
    """
    Download GSM8K (Grade School Math 8K) dataset.
    
    Args:
        target_dir: Directory to save the dataset
        force: Whether to force download even if data exists
    """
    logger = logging.getLogger("gemma_benchmark.utils.data_downloader")
    
    # Convert to absolute path if relative
    target_dir = os.path.abspath(target_dir)
    
    # Check if data already exists
    if not force and os.path.exists(target_dir) and os.listdir(target_dir):
        logger.info(f"GSM8K data already exists in {target_dir}. Use force=True to redownload.")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # GSM8K dataset URLs
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    
    # Download files
    logger.info(f"Downloading GSM8K training data from {train_url}")
    download_file(train_url, os.path.join(target_dir, "train.jsonl"), "Downloading GSM8K training data")
    
    logger.info(f"Downloading GSM8K test data from {test_url}")
    download_file(test_url, os.path.join(target_dir, "test.jsonl"), "Downloading GSM8K test data")
    
    logger.info("GSM8K data downloaded successfully")

def download_humaneval_data(target_dir="data/humaneval", force=False):
    """
    Download HumanEval dataset for code generation evaluation.
    
    Args:
        target_dir: Directory to save the dataset
        force: Whether to force download even if data exists
    """
    logger = logging.getLogger("gemma_benchmark.utils.data_downloader")
    
    # Convert to absolute path if relative
    target_dir = os.path.abspath(target_dir)
    
    # Check if data already exists
    if not force and os.path.exists(target_dir) and os.listdir(target_dir):
        logger.info(f"HumanEval data already exists in {target_dir}. Use force=True to redownload.")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # HumanEval dataset URL
    url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
    
    # Download file
    logger.info(f"Downloading HumanEval data from {url}")
    download_file(url, os.path.join(target_dir, "HumanEval.jsonl.gz"), "Downloading HumanEval data")
    
    logger.info("HumanEval data downloaded successfully")
