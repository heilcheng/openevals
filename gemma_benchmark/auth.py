"""
Authentication utilities for accessing Gemma models.
"""

import os
import logging
from huggingface_hub import login, HfApi

def setup_huggingface_auth():
    """Setup HuggingFace authentication for Gemma access."""
    logger = logging.getLogger("gemma_benchmark.auth")
    
    # Check if already authenticated
    try:
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"Already authenticated as: {user_info['name']}")
        return True
    except:
        pass
    
    # Try to get token from environment
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Authenticated using HF_TOKEN environment variable")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with token: {e}")
    
    # Interactive login
    logger.info("Please authenticate with HuggingFace to access Gemma models:")
    logger.info("1. Go to https://huggingface.co/settings/tokens")
    logger.info("2. Create a new token with 'Read' access")
    logger.info("3. Accept the Gemma license at https://huggingface.co/google/gemma-2-2b")
    
    try:
        login()
        logger.info("Authentication successful!")
        return True
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False