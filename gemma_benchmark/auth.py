"""
Authentication utilities for accessing Gemma models and other gated models.
"""

import os
import logging
from typing import Optional, Dict, Any
from huggingface_hub import login, HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError
import requests

def check_hf_token_validity(token: str) -> bool:
    """
    Check if a HuggingFace token is valid.
    
    Args:
        token: HuggingFace access token
        
    Returns:
        True if token is valid, False otherwise
    """
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        return user_info is not None
    except Exception:
        return False

def check_model_access(model_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if user has access to a specific model.
    
    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-2-2b")
        token: Optional HuggingFace token
        
    Returns:
        Dictionary with access status and details
    """
    logger = logging.getLogger("gemma_benchmark.auth")
    
    try:
        api = HfApi(token=token)
        model_info = api.model_info(model_id)
        
        result = {
            "has_access": True,
            "model_id": model_id,
            "model_name": model_info.modelId,
            "is_gated": getattr(model_info, 'gated', False),
            "license": getattr(model_info, 'license', 'Unknown'),
            "error": None
        }
        
        logger.info(f"Successfully accessed model: {model_id}")
        return result
        
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            return {
                "has_access": False,
                "model_id": model_id,
                "error": "Authentication required or invalid token",
                "suggestion": "Please provide a valid HuggingFace token"
            }
        elif e.response.status_code == 403:
            return {
                "has_access": False,
                "model_id": model_id,
                "error": "Access denied - you may need to accept the model license",
                "suggestion": f"Visit https://huggingface.co/{model_id} to accept the license"
            }
        elif e.response.status_code == 404:
            return {
                "has_access": False,
                "model_id": model_id,
                "error": "Model not found",
                "suggestion": "Please check the model ID is correct"
            }
        else:
            return {
                "has_access": False,
                "model_id": model_id,
                "error": f"HTTP error {e.response.status_code}: {str(e)}",
                "suggestion": "Please check your internet connection and try again"
            }
    except Exception as e:
        return {
            "has_access": False,
            "model_id": model_id,
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please check your configuration and try again"
        }

def get_stored_token() -> Optional[str]:
    """
    Get stored HuggingFace token from various sources.
    
    Returns:
        HuggingFace token if found, None otherwise
    """
    # Check environment variable first
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    
    # Check HuggingFace folder
    try:
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    
    return None

def setup_huggingface_auth(token: Optional[str] = None, force_reauth: bool = False) -> bool:
    """
    Setup HuggingFace authentication for Gemma and other model access.
    
    Args:
        token: Optional HuggingFace token. If None, will try to get from environment/storage
        force_reauth: Force re-authentication even if already authenticated
        
    Returns:
        True if authentication successful, False otherwise
    """
    logger = logging.getLogger("gemma_benchmark.auth")
    
    # Check if already authenticated (unless forced)
    if not force_reauth:
        try:
            api = HfApi()
            user_info = api.whoami()
            if user_info:
                logger.info(f"Already authenticated as: {user_info['name']}")
                return True
        except Exception:
            pass
    
    # Try to get token from parameter, environment, or storage
    if not token:
        token = get_stored_token()
    
    if token:
        # Validate token before using it
        if check_hf_token_validity(token):
            try:
                login(token=token, add_to_git_credential=True)
                api = HfApi()
                user_info = api.whoami()
                logger.info(f"Successfully authenticated as: {user_info['name']}")
                return True
            except Exception as e:
                logger.error(f"Failed to authenticate with provided token: {e}")
        else:
            logger.error("Provided token is invalid")
    
    # Interactive login as fallback
    logger.info("Setting up interactive authentication...")
    logger.info("Steps to authenticate:")
    logger.info("1. Go to https://huggingface.co/settings/tokens")
    logger.info("2. Create a new token with 'Read' access")
    logger.info("3. For Gemma models, accept the license at https://huggingface.co/google/gemma-2-2b")
    logger.info("4. Enter your token when prompted")
    
    try:
        login(add_to_git_credential=True)
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"Interactive authentication successful! Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"Interactive authentication failed: {e}")
        return False

def verify_model_access(model_ids: list[str], show_details: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Verify access to multiple models.
    
    Args:
        model_ids: List of model IDs to check
        show_details: Whether to log detailed access information
        
    Returns:
        Dictionary mapping model IDs to access status
    """
    logger = logging.getLogger("gemma_benchmark.auth")
    
    results = {}
    
    for model_id in model_ids:
        logger.info(f"Checking access to {model_id}...")
        access_info = check_model_access(model_id)
        results[model_id] = access_info
        
        if show_details:
            if access_info["has_access"]:
                logger.info(f"✅ {model_id}: Access granted")
                if access_info.get("is_gated"):
                    logger.info(f"   📋 Model is gated (license: {access_info.get('license', 'Unknown')})")
            else:
                logger.error(f"❌ {model_id}: {access_info['error']}")
                if access_info.get("suggestion"):
                    logger.info(f"   💡 Suggestion: {access_info['suggestion']}")
    
    return results

def get_authentication_status() -> Dict[str, Any]:
    """
    Get current authentication status and user information.
    
    Returns:
        Dictionary containing authentication status and user info
    """
    logger = logging.getLogger("gemma_benchmark.auth")
    
    try:
        api = HfApi()
        user_info = api.whoami()
        
        if user_info:
            return {
                "authenticated": True,
                "username": user_info['name'],
                "email": user_info.get('email'),
                "plan": user_info.get('plan', 'Unknown'),
                "avatar_url": user_info.get('avatarUrl'),
                "token_exists": get_stored_token() is not None
            }
    except Exception as e:
        logger.debug(f"Authentication check failed: {e}")
    
    return {
        "authenticated": False,
        "username": None,
        "email": None,
        "plan": None,
        "avatar_url": None,
        "token_exists": get_stored_token() is not None
    }

def check_gemma_access() -> bool:
    """
    Specifically check access to Gemma models.
    
    Returns:
        True if user has access to Gemma models, False otherwise
    """
    logger = logging.getLogger("gemma_benchmark.auth")
    
    gemma_models = [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it"
    ]
    
    for model_id in gemma_models:
        access_info = check_model_access(model_id)
        if access_info["has_access"]:
            logger.info(f"✅ Gemma access confirmed with {model_id}")
            return True
    
    logger.error("❌ No access to Gemma models found")
    logger.info("To access Gemma models:")
    logger.info("1. Ensure you're authenticated with HuggingFace")
    logger.info("2. Accept the Gemma license at https://huggingface.co/google/gemma-2-2b-it")
    logger.info("3. Wait a few minutes for permissions to propagate")
    
    return False

def setup_complete_auth_flow() -> bool:
    """
    Complete authentication flow for the benchmarking suite.
    
    Returns:
        True if fully authenticated and can access models, False otherwise
    """
    logger = logging.getLogger("gemma_benchmark.auth")
    
    logger.info("🔐 Starting authentication setup...")
    
    # Step 1: Basic HuggingFace authentication
    if not setup_huggingface_auth():
        logger.error("❌ Failed to authenticate with HuggingFace")
        return False
    
    # Step 2: Check Gemma access specifically
    if not check_gemma_access():
        logger.error("❌ Cannot access Gemma models")
        logger.info("Please accept the Gemma license and try again")
        return False
    
    # Step 3: Show authentication status
    status = get_authentication_status()
    logger.info(f"✅ Authentication complete!")
    logger.info(f"   👤 User: {status['username']}")
    logger.info(f"   📧 Email: {status.get('email', 'N/A')}")
    logger.info(f"   🎯 Plan: {status.get('plan', 'Unknown')}")
    
    return True

# Convenience function for the main authentication check
_auth_cache = {"authenticated": False, "last_check": 0}

def ensure_authenticated() -> bool:
    import time
    now = time.time()
    
    # Check cache (valid for 5 minutes)
    if _auth_cache["authenticated"] and (now - _auth_cache["last_check"]) < 300:
        return True
    
    result = setup_complete_auth_flow()
    _auth_cache["authenticated"] = result
    _auth_cache["last_check"] = now
    return result