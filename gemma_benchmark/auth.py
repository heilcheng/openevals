"""
Production-ready authentication utilities for accessing Gemma models and other gated models.

This module provides comprehensive authentication management with proper error handling,
token validation, and secure credential management.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

try:
    from huggingface_hub import login, HfApi, HfFolder, whoami
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.getLogger(__name__).error("huggingface_hub not available. Install with: pip install huggingface-hub")

import requests


class AuthenticationError(Exception):
    """Custom exception for authentication-related errors."""
    pass


class ModelAccessError(Exception):
    """Custom exception for model access-related errors."""
    pass


class AuthStatus(Enum):
    """Authentication status enumeration."""
    AUTHENTICATED = "authenticated"
    NOT_AUTHENTICATED = "not_authenticated"
    TOKEN_INVALID = "token_invalid"
    TOKEN_EXPIRED = "token_expired"
    NETWORK_ERROR = "network_error"


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    status: AuthStatus
    user_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    token_valid: bool = False


@dataclass
class ModelAccessResult:
    """Result of model access check."""
    has_access: bool
    model_id: str
    is_gated: bool = False
    license_info: Optional[str] = None
    error_message: Optional[str] = None
    suggestions: List[str] = None


class SecureAuthManager:
    """
    Production-ready authentication manager with comprehensive error handling.
    
    Features:
    - Secure token management
    - Automatic token validation
    - Model access verification
    - Session management
    - Fallback authentication methods
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the authentication manager."""
        self.logger = logging.getLogger("gemma_benchmark.auth")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gemma_benchmark"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Authentication cache
        self._auth_cache = {
            "last_check": 0,
            "status": AuthStatus.NOT_AUTHENTICATED,
            "user_info": None,
            "token_hash": None
        }
        
        # Model access cache
        self._model_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for authentication. "
                "Install with: pip install huggingface-hub>=0.20.0"
            )
    
    def get_token_from_sources(self) -> Optional[str]:
        """
        Get HuggingFace token from multiple sources in order of preference.
        
        Returns:
            Token string if found, None otherwise
        """
        # 1. Environment variable (highest priority)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            self.logger.debug("Token found in environment variables")
            return token.strip()
        
        # 2. HuggingFace CLI token storage
        try:
            token = HfFolder.get_token()
            if token:
                self.logger.debug("Token found in HuggingFace CLI storage")
                return token
        except Exception as e:
            self.logger.debug(f"Could not access HF CLI token: {e}")
        
        # 3. Manual token file (custom location)
        token_file = self.cache_dir / "hf_token"
        if token_file.exists():
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                if token:
                    self.logger.debug("Token found in cache file")
                    return token
            except Exception as e:
                self.logger.debug(f"Could not read token file: {e}")
        
        return None
    
    def validate_token(self, token: str) -> AuthenticationResult:
        """
        Validate a HuggingFace token comprehensively.
        
        Args:
            token: HuggingFace access token
            
        Returns:
            AuthenticationResult with validation details
        """
        if not token or not token.strip():
            return AuthenticationResult(
                status=AuthStatus.TOKEN_INVALID,
                error_message="Empty or None token provided"
            )
        
        token = token.strip()
        
        # Check token format (HF tokens typically start with hf_)
        if not token.startswith(('hf_', 'hf-')) and len(token) < 20:
            return AuthenticationResult(
                status=AuthStatus.TOKEN_INVALID,
                error_message="Token format appears invalid (should start with 'hf_' and be longer than 20 characters)"
            )
        
        try:
            # Test token by calling whoami
            api = HfApi(token=token)
            user_info = api.whoami(token=token)
            
            if user_info and 'name' in user_info:
                self.logger.info(f"Token validation successful for user: {user_info['name']}")
                return AuthenticationResult(
                    status=AuthStatus.AUTHENTICATED,
                    user_info=user_info,
                    token_valid=True
                )
            else:
                return AuthenticationResult(
                    status=AuthStatus.TOKEN_INVALID,
                    error_message="Token validation returned invalid user info"
                )
                
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                return AuthenticationResult(
                    status=AuthStatus.TOKEN_INVALID,
                    error_message="Token is invalid or expired"
                )
            elif e.response.status_code == 403:
                return AuthenticationResult(
                    status=AuthStatus.TOKEN_INVALID,
                    error_message="Token does not have sufficient permissions"
                )
            else:
                return AuthenticationResult(
                    status=AuthStatus.NETWORK_ERROR,
                    error_message=f"HTTP error {e.response.status_code}: {str(e)}"
                )
        except requests.exceptions.ConnectionError:
            return AuthenticationResult(
                status=AuthStatus.NETWORK_ERROR,
                error_message="Network connection error. Please check your internet connection."
            )
        except Exception as e:
            return AuthenticationResult(
                status=AuthStatus.NETWORK_ERROR,
                error_message=f"Unexpected error during token validation: {str(e)}"
            )
    
    def authenticate(self, token: Optional[str] = None, force_reauth: bool = False) -> AuthenticationResult:
        """
        Authenticate with HuggingFace with comprehensive error handling.
        
        Args:
            token: Optional token to use. If None, will search for token in various sources
            force_reauth: Force re-authentication even if cached credentials exist
            
        Returns:
            AuthenticationResult with authentication status
        """
        current_time = time.time()
        
        # Check cache if not forcing re-authentication
        if not force_reauth and (current_time - self._auth_cache["last_check"]) < self._cache_ttl:
            if self._auth_cache["status"] == AuthStatus.AUTHENTICATED:
                self.logger.debug("Using cached authentication")
                return AuthenticationResult(
                    status=self._auth_cache["status"],
                    user_info=self._auth_cache["user_info"],
                    token_valid=True
                )
        
        # Get token from sources
        if not token:
            token = self.get_token_from_sources()
        
        if not token:
            error_msg = (
                "No HuggingFace token found. Please provide a token through:\n"
                "1. Environment variable: export HF_TOKEN=your_token_here\n"
                "2. HuggingFace CLI: huggingface-cli login\n"
                "3. Direct parameter: authenticate(token='your_token')\n"
                "Get a token at: https://huggingface.co/settings/tokens"
            )
            return AuthenticationResult(
                status=AuthStatus.NOT_AUTHENTICATED,
                error_message=error_msg
            )
        
        # Validate the token
        validation_result = self.validate_token(token)
        
        if validation_result.status != AuthStatus.AUTHENTICATED:
            return validation_result
        
        # Attempt to login with the validated token
        try:
            login(token=token, add_to_git_credential=True)
            
            # Update cache
            self._auth_cache.update({
                "last_check": current_time,
                "status": AuthStatus.AUTHENTICATED,
                "user_info": validation_result.user_info,
                "token_hash": hash(token)  # Store hash, not the actual token
            })
            
            self.logger.info(f"Authentication successful for user: {validation_result.user_info['name']}")
            return validation_result
            
        except Exception as e:
            error_msg = f"Login failed even with valid token: {str(e)}"
            self.logger.error(error_msg)
            return AuthenticationResult(
                status=AuthStatus.NETWORK_ERROR,
                error_message=error_msg
            )
    
    def check_model_access(self, model_id: str, token: Optional[str] = None) -> ModelAccessResult:
        """
        Check access to a specific model with detailed error reporting.
        
        Args:
            model_id: HuggingFace model ID (e.g., "google/gemma-2-2b-it")
            token: Optional token to use for checking
            
        Returns:
            ModelAccessResult with access details
        """
        # Check cache first
        cache_key = f"{model_id}_{hash(token) if token else 'default'}"
        current_time = time.time()
        
        if cache_key in self._model_cache:
            cached_result, cache_time = self._model_cache[cache_key]
            if (current_time - cache_time) < self._cache_ttl:
                self.logger.debug(f"Using cached access result for {model_id}")
                return cached_result
        
        # Get token if not provided
        if not token:
            token = self.get_token_from_sources()
        
        try:
            api = HfApi(token=token)
            model_info = api.model_info(model_id, token=token)
            
            result = ModelAccessResult(
                has_access=True,
                model_id=model_id,
                is_gated=getattr(model_info, 'gated', False),
                license_info=getattr(model_info, 'license', 'Unknown')
            )
            
            # Cache the successful result
            self._model_cache[cache_key] = (result, current_time)
            
            self.logger.info(f"Successfully verified access to {model_id}")
            if result.is_gated:
                self.logger.info(f"Model {model_id} is gated (license: {result.license_info})")
            
            return result
            
        except HfHubHTTPError as e:
            suggestions = []
            
            if e.response.status_code == 401:
                error_msg = "Authentication required or invalid token"
                suggestions = [
                    "Ensure you have a valid HuggingFace token",
                    "Set your token: export HF_TOKEN=your_token_here",
                    "Login with CLI: huggingface-cli login"
                ]
            elif e.response.status_code == 403:
                error_msg = "Access denied - you may need to accept the model license"
                suggestions = [
                    f"Visit https://huggingface.co/{model_id} to accept the license",
                    "Ensure your token has the correct permissions",
                    "Wait a few minutes after accepting the license for permissions to propagate"
                ]
            elif e.response.status_code == 404:
                error_msg = "Model not found"
                suggestions = [
                    f"Verify the model ID '{model_id}' is correct",
                    "Check if the model has been moved or renamed",
                    "Ensure you have access to private models if applicable"
                ]
            else:
                error_msg = f"HTTP error {e.response.status_code}: {str(e)}"
                suggestions = ["Check your internet connection and try again"]
            
            result = ModelAccessResult(
                has_access=False,
                model_id=model_id,
                error_message=error_msg,
                suggestions=suggestions
            )
            
            # Cache failed results for a shorter time
            self._model_cache[cache_key] = (result, current_time - self._cache_ttl + 60)  # Cache for 1 minute
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error checking model access: {str(e)}"
            result = ModelAccessResult(
                has_access=False,
                model_id=model_id,
                error_message=error_msg,
                suggestions=["Check your internet connection and try again"]
            )
            return result
    
    def check_gemma_access(self) -> Tuple[bool, List[str]]:
        """
        Specifically check access to Gemma models.
        
        Returns:
            Tuple of (has_access, accessible_models)
        """
        gemma_models = [
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it"
        ]
        
        accessible_models = []
        
        for model_id in gemma_models:
            access_result = self.check_model_access(model_id)
            if access_result.has_access:
                accessible_models.append(model_id)
                self.logger.info(f"✅ Access confirmed: {model_id}")
            else:
                self.logger.warning(f"❌ No access to {model_id}: {access_result.error_message}")
                if access_result.suggestions:
                    for suggestion in access_result.suggestions:
                        self.logger.info(f"   💡 {suggestion}")
        
        has_any_access = len(accessible_models) > 0
        
        if not has_any_access:
            self.logger.error("No access to any Gemma models")
            self.logger.info("To access Gemma models:")
            self.logger.info("1. Get a HuggingFace token: https://huggingface.co/settings/tokens")
            self.logger.info("2. Accept Gemma license: https://huggingface.co/google/gemma-2-2b-it")
            self.logger.info("3. Set token: export HF_TOKEN=your_token_here")
        
        return has_any_access, accessible_models
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """
        Get comprehensive authentication status.
        
        Returns:
            Dictionary with authentication details
        """
        current_time = time.time()
        
        # Check if we have cached authentication
        if (current_time - self._auth_cache["last_check"]) < self._cache_ttl:
            if self._auth_cache["status"] == AuthStatus.AUTHENTICATED:
                return {
                    "authenticated": True,
                    "user_info": self._auth_cache["user_info"],
                    "cache_age": current_time - self._auth_cache["last_check"],
                    "token_available": self.get_token_from_sources() is not None
                }
        
        # Fresh authentication check
        auth_result = self.authenticate()
        
        return {
            "authenticated": auth_result.status == AuthStatus.AUTHENTICATED,
            "user_info": auth_result.user_info,
            "status": auth_result.status.value,
            "error_message": auth_result.error_message,
            "token_available": self.get_token_from_sources() is not None
        }
    
    def setup_complete_auth_flow(self) -> bool:
        """
        Complete authentication flow for the benchmarking suite.
        
        Returns:
            True if fully authenticated and can access models, False otherwise
        """
        self.logger.info("🔐 Starting comprehensive authentication setup...")
        
        # Step 1: Basic HuggingFace authentication
        auth_result = self.authenticate()
        if auth_result.status != AuthStatus.AUTHENTICATED:
            self.logger.error(f"❌ Authentication failed: {auth_result.error_message}")
            return False
        
        self.logger.info(f"✅ Authenticated as: {auth_result.user_info['name']}")
        
        # Step 2: Check Gemma access specifically
        has_gemma_access, accessible_models = self.check_gemma_access()
        
        if not has_gemma_access:
            self.logger.error("❌ Cannot access any Gemma models")
            return False
        
        self.logger.info(f"✅ Gemma access confirmed for {len(accessible_models)} models")
        for model in accessible_models:
            self.logger.info(f"   📦 {model}")
        
        # Step 3: Show final status
        status = self.get_authentication_status()
        self.logger.info("🎯 Authentication setup complete!")
        self.logger.info(f"   👤 User: {status['user_info']['name']}")
        self.logger.info(f"   📧 Email: {status['user_info'].get('email', 'N/A')}")
        
        return True


# Global instance for convenience (lazy initialization)
_auth_manager: Optional[SecureAuthManager] = None


def get_auth_manager() -> SecureAuthManager:
    """Get the global authentication manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SecureAuthManager()
    return _auth_manager


# Convenience functions for backward compatibility
def ensure_authenticated() -> bool:
    """Ensure user is authenticated. Returns True if successful."""
    try:
        auth_manager = get_auth_manager()
        return auth_manager.setup_complete_auth_flow()
    except Exception as e:
        logging.getLogger("gemma_benchmark.auth").error(f"Authentication failed: {e}")
        return False


def check_model_access(model_id: str) -> bool:
    """Check if user has access to a specific model."""
    try:
        auth_manager = get_auth_manager()
        result = auth_manager.check_model_access(model_id)
        return result.has_access
    except Exception as e:
        logging.getLogger("gemma_benchmark.auth").error(f"Model access check failed: {e}")
        return False


def setup_huggingface_auth(token: Optional[str] = None) -> bool:
    """Setup HuggingFace authentication (backward compatibility)."""
    try:
        auth_manager = get_auth_manager()
        result = auth_manager.authenticate(token=token)
        return result.status == AuthStatus.AUTHENTICATED
    except Exception as e:
        logging.getLogger("gemma_benchmark.auth").error(f"Authentication setup failed: {e}")
        return False