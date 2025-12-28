"""
Authentication manager for handling HuggingFace Hub access, especially for gated models like Gemma.
"""

import logging
import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

# HuggingFace Hub client
try:
    from huggingface_hub import HfApi, HfFolder, login
    from huggingface_hub.utils import HfHubHTTPError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class AccessResult(NamedTuple):
    """Result of a model access check."""

    has_access: bool
    error_message: Optional[str] = None
    suggestions: Optional[List[str]] = None


class AuthManager:
    """Manages authentication and access checks for HuggingFace models."""

    def __init__(self, cache_dir: str = ".cache"):
        self.logger = logging.getLogger("openevals.auth")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is not installed. Please run: pip install huggingface_hub"
            )

        self.api = HfApi()

    def get_token(self) -> Optional[str]:
        """
        Get HuggingFace token from environment variables or local cache.

        Returns:
            The HuggingFace token, if found.
        """
        # Priority: Environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            return token

        # Priority: HuggingFace's standard cache
        token = HfFolder.get_token()
        if token:
            return token

        return None

    def check_model_access(self, model_id: str) -> AccessResult:
        """
        Check if the current user has access to a specific model on the Hub.

        Args:
            model_id: The ID of the model to check (e.g., "google/gemma-2-9b-it")

        Returns:
            An AccessResult object indicating access status and providing helpful messages.
        """
        token = self.get_token()
        if not token:
            return AccessResult(
                has_access=False,
                error_message="HuggingFace token not found.",
                suggestions=[
                    "Set the HF_TOKEN environment variable.",
                    "Run `huggingface-cli login` in your terminal.",
                ],
            )

        try:
            self.api.model_info(model_id, token=token)
            return AccessResult(has_access=True)
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                return AccessResult(
                    has_access=False,
                    error_message=f"Unauthorized. Ensure you have accepted the license for {model_id}.",
                    suggestions=[
                        f"Accept the license at: https://huggingface.co/{model_id}",
                        "Ensure your HF_TOKEN has 'read' permissions.",
                    ],
                )
            elif e.response.status_code == 404:
                return AccessResult(
                    has_access=False,
                    error_message=f"Model not found: {model_id}. Check for typos.",
                )
            else:
                return AccessResult(
                    has_access=False,
                    error_message=f"HTTP Error {e.response.status_code}: {e.response.reason}",
                    suggestions=[
                        "Check your network connection and HuggingFace status."
                    ],
                )
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while checking model access: {e}"
            )
            return AccessResult(has_access=False, error_message=str(e))

    def check_gemma_access(
        self, model_patterns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check access to Gemma models with configurable model patterns.

        Args:
            model_patterns: Optional list of model ID patterns to check.
                            If None, uses default Gemma models.

        Returns:
            Tuple of (has_access, accessible_models)
        """
        if model_patterns is None:
            # Default Gemma models - can be extended or configured
            model_patterns = [
                "google/gemma-2-2b",
                "google/gemma-2-2b-it",
                "google/gemma-2-9b",
                "google/gemma-2-9b-it",
                "google/gemma-2-27b",
                "google/gemma-2-27b-it",
                # Legacy models
                "google/gemma-2b",
                "google/gemma-7b",
                "google/gemma-2b-it",
                "google/gemma-7b-it",
            ]

        accessible_models = []

        for model_pattern in model_patterns:
            # Note: Wildcard support is not fully implemented here
            if "*" in model_pattern:
                self.logger.warning(
                    f"Wildcard patterns not yet supported in access check: {model_pattern}"
                )
                continue

            access_result = self.check_model_access(model_pattern)
            if access_result.has_access:
                accessible_models.append(model_pattern)
                self.logger.info(f"✅ Access confirmed: {model_pattern}")
            else:
                self.logger.warning(
                    f"❌ No access to {model_pattern}: {access_result.error_message}"
                )
                if access_result.suggestions:
                    for suggestion in access_result.suggestions[:2]:
                        self.logger.info(f"   💡 {suggestion}")

        has_any_access = len(accessible_models) > 0

        if not has_any_access:
            self.logger.error("No access to any of the specified Gemma models.")
            self.logger.info("To access official Gemma models:")
            self.logger.info(
                "1. Get a HuggingFace token: https://huggingface.co/settings/tokens"
            )
            self.logger.info(
                "2. Accept a Gemma license, e.g., at: https://huggingface.co/google/gemma-2-9b-it"
            )
            self.logger.info(
                "3. Set the token in your environment: export HF_TOKEN=your_token_here"
            )

        return has_any_access, accessible_models


def setup_huggingface_auth():
    """Interactive script to help user set up HuggingFace authentication."""
    print("--- HuggingFace Authentication Setup ---")

    token = os.environ.get("HF_TOKEN")
    if token:
        print("✅ Found HF_TOKEN in environment variables.")
    else:
        print("🔹 HF_TOKEN environment variable not found.")
        try:
            p_token = input(
                "Enter your HuggingFace token (leave blank to skip): "
            ).strip()
            if p_token:
                login(token=p_token)
                print("✅ Token saved successfully.")
            else:
                print("Skipping token entry.")
        except Exception as e:
            print(f"❌ Error saving token: {e}")

    print("\n--- Checking Gemma Access ---")
    auth_manager = AuthManager()
    auth_manager.check_gemma_access()
    print("\nSetup complete.")
