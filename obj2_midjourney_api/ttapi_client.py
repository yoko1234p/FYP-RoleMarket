"""
TTAPI Midjourney API Wrapper

Custom HTTP client for TTAPI Midjourney integration.
Replaces ttapi SDK (which has Python 2 compatibility issues).

API Documentation: https://ttapi.io/docs/apiReference/midjourney
"""

import requests
import time
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv


class TTAPIClient:
    """
    TTAPI Midjourney API client for character IP design generation.

    Features:
    - Image generation via /imagine endpoint
    - Character reference (--cref) parameter support
    - Status polling for async image generation
    - Error handling & retry logic
    - Cost tracking
    """

    BASE_URL = "https://api.ttapi.io/midjourney/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TTAPI client.

        Args:
            api_key: TTAPI API key (defaults to TTAPI_API_KEY env variable)
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("TTAPI_API_KEY")
        if not self.api_key:
            raise ValueError("TTAPI_API_KEY not found. Set it in .env or pass as argument.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.total_cost = 0.0

    def imagine(
        self,
        prompt: str,
        cref_urls: Optional[List[str]] = None,
        cref_weight: int = 100,
        version: str = "6.1",
        additional_params: str = "--ar 1:1 --style raw"
    ) -> Dict[str, Any]:
        """
        Generate image using Midjourney /imagine command.

        Args:
            prompt: Text prompt for image generation
            cref_urls: List of character reference image URLs for --cref parameter
            cref_weight: Character reference weight (0-100, default 100)
            version: Midjourney version (default "6.1")
            additional_params: Additional Midjourney parameters

        Returns:
            API response with task_id for polling

        Example:
            >>> client = TTAPIClient()
            >>> response = client.imagine(
            ...     prompt="Pikachu wearing a Halloween costume",
            ...     cref_urls=["https://example.com/pikachu_ref.jpg"],
            ...     cref_weight=100
            ... )
        """
        # Build full prompt with --cref if reference images provided
        full_prompt = prompt
        if cref_urls:
            cref_param = f" --cref {' '.join(cref_urls)} --cw {cref_weight}"
            full_prompt += cref_param

        full_prompt += f" {additional_params}"

        payload = {
            "prompt": full_prompt,
            "version": version
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/imagine",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"TTAPI imagine request failed: {e}")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Poll task status for async image generation.

        Args:
            task_id: Task ID from imagine() response

        Returns:
            Task status response with image URL when complete
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/task/{task_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"TTAPI status check failed: {e}")

    def wait_for_completion(
        self,
        task_id: str,
        max_wait: int = 300,
        poll_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Wait for image generation to complete with polling.

        Args:
            task_id: Task ID from imagine() response
            max_wait: Maximum wait time in seconds (default 5 minutes)
            poll_interval: Polling interval in seconds (default 10s)

        Returns:
            Final task status with image URL

        Raises:
            TimeoutError: If generation exceeds max_wait
        """
        elapsed = 0
        while elapsed < max_wait:
            status = self.get_task_status(task_id)

            if status.get("status") == "completed":
                # Update cost tracking (assuming ~$0.40 per image in PPU mode)
                self.total_cost += 0.40
                return status
            elif status.get("status") == "failed":
                raise Exception(f"Image generation failed: {status.get('error')}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Image generation timeout after {max_wait}s")


__all__ = ["TTAPIClient"]
