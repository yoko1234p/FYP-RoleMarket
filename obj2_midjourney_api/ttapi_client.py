"""
TTAPI Midjourney API Client

Python wrapper for TTAPI.io Midjourney API with support for:
- Character reference (--cref) for IP consistency
- Task status polling
- Image download
- Error handling with retries
- Cost tracking

Author: Product Manager (John)
Epic: 3 - Objective 2: Midjourney API Integration
Story: 3.1 - TTAPI Midjourney API Client

API Documentation: https://docs.ttapi.io/
"""

import requests
import time
from typing import Optional, Dict, Any, List
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTAPIClient:
    """
    Client for TTAPI Midjourney API.

    Features:
    - Submit /imagine prompts with character reference (--cref)
    - Poll task status until completion
    - Download generated images
    - Track API costs
    - Automatic retry with exponential backoff

    Usage:
        >>> client = TTAPIClient()
        >>> result = client.imagine(
        ...     prompt="Lulu Pig celebrating Halloween",
        ...     cref_urls=["https://example.com/lulu.png"],
        ...     cref_weight=100
        ... )
        >>> print(f"Image URL: {result['image_url']}")
    """

    BASE_URL = "https://api.ttapi.io/midjourney/v1"
    DEFAULT_TIMEOUT = 300  # 5 minutes
    POLL_INTERVAL = 10  # seconds
    MAX_RETRIES = 3
    COST_PER_IMAGE = 0.40  # USD (estimated)

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        output_dir: str = 'data/generated_images'
    ):
        """
        Initialize TTAPI client.

        Args:
            api_key: TTAPI API key (defaults to env variable TTAPI_API_KEY)
            timeout: Maximum wait time for task completion (seconds)
            output_dir: Directory to save downloaded images
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("TTAPI_API_KEY")
        if not self.api_key:
            raise ValueError("TTAPI_API_KEY not found. Set it in .env or pass as argument.")

        self.headers = {
            "TT-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cost tracking
        self.images_generated = 0
        self.total_cost = 0.0

        logger.info(f"TTAPIClient initialized (timeout: {timeout}s)")

    def imagine(
        self,
        prompt: str,
        cref_urls: Optional[List[str]] = None,
        cref_weight: int = 100,
        version: str = "6.1",
        aspect_ratio: str = "1:1",
        style: str = "raw",
        wait_for_completion: bool = True,
        save_image: bool = True,
        image_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate image using Midjourney /imagine command.

        Args:
            prompt: Text prompt for image generation
            cref_urls: List of character reference image URLs for --cref parameter
            cref_weight: Character reference weight (0-100, default 100)
            version: Midjourney version (default "6.1")
            aspect_ratio: Image aspect ratio (default "1:1")
            style: Midjourney style (default "raw")
            wait_for_completion: Whether to wait for task to complete (default True)
            save_image: Whether to download and save the generated image
            image_filename: Custom filename for saved image (optional)

        Returns:
            Dictionary containing:
                - task_id: TTAPI task ID
                - status: Task status
                - image_url: URL of generated image (if completed)
                - local_path: Local path if saved
                - cost: Estimated cost in USD
                - duration: Generation time in seconds (if waited)
                - full_prompt: Full prompt with parameters

        Example:
            >>> client = TTAPIClient()
            >>> response = client.imagine(
            ...     prompt="Lulu Pig wearing a Halloween costume",
            ...     cref_urls=["https://example.com/lulu_ref.jpg"],
            ...     cref_weight=100
            ... )
        """
        start_time = time.time()

        # Build full prompt with --cref if reference images provided
        full_prompt = prompt
        if cref_urls:
            cref_param = f" --cref {' '.join(cref_urls)} --cw {cref_weight}"
            full_prompt += cref_param

        # Add Midjourney parameters
        full_prompt += f" --ar {aspect_ratio} --v {version} --style {style}"

        logger.info(f"Submitting imagine task...")
        logger.info(f"Prompt: {full_prompt[:100]}...")

        # Submit task with retry logic
        task_id = self._submit_with_retry(full_prompt, version)
        logger.info(f"Task submitted: {task_id}")

        result = {
            'task_id': task_id,
            'status': 'pending',
            'full_prompt': full_prompt
        }

        # Wait for completion if requested
        if wait_for_completion:
            task_result = self._wait_for_task(task_id)
            result.update(task_result)

            duration = time.time() - start_time
            result['duration'] = duration

            # Download image if requested
            if save_image and result.get('image_url'):
                local_path = self._download_image(
                    result['image_url'],
                    task_id,
                    custom_filename=image_filename
                )
                result['local_path'] = str(local_path) if local_path else None

            # Update cost tracking
            self.images_generated += 1
            self.total_cost += self.COST_PER_IMAGE
            result['cost'] = self.COST_PER_IMAGE

            logger.info(f"Image generated in {duration:.2f}s (Cost: ${self.COST_PER_IMAGE})")

        return result

    def _submit_with_retry(self, prompt: str, version: str) -> str:
        """
        Submit imagine task with retry logic.

        Args:
            prompt: Full prompt with parameters
            version: Midjourney version

        Returns:
            Task ID string

        Raises:
            Exception: If submission fails after retries
        """
        payload = {
            "prompt": prompt,
            "version": version
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/imagine",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if data.get('success') and data.get('task_id'):
                    return data['task_id']
                else:
                    error_msg = data.get('error', 'Unknown error')
                    raise Exception(f"API error: {error_msg}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to submit task after {self.MAX_RETRIES} attempts: {e}")

    def _wait_for_task(self, task_id: str) -> Dict[str, Any]:
        """
        Wait for image generation to complete with polling.

        Args:
            task_id: Task ID from imagine() response

        Returns:
            Final task status with image URL

        Raises:
            TimeoutError: If generation exceeds timeout
            Exception: If task fails
        """
        elapsed = 0
        while elapsed < self.timeout:
            status = self.get_task_status(task_id)

            task_status = status.get("status", "unknown").lower()

            if task_status == "completed":
                logger.info(f"Task {task_id} completed")
                return status
            elif task_status == "failed":
                error_msg = status.get('error', 'Unknown error')
                raise Exception(f"Task {task_id} failed: {error_msg}")
            elif task_status in ['pending', 'processing', 'queued']:
                logger.info(f"Task {task_id} status: {task_status} ({elapsed:.0f}s)")
                time.sleep(self.POLL_INTERVAL)
                elapsed += self.POLL_INTERVAL
            else:
                logger.warning(f"Unknown task status: {task_status}")
                time.sleep(self.POLL_INTERVAL)
                elapsed += self.POLL_INTERVAL

        raise TimeoutError(f"Task {task_id} timed out after {self.timeout}s")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get current status of a task.

        Args:
            task_id: TTAPI task ID

        Returns:
            Status dictionary from API

        Raises:
            Exception: If status check fails
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
            raise Exception(f"Failed to get task status: {e}")

    def _download_image(
        self,
        image_url: str,
        task_id: str,
        custom_filename: Optional[str] = None
    ) -> Path:
        """
        Download generated image from URL.

        Args:
            image_url: URL of generated image
            task_id: TTAPI task ID
            custom_filename: Custom filename (optional)

        Returns:
            Path to downloaded image file

        Raises:
            Exception: If download fails
        """
        if custom_filename:
            filename = custom_filename
        else:
            # Extract extension from URL or default to .png
            ext = '.png'
            if '.' in image_url.split('/')[-1]:
                ext = '.' + image_url.split('.')[-1].split('?')[0]  # Remove query params
            filename = f"{task_id}{ext}"

        output_path = self.output_dir / filename

        logger.info(f"Downloading image to: {output_path}")

        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Image saved: {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download image: {e}")

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost tracking summary.

        Returns:
            Dictionary with cost statistics:
                - images_generated: Number of images
                - total_cost: Total cost in USD
                - avg_cost_per_image: Average cost per image
        """
        return {
            'images_generated': self.images_generated,
            'total_cost': round(self.total_cost, 2),
            'avg_cost_per_image': round(
                self.total_cost / self.images_generated if self.images_generated > 0 else 0,
                2
            )
        }


def demo():
    """
    Demo function to test TTAPI client.

    This function demonstrates:
    1. Client initialization
    2. Submitting a simple prompt
    3. Using character reference (--cref)
    4. Cost tracking

    Usage:
        python obj2_midjourney_api/ttapi_client.py
    """
    print("\n" + "="*80)
    print("TTAPI Midjourney Client Demo")
    print("="*80 + "\n")

    # Initialize client
    try:
        client = TTAPIClient()
        print("✅ Client initialized\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set TTAPI_API_KEY in .env file")
        return

    # Example 1: Simple prompt (without character reference)
    print("-" * 80)
    print("Example 1: Simple Prompt (No Character Reference)")
    print("-" * 80)

    try:
        result = client.imagine(
            prompt="Cute pink pig character, Halloween costume, holding pumpkin, kawaii style",
            wait_for_completion=True,
            save_image=True,
            image_filename="demo_simple.png"
        )

        print(f"✅ Image generated successfully!")
        print(f"   Task ID: {result['task_id']}")
        print(f"   Image URL: {result.get('image_url', 'N/A')}")
        print(f"   Local Path: {result.get('local_path', 'N/A')}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")
        print(f"   Cost: ${result.get('cost', 0)}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    # Example 2: With character reference (--cref)
    print("-" * 80)
    print("Example 2: With Character Reference (--cref)")
    print("-" * 80)

    # Note: Replace with actual Lulu Pig reference image URL
    cref_url = "https://example.com/lulu_pig_reference.png"

    print(f"Character Reference: {cref_url}")
    print("⚠️  Note: Replace with actual Lulu Pig reference URL for real testing\n")

    # Uncomment to test with real reference image
    # try:
    #     result = client.imagine(
    #         prompt="Lulu Pig celebrating Christmas, Santa hat, snow, gifts",
    #         cref_urls=[cref_url],
    #         cref_weight=100,
    #         wait_for_completion=True,
    #         save_image=True,
    #         image_filename="demo_with_cref.png"
    #     )
    #
    #     print(f"✅ Image generated with character reference!")
    #     print(f"   Task ID: {result['task_id']}")
    #     print(f"   Duration: {result.get('duration', 0):.2f}s\n")
    #
    # except Exception as e:
    #     print(f"❌ Error: {e}\n")

    # Cost summary
    print("-" * 80)
    print("Cost Summary")
    print("-" * 80)

    summary = client.get_cost_summary()
    print(f"Images Generated: {summary['images_generated']}")
    print(f"Total Cost: ${summary['total_cost']}")
    print(f"Avg Cost per Image: ${summary['avg_cost_per_image']}\n")

    print("="*80)
    print("Demo completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo()
