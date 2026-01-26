"""
TTAPI Gemini Image Model Client

Python wrapper for TTAPI.io Gemini Image Model API with support for:
- Text-to-image generation
- Reference images (refer_images) for character consistency
- Multiple aspect ratios
- Direct response (no polling needed)
- Cost tracking

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0

API Documentation: https://docs-zh.mjapiapp.com/api/gemini-image-model
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


class GeminiImageClient:
    """
    Client for TTAPI Gemini Image Model API.

    Features:
    - Generate images from text prompts
    - Use reference images for character consistency (similar to Midjourney --cref)
    - Multiple aspect ratios (1:1, 16:9, 9:16, etc.)
    - Direct response (no polling required)
    - Automatic retry with exponential backoff
    - Cost tracking

    Usage:
        >>> client = GeminiImageClient()
        >>> result = client.generate(
        ...     prompt="Lulu Pig celebrating Christmas",
        ...     refer_images=["https://example.com/lulu.png"],
        ...     aspect_ratio="1:1"
        ... )
        >>> print(f"Image URL: {result['image_url']}")
    """

    BASE_URL = "https://api.ttapi.io/gemini/image/generate"
    DEFAULT_MODE = "gemini-2.5-flash-image"  # Stable version with aspect ratio support
    PREVIEW_MODE = "gemini-2.5-flash-image-preview"  # Preview version
    MAX_RETRIES = 3
    COST_PER_IMAGE = 0.10  # USD (estimated, adjust based on actual pricing)

    # Available aspect ratios (only for gemini-2.5-flash-image)
    ASPECT_RATIOS = [
        "1:1", "2:3", "3:2", "3:4", "4:3",
        "4:5", "5:4", "9:16", "16:9", "21:9"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = 'data/generated_images'
    ):
        """
        Initialize Gemini Image client.

        Args:
            api_key: TTAPI API key (defaults to env variable TTAPI_API_KEY)
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cost tracking
        self.images_generated = 0
        self.total_cost = 0.0

        logger.info(f"GeminiImageClient initialized")

    def generate(
        self,
        prompt: str,
        refer_images: Optional[List[str]] = None,
        mode: str = DEFAULT_MODE,
        aspect_ratio: str = "1:1",
        save_image: bool = True,
        image_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate image using Gemini Image Model.

        Args:
            prompt: Text prompt for image generation
            refer_images: List of reference image URLs for character consistency
                         (similar to Midjourney --cref)
            mode: Model version:
                  - "gemini-2.5-flash-image" (stable, with aspect ratio support)
                  - "gemini-2.5-flash-image-preview" (preview version)
            aspect_ratio: Image aspect ratio (only for gemini-2.5-flash-image):
                         1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
                         Default: "1:1"
            save_image: Whether to download and save the generated image
            image_filename: Custom filename for saved image (optional)

        Returns:
            Dictionary containing:
                - status: API status
                - message: API message
                - prompt: Input prompt
                - refer_images: Reference images used
                - image_url: URL of generated image
                - quota: Remaining quota
                - local_path: Local path if saved
                - cost: Estimated cost in USD
                - duration: Generation time in seconds

        Example:
            >>> client = GeminiImageClient()
            >>> response = client.generate(
            ...     prompt="Lulu Pig wearing a Santa hat",
            ...     refer_images=["https://example.com/lulu_ref.jpg"],
            ...     aspect_ratio="1:1"
            ... )
        """
        start_time = time.time()

        # Validate aspect ratio
        if mode == self.DEFAULT_MODE and aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(
                f"Invalid aspect_ratio: {aspect_ratio}. "
                f"Must be one of: {', '.join(self.ASPECT_RATIOS)}"
            )

        logger.info(f"Generating image with Gemini...")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Mode: {mode}")
        if refer_images:
            logger.info(f"Reference images: {len(refer_images)} images")
        logger.info(f"Aspect ratio: {aspect_ratio}")

        # Build payload
        payload = {
            "prompt": prompt,
            "mode": mode
        }

        # Add optional parameters
        if refer_images:
            payload["refer_images"] = refer_images

        if mode == self.DEFAULT_MODE:
            payload["aspect_ratio"] = aspect_ratio

        # Submit generation with retry logic
        response_data = self._generate_with_retry(payload)

        # Extract data
        data = response_data.get('data', {})
        image_url = data.get('image_url')

        result = {
            'status': response_data.get('status'),
            'message': response_data.get('message'),
            'prompt': data.get('prompt'),
            'refer_images': data.get('refer_images'),
            'image_url': image_url,
            'quota': data.get('quota'),
            'raw_response': response_data
        }

        duration = time.time() - start_time
        result['duration'] = duration

        # Download image if requested
        if save_image and image_url:
            local_path = self._download_image(
                image_url,
                custom_filename=image_filename
            )
            result['local_path'] = str(local_path) if local_path else None

        # Update cost tracking
        self.images_generated += 1
        self.total_cost += self.COST_PER_IMAGE
        result['cost'] = self.COST_PER_IMAGE

        logger.info(f"Image generated in {duration:.2f}s (Cost: ${self.COST_PER_IMAGE})")
        logger.info(f"Image URL: {image_url}")

        return result

    def _generate_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit generation request with retry logic.

        Args:
            payload: Request payload

        Returns:
            Response data dictionary

        Raises:
            Exception: If generation fails after retries
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=60  # Gemini may take longer
                )
                response.raise_for_status()
                data = response.json()

                # Check response status
                if data.get('status') == 'SUCCESS':
                    return data
                else:
                    error_msg = data.get('message', 'Unknown error')
                    logger.error(f"API response: {data}")
                    raise Exception(f"API error: {error_msg}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate image after {self.MAX_RETRIES} attempts: {e}")

    def _download_image(
        self,
        image_url: str,
        custom_filename: Optional[str] = None
    ) -> Path:
        """
        Download generated image from URL.

        Args:
            image_url: URL of generated image
            custom_filename: Custom filename (optional)

        Returns:
            Path to downloaded image file

        Raises:
            Exception: If download fails
        """
        if custom_filename:
            filename = custom_filename
        else:
            # Extract filename from URL or generate timestamp-based name
            import uuid
            ext = '.png'
            if '.' in image_url.split('/')[-1]:
                ext = '.' + image_url.split('.')[-1].split('?')[0]
            filename = f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"

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
    Demo function to test Gemini Image client.

    This function demonstrates:
    1. Client initialization
    2. Simple text-to-image generation
    3. Using reference images for character consistency
    4. Cost tracking

    Usage:
        python obj2_midjourney_api/gemini_image_client.py
    """
    print("\n" + "="*80)
    print("TTAPI Gemini Image Model Client Demo")
    print("="*80 + "\n")

    # Initialize client
    try:
        client = GeminiImageClient()
        print("✅ Client initialized\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set TTAPI_API_KEY in .env file")
        return

    # Example 1: Simple text-to-image
    print("-" * 80)
    print("Example 1: Simple Text-to-Image")
    print("-" * 80)

    try:
        result = client.generate(
            prompt="Cute pink pig character, Christmas scene, Santa hat, kawaii style",
            aspect_ratio="1:1",
            save_image=True,
            image_filename="demo_gemini_simple.png"
        )

        print(f"✅ Image generated successfully!")
        print(f"   Status: {result['status']}")
        print(f"   Image URL: {result['image_url']}")
        print(f"   Local Path: {result.get('local_path', 'N/A')}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Cost: ${result['cost']}")
        print(f"   Quota remaining: {result['quota']}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    # Example 2: With reference images
    print("-" * 80)
    print("Example 2: With Reference Images (Character Consistency)")
    print("-" * 80)

    # Note: Replace with actual Lulu Pig reference image URLs
    refer_urls = [
        "https://example.com/lulu_pig_reference.png"
    ]

    print(f"Reference images: {len(refer_urls)} images")
    print("⚠️  Note: Replace with actual Lulu Pig reference URLs for real testing\n")

    # Uncomment to test with real reference images
    # try:
    #     result = client.generate(
    #         prompt="Lulu Pig celebrating Christmas, Santa hat, snow, gifts, kawaii style",
    #         refer_images=refer_urls,
    #         aspect_ratio="1:1",
    #         save_image=True,
    #         image_filename="demo_gemini_with_ref.png"
    #     )
    #
    #     print(f"✅ Image generated with reference images!")
    #     print(f"   Duration: {result['duration']:.2f}s")
    #     print(f"   Quota remaining: {result['quota']}\n")
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
