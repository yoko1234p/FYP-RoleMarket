"""
Gemini 2.5 Flash Image Client (OpenAI-Compatible API)

使用 OpenAI-compatible API 格式的 Gemini 圖像生成客戶端。

Author: Developer (James)
Date: 2025-11-10
Version: 1.0

API Documentation: https://newapi.aisonnet.org/
"""

import base64
import os
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from dotenv import load_dotenv
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiOpenAIImageClient:
    """
    Client for Gemini 2.5 Flash Image using OpenAI-compatible API.

    Features:
    - Generate images from text prompts
    - Support reference images (image-to-image)
    - Aspect ratio control (16:9, 4:3, etc.)
    - Automatic image download and saving
    - Cost tracking

    Usage:
        >>> client = GeminiOpenAIImageClient()
        >>> result = client.generate(
        ...     prompt="Lulu Pig celebrating Christmas",
        ...     image_filename="lulu_christmas.png"
        ... )
        >>> print(f"Image saved to: {result['local_path']}")
    """

    API_URL = "https://newapi.aisonnet.org/v1/chat/completions"
    MODEL = "gemini-2.5-flash-image"  # 正式版
    MODEL_PREVIEW = "gemini-2.5-flash-image-preview"  # Preview 版

    # Supported aspect ratios
    ASPECT_RATIOS = ["2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

    COST_PER_IMAGE = 0.00  # Free tier available

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = 'data/generated_images',
        use_preview: bool = False
    ):
        """
        Initialize Gemini OpenAI Image client.

        Args:
            api_key: API key (defaults to env variable GEMINI_OPENAI_API_KEY)
            output_dir: Directory to save downloaded images
            use_preview: Use preview model (supports aspect ratio control)
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_OPENAI_API_KEY not found. Set it in .env or pass as argument.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = self.MODEL_PREVIEW if use_preview else self.MODEL

        # Cost tracking
        self.images_generated = 0
        self.total_cost = 0.0

        logger.info(f"GeminiOpenAIImageClient initialized (Model: {self.model})")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

    def _download_image_from_url(self, url: str, output_path: Path) -> None:
        """
        Download image from URL.

        Args:
            url: Image URL
            output_path: Local path to save image
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Image downloaded from URL to: {output_path}")

    def _save_base64_image(self, base64_data: str, output_path: Path) -> None:
        """
        Save base64 encoded image to file.

        Args:
            base64_data: Base64 encoded image data
            output_path: Local path to save image
        """
        # Remove data URL prefix if present
        if 'base64,' in base64_data:
            base64_data = base64_data.split('base64,')[1]

        image_bytes = base64.b64decode(base64_data)

        with open(output_path, 'wb') as f:
            f.write(image_bytes)

        logger.info(f"Image saved from base64 to: {output_path}")

    def _extract_image_url_from_markdown(self, text: str) -> Optional[str]:
        """
        Extract image URL from markdown format text.

        Args:
            text: Markdown text (e.g., "![image](https://example.com/image.png)")

        Returns:
            Extracted URL or None
        """
        # Match markdown image syntax: ![...](url)
        match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', text)
        if match:
            return match.group(1)

        # Match direct URL
        match = re.search(r'(https?://[^\s]+\.(?:png|jpg|jpeg|webp))', text)
        if match:
            return match.group(1)

        return None

    def generate(
        self,
        prompt: str,
        image_filename: Optional[str] = None,
        reference_images: Optional[List[str]] = None,
        aspect_ratio: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate image using Gemini OpenAI API.

        Args:
            prompt: Text prompt for image generation
            image_filename: Custom filename for saved image (optional)
            reference_images: List of reference image paths (optional)
            aspect_ratio: Image aspect ratio (e.g., "16:9", "4:3") - requires preview model
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing:
                - prompt: Input prompt
                - local_path: Local path to saved image
                - cost: Cost in USD
                - model: Model name used
                - duration: Generation time in seconds

        Example:
            >>> client = GeminiOpenAIImageClient()
            >>> # Text-to-image
            >>> result = client.generate(
            ...     prompt="A cute pig wearing a Santa hat",
            ...     image_filename="lulu_santa.png"
            ... )
            >>> # Image-to-image with reference
            >>> result = client.generate(
            ...     prompt="Based on this pig character, generate a new image wearing summer clothes",
            ...     reference_images=["data/reference_images/lulu_pig_ref_1.jpg"],
            ...     image_filename="lulu_summer.png"
            ... )
        """
        start_time = time.time()

        logger.info(f"Generating image with Gemini OpenAI API...")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Model: {self.model}")
        if reference_images:
            logger.info(f"Reference images: {len(reference_images)} image(s)")
        if aspect_ratio:
            logger.info(f"Aspect ratio: {aspect_ratio}")

        # Build message content
        content = []

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        # Add reference images if provided
        if reference_images:
            for ref_path in reference_images:
                ref_file = Path(ref_path)
                if not ref_file.exists():
                    raise FileNotFoundError(f"Reference image not found: {ref_path}")

                # Check if it's a URL or local file
                if str(ref_path).startswith('http'):
                    # Use URL directly
                    image_url = str(ref_path)
                else:
                    # Encode local file to base64
                    base64_data = self._encode_image_to_base64(str(ref_path))
                    # Detect mime type
                    mime_type = 'image/png' if ref_file.suffix.lower() == '.png' else 'image/jpeg'
                    image_url = f"data:{mime_type};base64,{base64_data}"

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })

        # Build request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }

        # Add aspect ratio configuration if specified (only for preview model)
        if aspect_ratio and aspect_ratio in self.ASPECT_RATIOS:
            if self.model == self.MODEL_PREVIEW:
                payload["extra_body"] = {
                    "imageConfig": {
                        "aspectRatio": aspect_ratio
                    }
                }
                # Add system message for aspect ratio
                payload["messages"].insert(0, {
                    "role": "system",
                    "content": f'{{"imageConfig": {{"aspectRatio": "{aspect_ratio}"}}}}'
                })
            else:
                logger.warning(f"Aspect ratio control requires preview model, ignoring aspect_ratio={aspect_ratio}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Retry loop
        saved_path = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()

                result_data = response.json()

                # Extract image data
                message = result_data['choices'][0]['message']
                content_text = message.get('content', '')
                images = message.get('images', [])

                # Try to get image from images array (base64)
                if images and len(images) > 0:
                    image_data = images[0].get('imageUrl', {}).get('url', '')

                    if image_data:
                        # Generate filename
                        if image_filename:
                            filename = image_filename
                            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                                filename = filename + '.png'
                        else:
                            import uuid
                            filename = f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"

                        output_path = self.output_dir / filename

                        # Save base64 image
                        self._save_base64_image(image_data, output_path)
                        saved_path = output_path

                # If no base64 image, try to extract URL from markdown content
                if not saved_path and content_text:
                    image_url = self._extract_image_url_from_markdown(content_text)

                    if image_url:
                        # Generate filename
                        if image_filename:
                            filename = image_filename
                            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                                filename = filename + '.png'
                        else:
                            import uuid
                            filename = f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"

                        output_path = self.output_dir / filename

                        # Download image from URL
                        self._download_image_from_url(image_url, output_path)
                        saved_path = output_path

                if saved_path:
                    break
                else:
                    raise ValueError("No image data found in API response")

            except requests.exceptions.RequestException as e:
                error_msg = str(e)

                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Request failed (Attempt {attempt + 1}/{max_retries}): {error_msg}")
                    logger.info(f"Retrying in 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    logger.error(f"❌ Failed after {max_retries} attempts")
                    raise

            except Exception as e:
                logger.error(f"Failed to generate image: {e}")
                raise

        duration = time.time() - start_time

        # Update cost tracking
        self.images_generated += 1
        self.total_cost += self.COST_PER_IMAGE

        result = {
            'prompt': prompt,
            'local_path': str(saved_path) if saved_path else None,
            'cost': self.COST_PER_IMAGE,
            'model': self.model,
            'duration': duration,
            'reference_images': reference_images,
            'aspect_ratio': aspect_ratio
        }

        logger.info(f"✅ Image generated in {duration:.2f}s (Cost: ${self.COST_PER_IMAGE})")

        return result

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost tracking summary.

        Returns:
            Dictionary with cost statistics
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
    """Demo function to test Gemini OpenAI Image client."""
    print("\n" + "="*80)
    print("Gemini 2.5 Flash Image Client (OpenAI-Compatible) Demo")
    print("="*80 + "\n")

    # Initialize client (use preview for testing)
    try:
        client = GeminiOpenAIImageClient(use_preview=True)
        print("✅ Client initialized (using preview model)\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set GEMINI_OPENAI_API_KEY in .env file")
        return

    # Example 1: Text-to-image
    print("-" * 80)
    print("Example 1: Text-to-Image Generation")
    print("-" * 80)

    try:
        result = client.generate(
            prompt="A cute pink pig character wearing a Christmas Santa hat, kawaii style",
            image_filename="demo_text_to_image.png"
        )

        print(f"✅ Image generated successfully!")
        print(f"   Local Path: {result['local_path']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Cost: ${result['cost']}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    # Example 2: Image-to-image with reference
    print("-" * 80)
    print("Example 2: Image-to-Image with Reference")
    print("-" * 80)

    try:
        # Use the image from Example 1 as reference
        if result['local_path']:
            result2 = client.generate(
                prompt="Based on this pig character, generate a new image of the same character wearing summer clothes and sunglasses",
                reference_images=[result['local_path']],
                image_filename="demo_image_to_image.png"
            )

            print(f"✅ Image generated successfully!")
            print(f"   Local Path: {result2['local_path']}")
            print(f"   Duration: {result2['duration']:.2f}s")
            print(f"   Cost: ${result2['cost']}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

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
