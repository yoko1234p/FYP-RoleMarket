"""
Google Gemini 2.5 Flash Image (Nano Banana) Client

Official Google AI Studio API client for image generation.

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0

API Documentation: https://ai.google.dev/gemini-api/docs/imagen
"""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai package not found. Install with: pip install google-genai"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleGeminiImageClient:
    """
    Client for Google Gemini 2.5 Flash Image (Nano Banana) API.

    Features:
    - Generate images from text prompts
    - Direct streaming response
    - Automatic image saving
    - Cost tracking

    Usage:
        >>> client = GoogleGeminiImageClient()
        >>> result = client.generate(
        ...     prompt="Lulu Pig celebrating Christmas",
        ...     image_filename="lulu_christmas.png"
        ... )
        >>> print(f"Image saved to: {result['local_path']}")
    """

    MODEL = "gemini-2.5-flash-image"  # Nano Banana
    COST_PER_IMAGE = 0.00  # Free tier available

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = 'data/generated_images'
    ):
        """
        Initialize Google Gemini Image client.

        Args:
            api_key: Google AI Studio API key (defaults to env variable GEMINI_API_KEY)
            output_dir: Directory to save downloaded images
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env or pass as argument.")

        self.client = genai.Client(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cost tracking
        self.images_generated = 0
        self.total_cost = 0.0

        logger.info(f"GoogleGeminiImageClient initialized (Model: {self.MODEL})")

    def generate(
        self,
        prompt: str,
        image_filename: Optional[str] = None,
        reference_images: Optional[list] = None,
        max_retries: int = 3,
        retry_delay: int = 45
    ) -> Dict[str, Any]:
        """
        Generate image using Google Gemini 2.5 Flash Image.

        Args:
            prompt: Text prompt for image generation
            image_filename: Custom filename for saved image (optional)
            reference_images: List of reference image paths for character consistency (optional)
            max_retries: Maximum number of retry attempts for quota errors (default: 3)
            retry_delay: Delay in seconds between retries (default: 45)

        Returns:
            Dictionary containing:
                - prompt: Input prompt
                - local_path: Local path to saved image
                - cost: Cost in USD
                - model: Model name used

        Example:
            >>> client = GoogleGeminiImageClient()
            >>> # Without reference
            >>> response = client.generate(
            ...     prompt="Lulu Pig wearing a Santa hat",
            ...     image_filename="lulu_santa.png"
            ... )
            >>> # With reference images
            >>> response = client.generate(
            ...     prompt="Generate a Halloween version of this character",
            ...     reference_images=["data/generated_images/lulu_christmas.png"],
            ...     image_filename="lulu_halloween.png"
            ... )
        """
        import time
        start_time = time.time()

        logger.info(f"Generating image with Google Gemini...")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Model: {self.MODEL}")
        if reference_images:
            logger.info(f"Reference images: {len(reference_images)} image(s)")

        # Retry loop for handling quota errors
        for attempt in range(max_retries):
            try:
                # Build request parts
                parts = []

                # Add reference images first if provided
                if reference_images:
                    for ref_path in reference_images:
                        ref_file = Path(ref_path)
                        if not ref_file.exists():
                            raise FileNotFoundError(f"Reference image not found: {ref_path}")

                        # Read image and convert to base64
                        with open(ref_file, 'rb') as f:
                            image_data = f.read()

                        # Detect mime type
                        mime_type = 'image/png' if ref_file.suffix.lower() == '.png' else 'image/jpeg'

                        # Add image part
                        parts.append(types.Part.from_bytes(
                            data=image_data,
                            mime_type=mime_type
                        ))

                # Add text prompt
                parts.append(types.Part.from_text(text=prompt))

                contents = [
                    types.Content(
                        role="user",
                        parts=parts,
                    ),
                ]

                generate_content_config = types.GenerateContentConfig(
                    response_modalities=[
                        "IMAGE",
                        "TEXT",
                    ],
                )

                # Generate and save image
                saved_path = None
                text_response = []

                for chunk in self.client.models.generate_content_stream(
                    model=self.MODEL,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if (
                        chunk.candidates is None
                        or chunk.candidates[0].content is None
                        or chunk.candidates[0].content.parts is None
                    ):
                        continue

                    part = chunk.candidates[0].content.parts[0]

                    # Handle image data
                    if part.inline_data and part.inline_data.data:
                        inline_data = part.inline_data
                        data_buffer = inline_data.data
                        file_extension = mimetypes.guess_extension(inline_data.mime_type) or '.png'

                        # Generate filename
                        if image_filename:
                            filename = image_filename
                            # Ensure extension
                            if not filename.endswith(file_extension):
                                filename = filename.rsplit('.', 1)[0] + file_extension
                        else:
                            import uuid
                            filename = f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}{file_extension}"

                        output_path = self.output_dir / filename

                        # Save image
                        with open(output_path, 'wb') as f:
                            f.write(data_buffer)

                        saved_path = output_path
                        logger.info(f"Image saved to: {output_path}")

                    # Handle text response
                    elif hasattr(chunk, 'text') and chunk.text:
                        text_response.append(chunk.text)

                # Success - break retry loop
                break

            except Exception as e:
                error_str = str(e)

                # Check if it's a quota error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Quota exceeded (Attempt {attempt + 1}/{max_retries})")
                        logger.info(f"等待 {retry_delay} 秒後重試...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"❌ 配額超限，已嘗試 {max_retries} 次")
                        raise
                else:
                    # Non-quota error, don't retry
                    logger.error(f"Failed to generate image: {e}")
                    raise

        duration = time.time() - start_time

        # Update cost tracking
        self.images_generated += 1
        self.total_cost += self.COST_PER_IMAGE

        result = {
            'prompt': prompt,
            'local_path': str(saved_path) if saved_path else None,
            'text_response': ''.join(text_response) if text_response else None,
            'cost': self.COST_PER_IMAGE,
            'model': self.MODEL,
            'duration': duration,
            'reference_images': reference_images
        }

        logger.info(f"Image generated in {duration:.2f}s (Cost: ${self.COST_PER_IMAGE})")

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
    """Demo function to test Google Gemini Image client."""
    print("\n" + "="*80)
    print("Google Gemini 2.5 Flash Image (Nano Banana) Client Demo")
    print("="*80 + "\n")

    # Initialize client
    try:
        client = GoogleGeminiImageClient()
        print("✅ Client initialized\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set GEMINI_API_KEY in .env file")
        return

    # Example: Simple text-to-image
    print("-" * 80)
    print("Example: Text-to-Image Generation")
    print("-" * 80)

    try:
        result = client.generate(
            prompt="A cute pink pig character wearing a Christmas Santa hat, kawaii style",
            image_filename="demo_gemini_pig.png"
        )

        print(f"✅ Image generated successfully!")
        print(f"   Local Path: {result['local_path']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Cost: ${result['cost']}")
        if result['text_response']:
            print(f"   Text Response: {result['text_response']}\n")

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
