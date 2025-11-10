"""
Design Generator API Wrapper - Obj 2 Integration Layer

封裝 Google Gemini Image Generation 和 CLIP Validation 為 Streamlit 友善的 API。

Author: Developer (James)
Date: 2025-11-06
Version: 1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
from PIL import Image
import time
import io

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj2_midjourney_api.google_gemini_client import GoogleGeminiImageClient
from obj2_midjourney_api.gemini_openai_client import GeminiOpenAIImageClient
from obj2_midjourney_api.character_focused_validator import CharacterFocusedValidator

logger = logging.getLogger(__name__)


class DesignGenerationError(Exception):
    """Raised when design generation fails."""
    pass


class CLIPValidationError(Exception):
    """Raised when CLIP validation fails."""
    pass


class DesignGeneratorWrapper:
    """
    Wrapper for Obj 2 Design Generation and CLIP Validation.

    提供簡化的 API 供 Streamlit 使用。
    """

    def __init__(self, api_key: Optional[str] = None, use_openai_api: bool = True):
        """
        Initialize DesignGeneratorWrapper.

        Args:
            api_key: API key (optional, defaults to env variable)
            use_openai_api: Use OpenAI-compatible API (default: True, uses GEMINI_OPENAI_API_KEY)
                           If False, uses official Google API (uses GEMINI_API_KEY)
        """
        # Initialize Gemini client
        try:
            if use_openai_api:
                # Use preview model for OpenAI API (supports all features)
                self.client = GeminiOpenAIImageClient(api_key=api_key, use_preview=True)
                logger.info("GeminiOpenAIImageClient initialized (preview model)")
            else:
                self.client = GoogleGeminiImageClient(api_key=api_key)
                logger.info("GoogleGeminiImageClient initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise DesignGenerationError(f"無法初始化 Gemini 客戶端：{str(e)}")

        # Initialize CLIP validator (lazy load)
        self._validator = None

        logger.info("DesignGeneratorWrapper initialized")

    @property
    def validator(self) -> CharacterFocusedValidator:
        """
        Lazy load CLIP validator (避免啟動時載入大模型).

        Returns:
            CharacterFocusedValidator instance
        """
        if self._validator is None:
            logger.info("Loading CLIP model...")
            self._validator = CharacterFocusedValidator()
            logger.info("CLIP model loaded")
        return self._validator

    def generate_single_design(
        self,
        prompt: str,
        reference_image_path: str,
        output_filename: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        生成單張設計圖。

        Args:
            prompt: 設計 Prompt
            reference_image_path: Reference Image 路徑
            output_filename: 輸出檔名 (optional)
            max_retries: 最大重試次數

        Returns:
            字典包含:
            {
                'image': PIL.Image,
                'image_path': str,
                'generation_time': float,
                'success': bool,
                'error': str (if failed)
            }

        Raises:
            DesignGenerationError: 當生成失敗時
        """
        start_time = time.time()

        try:
            logger.info(f"Generating design with prompt: {prompt[:50]}...")
            logger.info(f"Reference image: {reference_image_path}")

            # Call Google Gemini API
            result = self.client.generate(
                prompt=prompt,
                reference_images=[reference_image_path],
                image_filename=output_filename,
                max_retries=max_retries
            )

            generation_time = time.time() - start_time

            # Load generated image
            image_path = result['local_path']
            image = Image.open(image_path).convert('RGB')

            logger.info(f"✅ Design generated successfully in {generation_time:.2f}s")

            return {
                'image': image,
                'image_path': image_path,
                'generation_time': generation_time,
                'success': True
            }

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Design generation failed: {str(e)}")

            return {
                'image': None,
                'image_path': None,
                'generation_time': generation_time,
                'success': False,
                'error': str(e)
            }

    def compute_clip_similarity(
        self,
        generated_image: Image.Image,
        reference_image_path: str,
        strategy: str = "center_crop"
    ) -> float:
        """
        計算 CLIP 相似度。

        Args:
            generated_image: 生成的圖片 (PIL Image)
            reference_image_path: Reference Image 路徑
            strategy: CLIP 驗證策略 (center_crop, background_removal, original)

        Returns:
            相似度分數 (0.0 - 1.0)

        Raises:
            CLIPValidationError: 當計算失敗時
        """
        try:
            logger.info("Computing CLIP similarity...")

            # Save generated image temporarily for validator
            temp_path = PROJECT_ROOT / "data" / "temp" / "temp_generated.png"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            generated_image.save(temp_path)

            # Validate with strategy
            result = self.validator.validate_with_strategy(
                generated_image_path=str(temp_path),
                reference_image_path=reference_image_path,
                strategy=strategy
            )

            similarity = result['similarity']
            logger.info(f"CLIP similarity: {similarity:.4f}")

            # Cleanup temp file
            temp_path.unlink(missing_ok=True)

            return similarity

        except Exception as e:
            logger.error(f"CLIP validation failed: {str(e)}")
            raise CLIPValidationError(f"CLIP 驗證失敗：{str(e)}")

    def generate_designs(
        self,
        prompt: str,
        reference_image_path: str,
        num_images: int = 4,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_retries: int = 3
    ) -> List[Dict]:
        """
        生成多張設計圖並計算 CLIP 相似度。

        Args:
            prompt: 設計 Prompt
            reference_image_path: Reference Image 路徑
            num_images: 生成數量 (1-4)
            progress_callback: 進度回調函數 (progress: float, message: str)
            max_retries: 最大重試次數

        Returns:
            List[Dict]: [
                {
                    'image': PIL.Image,
                    'image_path': str,
                    'clip_similarity': float,
                    'generation_time': float,
                    'success': bool,
                    'error': str (if failed)
                },
                ...
            ]

        Example:
            >>> wrapper = DesignGeneratorWrapper()
            >>> results = wrapper.generate_designs(
            ...     prompt="Lulu Pig celebrating Spring Festival",
            ...     reference_image_path="data/reference_images/lulu_pig_ref_1.jpg",
            ...     num_images=4,
            ...     progress_callback=lambda p, m: print(f"{int(p*100)}%: {m}")
            ... )
            >>> for i, result in enumerate(results):
            ...     if result['success']:
            ...         print(f"Image {i+1}: CLIP = {result['clip_similarity']:.4f}")
        """
        if not 1 <= num_images <= 4:
            raise ValueError("num_images must be between 1 and 4")

        results = []
        successful_count = 0

        for i in range(num_images):
            # Update progress
            if progress_callback:
                progress = i / num_images
                message = f"正在生成第 {i+1}/{num_images} 張設計圖..."
                progress_callback(progress, message)

            # Generate filename
            timestamp = int(time.time())
            filename = f"design_{timestamp}_var{i+1}.png"

            # Generate design
            design_result = self.generate_single_design(
                prompt=prompt,
                reference_image_path=reference_image_path,
                output_filename=filename,
                max_retries=max_retries
            )

            if design_result['success']:
                # Compute CLIP similarity
                try:
                    similarity = self.compute_clip_similarity(
                        generated_image=design_result['image'],
                        reference_image_path=reference_image_path,
                        strategy="center_crop"
                    )
                    design_result['clip_similarity'] = similarity
                    successful_count += 1

                except CLIPValidationError as e:
                    logger.warning(f"CLIP validation failed for image {i+1}: {e}")
                    design_result['clip_similarity'] = 0.0
                    design_result['error'] = f"CLIP 驗證失敗：{str(e)}"

            results.append(design_result)

            # Update progress (generation complete)
            if progress_callback:
                progress = (i + 1) / num_images
                message = f"已完成 {i+1}/{num_images} 張設計圖"
                progress_callback(progress, message)

        # Final summary
        logger.info(f"✅ 生成完成：{successful_count}/{num_images} 張成功")

        return results

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """
        轉換 PIL Image 為 bytes (用於下載)。

        Args:
            image: PIL Image
            format: 圖片格式 (PNG, JPEG)

        Returns:
            圖片 bytes
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()

    def get_average_similarity(self, results: List[Dict]) -> float:
        """
        計算平均 CLIP 相似度。

        Args:
            results: generate_designs() 的回傳結果

        Returns:
            平均相似度 (0.0 - 1.0)
        """
        similarities = [
            r['clip_similarity']
            for r in results
            if r.get('success') and 'clip_similarity' in r
        ]

        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)
