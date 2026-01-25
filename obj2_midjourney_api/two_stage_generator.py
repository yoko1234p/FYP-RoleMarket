"""
Two-Stage Image Generation Strategy

解決角色一致性問題的兩階段生成策略：
- Stage 1: 生成極簡基礎角色（高一致性）
- Stage 2: 添加主題元素（保持角色特徵）

Author: Developer
Date: 2026-01-25
Version: 1.0
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any
from PIL import Image

from obj2_midjourney_api.gemini_openai_client import GeminiOpenAIImageClient

logger = logging.getLogger(__name__)


class TwoStageGenerator:
    """
    Two-stage image generation for improved character consistency.

    Strategy:
    1. Stage 1: Generate minimal base character with high consistency
    2. Stage 2: Add theme elements using Stage 1 output as reference

    Expected improvement: CLIP Similarity 0.66-0.70 → 0.75-0.85

    Usage:
        >>> generator = TwoStageGenerator()
        >>> result = generator.generate_two_stage(
        ...     character_prompt="Lulu Pig",
        ...     reference_image_path="data/reference_images/lulu_pig_ref_1.jpg",
        ...     theme_elements="wearing Christmas sweater, holding a book",
        ...     theme_description="cozy Christmas indoor scene"
        ... )
        >>> print(f"Final image: {result['final_image_path']}")
        >>> print(f"CLIP Similarity: {result.get('clip_similarity', 'N/A')}")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TwoStageGenerator.

        Args:
            api_key: Gemini API key (optional, defaults to env variable)
        """
        # Use preview model for better control
        self.client = GeminiOpenAIImageClient(api_key=api_key, use_preview=True)

        # Output directory for intermediate results
        self.output_dir = Path("data/two_stage_generations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("TwoStageGenerator initialized")

    def generate_stage1(
        self,
        character_prompt: str,
        reference_image_path: str,
        output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Generate minimal base character with high consistency.

        Args:
            character_prompt: Character description (e.g., "Lulu Pig")
            reference_image_path: Path to reference image
            output_filename: Custom output filename (optional)

        Returns:
            Dictionary containing:
            {
                'image_path': str,
                'image': PIL.Image,
                'prompt_used': str,
                'generation_time': float,
                'success': bool,
                'error': str (if failed)
            }
        """
        logger.info("Stage 1: Generating base character")
        start_time = time.time()

        # 構建 Stage 1 極簡 prompt（避免過度裝飾）
        stage1_prompt = (
            f"{character_prompt}, exactly as shown in reference image, "
            f"minimal style, simple clean background, "
            f"no extra decorations, no accessories, "
            f"focus on character appearance only, plain lighting"
        )

        logger.info(f"Character prompt: {character_prompt}")
        logger.info(f"Stage 1 prompt: {stage1_prompt}")
        logger.info(f"Reference image: {reference_image_path}")

        try:
            # 調用 Gemini API
            api_result = self.client.generate(
                prompt=stage1_prompt,
                reference_images=[reference_image_path],
                image_filename=output_filename
            )

            # 載入 PIL Image
            image_path = api_result['local_path']
            image = Image.open(image_path).convert('RGB')

            generation_time = time.time() - start_time

            logger.info(f"Stage 1 generation completed in {generation_time:.2f}s")
            logger.info(f"Image saved to: {image_path}")

            return {
                'image_path': image_path,
                'image': image,
                'prompt_used': stage1_prompt,
                'generation_time': generation_time,
                'success': True
            }

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Stage 1 generation failed: {str(e)}")

            return {
                'image_path': None,
                'image': None,
                'prompt_used': stage1_prompt,
                'generation_time': generation_time,
                'success': False,
                'error': str(e)
            }

    def generate_stage2(
        self,
        stage1_result: Dict[str, Any],
        theme_elements: str,
        theme_description: str,
        output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Add theme elements while preserving character.

        Args:
            stage1_result: Result from generate_stage1()
            theme_elements: Theme elements (e.g., "Santa hat, gift box")
            theme_description: Scene description (e.g., "Christmas celebration")
            output_filename: Custom output filename (optional)

        Returns:
            Dictionary containing:
            {
                'image_path': str,
                'image': PIL.Image,
                'prompt_used': str,
                'generation_time': float,
                'success': bool,
                'error': str (if failed)
            }
        """
        logger.info("Stage 2: Adding theme elements")
        start_time = time.time()

        # 驗證 Stage 1 成功狀態
        if not stage1_result.get('success'):
            return {
                'image_path': None,
                'image': None,
                'prompt_used': '',
                'generation_time': 0.0,
                'success': False,
                'error': 'Stage 1 failed, cannot proceed to Stage 2'
            }

        stage1_image_path = stage1_result['image_path']

        # 構建 Stage 2 prompt（強調保持角色一致性）
        stage2_prompt = (
            f"Based on the character shown in the reference image, "
            f"keep the character appearance EXACTLY the same, "
            f"but add the following: {theme_elements}. "
            f"Scene setting: {theme_description}. "
            f"IMPORTANT: Do not change the character's face, body shape, or basic features."
        )

        logger.info(f"Theme elements: {theme_elements}")
        logger.info(f"Scene description: {theme_description}")
        logger.info(f"Stage 2 prompt: {stage2_prompt}")
        logger.info(f"Stage 1 reference: {stage1_image_path}")

        try:
            # 使用 Stage 1 圖片作為 reference，生成 Stage 2 圖片
            api_result = self.client.generate(
                prompt=stage2_prompt,
                reference_images=[stage1_image_path],
                image_filename=output_filename
            )

            # 載入 PIL Image
            image_path = api_result['local_path']
            image = Image.open(image_path).convert('RGB')

            generation_time = time.time() - start_time

            logger.info(f"Stage 2 generation completed in {generation_time:.2f}s")
            logger.info(f"Image saved to: {image_path}")

            return {
                'image_path': image_path,
                'image': image,
                'prompt_used': stage2_prompt,
                'generation_time': generation_time,
                'success': True
            }

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Stage 2 generation failed: {str(e)}")

            return {
                'image_path': None,
                'image': None,
                'prompt_used': stage2_prompt,
                'generation_time': generation_time,
                'success': False,
                'error': str(e)
            }

    def generate_two_stage(
        self,
        character_prompt: str,
        reference_image_path: str,
        theme_elements: str,
        theme_description: str,
        stage1_filename: Optional[str] = None,
        stage2_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        完整兩階段生成流程

        Args:
            character_prompt: 角色描述
            reference_image_path: 原始參考圖路徑
            theme_elements: 主題元素描述
            theme_description: 場景描述
            stage1_filename: Stage 1 檔名（可選）
            stage2_filename: Stage 2 檔名（可選）

        Returns:
            包含兩階段結果的字典
        """
        raise NotImplementedError("generate_two_stage 未實作")
