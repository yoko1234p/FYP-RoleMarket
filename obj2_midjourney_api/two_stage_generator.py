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

        # Output directory for intermediate results (use absolute path for security)
        self.output_dir = Path("data/two_stage_generations").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("TwoStageGenerator initialized")

    def _validate_filename(self, filename: str) -> str:
        """
        Validate filename to prevent path traversal attacks.

        Args:
            filename: Filename to validate

        Returns:
            Safe filename

        Raises:
            ValueError: If filename contains path traversal attempts
        """
        # Remove any directory separators
        safe_filename = Path(filename).name

        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValueError(f"Invalid filename: {filename} (path traversal detected)")

        return safe_filename

    def generate_stage1(
        self,
        character_prompt: str,
        reference_image_path: str,
        output_filename: Optional[str] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Generate minimal base character with high consistency.

        Args:
            character_prompt: Character description (e.g., "Lulu Pig")
            reference_image_path: Path to reference image
            output_filename: Custom output filename (optional)
            template: Custom prompt template (optional, uses default if not provided)

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
        if template:
            # Use custom template (must include {character_prompt} placeholder)
            stage1_prompt = template.format(character_prompt=character_prompt)
        else:
            # Use default minimal template
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
            # Validate output filename if provided
            safe_filename = self._validate_filename(output_filename) if output_filename else None

            # 調用 Gemini API
            api_result = self.client.generate(
                prompt=stage1_prompt,
                reference_images=[reference_image_path],
                image_filename=safe_filename
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
            logger.error(f"Stage 1 generation failed: {str(e)}", exc_info=True)

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
        output_filename: Optional[str] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Add theme elements while preserving character.

        Args:
            stage1_result: Result from generate_stage1()
            theme_elements: Theme elements (e.g., "Santa hat, gift box")
            theme_description: Scene description (e.g., "Christmas celebration")
            output_filename: Custom output filename (optional)
            template: Custom prompt template (optional, uses default if not provided)

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
        if template:
            # Use custom template (must include {theme_elements} and {theme_description} placeholders)
            stage2_prompt = template.format(
                theme_elements=theme_elements,
                theme_description=theme_description
            )
        else:
            # Use default preservation template
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
            # Validate output filename if provided
            safe_filename = self._validate_filename(output_filename) if output_filename else None

            # 使用 Stage 1 圖片作為 reference，生成 Stage 2 圖片
            api_result = self.client.generate(
                prompt=stage2_prompt,
                reference_images=[stage1_image_path],
                image_filename=safe_filename
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
            logger.error(f"Stage 2 generation failed: {str(e)}", exc_info=True)

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
        base_filename: Optional[str] = None,
        stage1_template: Optional[str] = None,
        stage2_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full two-stage generation workflow.

        Workflow:
        1. Generate minimal base character (Stage 1)
        2. Add theme elements using Stage 1 as reference (Stage 2)

        Args:
            character_prompt: Character description (e.g., "Lulu Pig")
            reference_image_path: Path to original reference image
            theme_elements: Elements to add (e.g., "wearing sweater, holding book")
            theme_description: Scene description (e.g., "cozy indoor scene")
            base_filename: Base name for output files (optional)
            stage1_template: Custom Stage 1 prompt template with {character_prompt} placeholder (optional)
            stage2_template: Custom Stage 2 prompt template with {theme_elements} and {theme_description} placeholders (optional)

        Returns:
            Dictionary containing:
            {
                'stage1_image_path': str,
                'final_image_path': str,
                'final_image': PIL.Image,
                'total_time': float,
                'success': bool,
                'stage1_prompt': str,
                'stage2_prompt': str,
                'error': str (if failed)
            }
        """
        logger.info("=" * 80)
        logger.info("=== TWO-STAGE GENERATION WORKFLOW ===")
        logger.info("=" * 80)

        # Validate parameters
        if not character_prompt or not character_prompt.strip():
            raise ValueError("character_prompt cannot be empty")

        if not theme_elements or not theme_elements.strip():
            raise ValueError("theme_elements cannot be empty")

        if not theme_description or not theme_description.strip():
            raise ValueError("theme_description cannot be empty")

        # Validate reference image exists
        ref_path = Path(reference_image_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        workflow_start = time.time()

        # Generate filenames
        if not base_filename:
            timestamp = int(time.time())
            base_filename = f"two_stage_{timestamp}"

        # Validate filename for security
        safe_base_filename = self._validate_filename(base_filename)

        stage1_filename = f"{safe_base_filename}_stage1.png"
        stage2_filename = f"{safe_base_filename}_stage2_final.png"

        # Execute Stage 1
        logger.info("")
        stage1_result = self.generate_stage1(
            character_prompt=character_prompt,
            reference_image_path=reference_image_path,
            output_filename=stage1_filename,
            template=stage1_template
        )

        if not stage1_result['success']:
            total_time = time.time() - workflow_start
            logger.error("Workflow failed at Stage 1")
            return {
                'stage1_image_path': None,
                'final_image_path': None,
                'final_image': None,
                'total_time': total_time,
                'success': False,
                'stage1_prompt': stage1_result.get('prompt_used', ''),
                'stage2_prompt': '',
                'error': f"Stage 1 failed: {stage1_result.get('error', 'Unknown error')}"
            }

        # Execute Stage 2
        logger.info("")
        stage2_result = self.generate_stage2(
            stage1_result=stage1_result,
            theme_elements=theme_elements,
            theme_description=theme_description,
            output_filename=stage2_filename,
            template=stage2_template
        )

        total_time = time.time() - workflow_start

        if not stage2_result['success']:
            logger.error("Workflow failed at Stage 2")
            return {
                'stage1_image_path': stage1_result['image_path'],
                'final_image_path': None,
                'final_image': None,
                'total_time': total_time,
                'success': False,
                'stage1_prompt': stage1_result['prompt_used'],
                'stage2_prompt': stage2_result.get('prompt_used', ''),
                'error': f"Stage 2 failed: {stage2_result.get('error', 'Unknown error')}"
            }

        # Workflow complete
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TWO-STAGE WORKFLOW COMPLETE in {total_time:.2f}s")
        logger.info(f"   Stage 1 time: {stage1_result['generation_time']:.2f}s")
        logger.info(f"   Stage 2 time: {stage2_result['generation_time']:.2f}s")
        logger.info(f"   Stage 1 image: {stage1_result['image_path']}")
        logger.info(f"   Final image: {stage2_result['image_path']}")
        logger.info("=" * 80)

        return {
            'stage1_image_path': stage1_result['image_path'],
            'final_image_path': stage2_result['image_path'],
            'final_image': stage2_result['image'],
            'total_time': total_time,
            'success': True,
            'stage1_prompt': stage1_result['prompt_used'],
            'stage2_prompt': stage2_result['prompt_used']
        }
