"""
TwoStageGenerator - 兩階段圖像生成策略

Stage 1: 生成極簡基礎角色（minimal decorations）
Stage 2: 添加主題元素（controlled theme addition）

Author: Developer
Date: 2026-01-25
Version: 1.0
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TwoStageGenerator:
    """
    兩階段圖像生成器

    解決角色一致性問題：
    - 問題：單階段生成會添加過多裝飾（毛衣、眼鏡、書本等）
    - 目標：提升 CLIP 相似度從 0.66-0.70 至 0.75-0.85

    策略：
    1. Stage 1: 生成極簡基礎角色（高一致性）
    2. Stage 2: 使用 Stage 1 輸出作為 reference，添加主題元素
    """

    def __init__(
        self,
        gemini_client,
        validator,
        output_dir: str = "data/generated_images/two_stage"
    ):
        """
        初始化兩階段生成器

        Args:
            gemini_client: GeminiOpenAIImageClient 實例
            validator: CharacterFocusedValidator 實例
            output_dir: 輸出目錄
        """
        self.gemini_client = gemini_client
        self.validator = validator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("TwoStageGenerator initialized")

    def generate_stage1(
        self,
        character_prompt: str,
        reference_image_path: str,
        image_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: 生成極簡基礎角色

        Args:
            character_prompt: 角色描述
            reference_image_path: 原始參考圖路徑
            image_filename: 自定義檔名（可選）

        Returns:
            生成結果字典（包含 local_path, clip_similarity 等）
        """
        logger.info("=" * 80)
        logger.info("🎯 Stage 1: 生成極簡基礎角色")
        logger.info("=" * 80)

        # 構建 Stage 1 極簡 prompt（避免過度裝飾）
        stage1_prompt = (
            f"{character_prompt}, exactly as shown in reference image, "
            f"minimal style, simple clean background, "
            f"no extra decorations, no accessories, "
            f"focus on character appearance only, plain lighting"
        )

        logger.info(f"📝 Character Prompt: {character_prompt}")
        logger.info(f"🔧 Stage 1 Prompt: {stage1_prompt}")
        logger.info(f"📷 Reference Image: {reference_image_path}")

        # 使用 Gemini API 生成 Stage 1 圖片
        result = self.gemini_client.generate(
            prompt=stage1_prompt,
            reference_images=[reference_image_path],
            image_filename=image_filename
        )

        logger.info(f"✅ Stage 1 生成完成")
        logger.info(f"   Local Path: {result['local_path']}")
        logger.info(f"   Duration: {result.get('duration', 0):.2f}s")
        logger.info(f"   Cost: ${result.get('cost', 0)}")

        return result

    def generate_stage2(
        self,
        stage1_image_path: str,
        theme_elements: str,
        theme_description: str,
        image_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: 添加主題元素

        Args:
            stage1_image_path: Stage 1 生成的圖片路徑
            theme_elements: 主題元素描述（如 "Santa hat, gift box"）
            theme_description: 場景描述（如 "Christmas celebration"）
            image_filename: 自定義檔名（可選）

        Returns:
            生成結果字典（包含 local_path, clip_similarity 等）
        """
        raise NotImplementedError("generate_stage2 未實作")

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
