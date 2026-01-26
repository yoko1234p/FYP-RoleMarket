"""
Trends API Wrapper - Obj 1 Integration Layer

封裝 enhanced_trends_pipeline 為 Streamlit 友善的 API。

Author: Developer (James)
Date: 2025-11-06
Version: 1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj1_nlp_prompt.prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)


class TrendsAPIError(Exception):
    """Raised when trends extraction fails."""
    pass


class PromptGenerationError(Exception):
    """Raised when prompt generation fails."""
    pass


class TrendsAPIWrapper:
    """
    Wrapper for Obj 1 NLP Prompt Generation Pipeline.

    提供簡化的 API 供 Streamlit 使用。
    """

    def __init__(self, region: str = 'HK', lang: str = 'zh-TW'):
        """
        Initialize TrendsAPIWrapper.

        Args:
            region: Google Trends region code
            lang: Language code
        """
        self.region = region
        self.lang = lang

        # Initialize PromptGenerator
        template_path = PROJECT_ROOT / 'obj1_nlp_prompt' / 'templates' / 'prompt_template.txt'
        character_desc_path = PROJECT_ROOT / 'data' / 'character_descriptions' / 'lulu_pig.txt'

        self.prompt_generator = PromptGenerator(
            template_path=str(template_path),
            character_desc_path=str(character_desc_path)
        )

        logger.info(f"TrendsAPIWrapper initialized (region={region}, lang={lang})")

    def generate_prompt(
        self,
        character_name: str,
        character_desc: str,
        trend_keywords: List[str],
        max_retries: int = 3
    ) -> str:
        """
        生成設計 Prompt。

        Args:
            character_name: 角色名稱 (e.g., "Lulu Pig")
            character_desc: 角色描述 (e.g., "可愛粉紅豬")
            trend_keywords: 趨勢關鍵字列表 (e.g., ["春節", "紅色", "喜慶"])
            max_retries: 最大重試次數

        Returns:
            生成的 Prompt 字串

        Raises:
            PromptGenerationError: 當 Prompt 生成失敗時

        Example:
            >>> wrapper = TrendsAPIWrapper()
            >>> prompt = wrapper.generate_prompt(
            ...     character_name="Lulu Pig",
            ...     character_desc="可愛粉紅豬，大眼睛",
            ...     trend_keywords=["春節", "紅色", "喜慶"]
            ... )
            >>> print(prompt)
        """
        if not trend_keywords:
            raise ValueError("trend_keywords cannot be empty")

        character_info = {
            'character_name': character_name,
            'character_description': character_desc
        }

        # Use character_name as theme for now
        # In future, could be extracted from keywords
        theme = f"{character_name} design with trending elements"

        # Retry logic
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating prompt (attempt {attempt + 1}/{max_retries})...")

                # Call PromptGenerator with correct parameters
                prompt = self.prompt_generator.generate_prompt(
                    theme=theme,
                    keywords=trend_keywords,
                    variation_hint=""
                )

                logger.info("✅ Prompt generated successfully")
                return prompt

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All attempts failed
        raise PromptGenerationError(
            f"Prompt generation failed after {max_retries} attempts: {str(last_error)}"
        )

    def extract_keywords_simple(self, keywords_str: str) -> List[str]:
        """
        簡單的關鍵字提取（逗號分隔）。

        For Story 4.1, we simplify by directly using user input.
        Full Google Trends integration can be added in future stories.

        Args:
            keywords_str: 逗號分隔的關鍵字字串

        Returns:
            關鍵字列表

        Example:
            >>> wrapper.extract_keywords_simple("春節, 紅色, 喜慶")
            ['春節', '紅色', '喜慶']
        """
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
        return keywords
