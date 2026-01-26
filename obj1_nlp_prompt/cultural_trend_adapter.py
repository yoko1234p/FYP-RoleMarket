"""
文化趨勢智能轉化器（Cultural Trend Adapter）

將各種文化趨勢（Meme、節日、設計風格、社交媒體潮流等）
智能轉化為適合目標 IP 角色的描述。

擴展自 MemeToCharacterAdapter，支援更廣泛的文化趨勢類型。

Author: Product Manager (John)
Date: 2025-10-27
Version: 2.0 - Extended to support all cultural trends
"""

import os
from typing import Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CulturalTrendAdapter:
    """
    智能轉化各種文化趨勢為角色適配描述。

    支援的趨勢類型：
    - Meme 文化 (e.g., "chill guy", "woman yelling at cat")
    - 節日文化 (e.g., "Christmas", "Lunar New Year")
    - 設計風格 (e.g., "minimalism", "cyberpunk")
    - 社交媒體潮流 (e.g., "cottagecore", "dark academia")
    - 情緒氛圍 (e.g., "cozy vibes", "energetic mood")

    核心功能：
    1. 識別趨勢類型和文化內涵
    2. 提取核心情緒/視覺元素/氛圍
    3. 移除不適合目標角色的元素（物種、性別、年齡等）
    4. 生成適合目標角色的場景/服飾/姿勢/氛圍描述
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化文化趨勢轉化器。

        Args:
            api_key: OpenAI API key (defaults to env variable GPT_API_FREE_KEY or OPENAI_API_KEY)
            base_url: OpenAI API base URL (defaults to env variable GPT_API_FREE_BASE_URL)
        """
        load_dotenv()

        # Try GPT_API_FREE first, fallback to OPENAI_API_KEY
        self.api_key = api_key or os.getenv("GPT_API_FREE_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Set GPT_API_FREE_KEY or OPENAI_API_KEY in .env or pass as argument.")

        # Use base_url if provided
        self.base_url = base_url or os.getenv("GPT_API_FREE_BASE_URL")

        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.model = os.getenv("GPT_API_FREE_MODEL", "gpt-3.5-turbo")

        logger.info(f"CulturalTrendAdapter initialized (Model: {self.model})")

    def adapt_trend_to_character(
        self,
        trend_keyword: str,
        character_name: str,
        character_species: str,
        trend_context: Optional[Dict] = None,
        target_elements: Optional[List[str]] = None
    ) -> Dict:
        """
        將文化趨勢轉化為角色適配描述。

        Args:
            trend_keyword: 趨勢關鍵字 (e.g., "chill guy", "Christmas cozy vibes", "cyberpunk neon")
            character_name: 角色名稱 (e.g., "Lulu豬")
            character_species: 角色物種 (e.g., "pig", "cat", "bear")
            trend_context: 趨勢背景資訊（可選）
                - trend_type: 趨勢類型 (meme/holiday/design_style/social_media/mood)
                - description: 趨勢描述
                - visual_keywords: 視覺關鍵字列表
            target_elements: 要轉化的元素類型（可選）
                - ["emotion", "scene", "costume", "pose", "atmosphere", "props"]
                - 默認：["scene", "costume", "atmosphere"]

        Returns:
            Dictionary containing:
                - trend_keyword: 輸入的趨勢關鍵字
                - character_name: 角色名稱
                - trend_type: 識別的趨勢類型
                - adapted_description: 轉化後的完整描述
                - elements: 分解的元素描述
                    - emotion: 情緒描述（如有）
                    - scene: 場景描述
                    - costume: 服飾描述
                    - pose: 姿勢描述（如有）
                    - atmosphere: 氛圍描述
                    - props: 道具描述（如有）
                - removed_terms: 移除的不適用詞彙
                - cultural_essence: 文化本質說明

        Example:
            >>> adapter = CulturalTrendAdapter()
            >>> result = adapter.adapt_trend_to_character(
            ...     trend_keyword="chill guy",
            ...     character_name="Lulu豬",
            ...     character_species="pig"
            ... )
            >>> print(result['adapted_description'])
            "sporting a cozy hoodie and relaxed pose, surrounded by urban setting with calm atmosphere"
        """
        # 預設目標元素
        if target_elements is None:
            target_elements = ["scene", "costume", "atmosphere"]

        # 構建 GPT prompt
        system_prompt = """You are a cultural trend expert and creative director specializing in IP character design.

Your task is to adapt cultural trends (memes, holidays, design styles, social media trends, moods) into character-appropriate descriptions.

Key principles:
1. REMOVE species-specific terms (guy, girl, human, dog, cat, etc.)
2. EXTRACT core emotion, visual elements, and atmosphere
3. ADAPT to the target character's species and form
4. FOCUS on scene, costume, props, pose, and atmosphere that work for any character
5. MAINTAIN the cultural essence while making it universally applicable

Output format (JSON):
{
  "trend_type": "meme|holiday|design_style|social_media|mood",
  "cultural_essence": "Brief explanation of the trend's cultural meaning",
  "removed_terms": ["list", "of", "removed", "species-specific", "terms"],
  "elements": {
    "emotion": "emotional quality (if applicable)",
    "scene": "background and environment description",
    "costume": "clothing and accessories description",
    "pose": "body language and posture (if applicable)",
    "atmosphere": "mood and lighting description",
    "props": "objects and decorative elements (if applicable)"
  },
  "adapted_description": "Complete integrated description suitable for image generation"
}"""

        # 構建 user prompt
        context_info = ""
        if trend_context:
            context_info = f"\nTrend Context: {json.dumps(trend_context, ensure_ascii=False)}"

        target_elements_str = ", ".join(target_elements)

        user_prompt = f"""Trend Keyword: "{trend_keyword}"
Target Character: {character_name} (a {character_species})
Target Elements: {target_elements_str}{context_info}

Task:
1. Identify the trend type and cultural essence
2. Extract elements: {target_elements_str}
3. Remove species-specific terms (guy, girl, human, etc.)
4. Adapt the description to work for {character_name} (a {character_species})
5. Output in JSON format

Focus on creating a description that can be added to a reference image of {character_name}, changing only the specified elements while preserving character identity."""

        logger.info(f"Adapting trend: '{trend_keyword}' for {character_name}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # 添加元數據
            result['trend_keyword'] = trend_keyword
            result['character_name'] = character_name
            result['character_species'] = character_species
            result['target_elements'] = target_elements

            logger.info(f"✅ Adaptation successful")
            logger.info(f"   Trend Type: {result.get('trend_type', 'N/A')}")
            logger.info(f"   Removed Terms: {result.get('removed_terms', [])}")

            return result

        except Exception as e:
            logger.error(f"Failed to adapt trend: {e}")
            raise

    def batch_adapt(
        self,
        trends: List[str],
        character_name: str,
        character_species: str
    ) -> Dict[str, Dict]:
        """
        批量轉化多個文化趨勢。

        Args:
            trends: 趨勢關鍵字列表
            character_name: 角色名稱
            character_species: 角色物種

        Returns:
            Dictionary mapping trend_keyword -> adaptation result
        """
        results = {}
        for trend in trends:
            try:
                result = self.adapt_trend_to_character(
                    trend_keyword=trend,
                    character_name=character_name,
                    character_species=character_species
                )
                results[trend] = result
            except Exception as e:
                logger.error(f"Failed to adapt '{trend}': {e}")
                results[trend] = {'error': str(e)}

        return results


def demo():
    """Demo function to test Cultural Trend Adapter."""
    print("\n" + "="*80)
    print("Cultural Trend Adapter Demo")
    print("="*80 + "\n")

    # Initialize adapter
    try:
        adapter = CulturalTrendAdapter()
        print("✅ Adapter initialized\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set OPENAI_API_KEY in .env file")
        return

    # Test cases covering different trend types
    test_cases = [
        {
            "trend": "chill guy",
            "type": "Meme",
            "description": "Meme character with relaxed, unbothered attitude"
        },
        {
            "trend": "Christmas cozy vibes",
            "type": "Holiday Mood",
            "description": "Warm festive atmosphere with winter comfort"
        },
        {
            "trend": "cyberpunk neon",
            "type": "Design Style",
            "description": "Futuristic aesthetic with neon lights and tech elements"
        },
        {
            "trend": "cottagecore aesthetic",
            "type": "Social Media Trend",
            "description": "Rural pastoral lifestyle with vintage elements"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print("-" * 80)
        print(f"Test {i}: {test['type']}")
        print("-" * 80)
        print(f"Trend: {test['trend']}")
        print(f"Description: {test['description']}\n")

        try:
            result = adapter.adapt_trend_to_character(
                trend_keyword=test['trend'],
                character_name="Lulu豬",
                character_species="pig"
            )

            print(f"✅ Adaptation successful!")
            print(f"   Trend Type: {result.get('trend_type', 'N/A')}")
            print(f"   Cultural Essence: {result.get('cultural_essence', 'N/A')}")
            print(f"   Removed Terms: {result.get('removed_terms', [])}")
            print(f"\n   Adapted Description:")
            print(f"   {result.get('adapted_description', 'N/A')}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("="*80)
    print("Demo completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo()
