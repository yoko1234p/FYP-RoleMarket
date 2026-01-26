"""
Meme-to-Character Adapter - 智能轉化 Meme 關鍵字為角色描述

將 meme 關鍵字（如 "chill guy"）透過 GPT 智能轉化為適合特定角色的描述。
關鍵：移除物種特定詞彙（guy, dog, cat），保留文化情緒（chill, vibe）。

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0 - Intelligent Meme Adaptation
"""

import os
import logging
from typing import Dict, List
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemeToCharacterAdapter:
    """
    智能轉化 Meme 關鍵字為角色描述。

    核心功能：
    1. 接收 meme 關鍵字（如 "chill guy"）
    2. 理解 meme 的文化含義和情緒
    3. 移除物種特定詞彙（guy, dog, humanoid）
    4. 生成適合目標角色的描述片段
    """

    def __init__(self):
        """Initialize adapter with GPT API."""
        # Use existing GPT API configuration from .env
        api_key = os.getenv('GPT_API_FREE_KEY')
        base_url = os.getenv('GPT_API_FREE_BASE_URL')
        self.model = os.getenv('GPT_API_FREE_MODEL', 'gpt-3.5-turbo')

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"MemeToCharacterAdapter initialized with model: {self.model}")

    def adapt_meme_to_character(
        self,
        meme_keyword: str,
        character_name: str,
        character_species: str,
        meme_context: Dict = None
    ) -> Dict:
        """
        將 meme 關鍵字智能轉化為適合角色的描述。

        Args:
            meme_keyword: Meme 關鍵字（如 "chill guy"）
            character_name: 角色名稱（如 "Lulu豬"）
            character_species: 角色物種（如 "piglet"）
            meme_context: Meme 的文化背景資訊（optional）

        Returns:
            Dictionary with:
                - adapted_description: 轉化後的描述
                - extracted_emotion: 提取的情緒
                - removed_terms: 移除的詞彙
                - cultural_essence: 文化精髓
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"智能轉化 Meme: '{meme_keyword}' → {character_name}")
        logger.info(f"{'='*60}\n")

        # Build prompt for GPT
        system_prompt = """You are an expert in cultural trends and character design.
Your task is to intelligently adapt meme keywords into character descriptions.

CRITICAL RULES:
1. REMOVE species-specific terms (guy, dog, cat, humanoid, etc.)
2. PRESERVE cultural essence and emotional vibe
3. ADAPT the description to fit the target character's species
4. Output ONLY the adapted description suitable for image generation prompts
5. Focus on MOOD, ATMOSPHERE, and EMOTION rather than physical features
"""

        user_prompt = f"""Meme Keyword: "{meme_keyword}"
Target Character: {character_name} (a {character_species})

Meme Cultural Context:
{meme_context.get('cultural_meaning', 'N/A') if meme_context else 'N/A'}

Meme Mood/Emotion:
{meme_context.get('mood', 'N/A') if meme_context else 'N/A'}

Task:
1. Identify the core EMOTION/VIBE of this meme (remove "guy" or any species terms)
2. Adapt it into a description suitable for {character_name}
3. Output format: "embodying [emotion/vibe], radiating [atmosphere], maintaining [demeanor]"

Example:
Input: "chill guy"
Output: "embodying a chill, unbothered attitude, radiating calm nonchalance, maintaining a relaxed demeanor with 'my life is going ok' vibe"

Now adapt: "{meme_keyword}" for {character_name}
Output ONLY the adapted description (no explanation):"""

        try:
            # Call GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            adapted_description = response.choices[0].message.content.strip()

            # Extract information
            result = {
                'meme_keyword': meme_keyword,
                'character_name': character_name,
                'adapted_description': adapted_description,
                'extracted_emotion': self._extract_emotion(meme_keyword),
                'removed_terms': self._identify_removed_terms(meme_keyword),
                'cultural_essence': meme_context.get('cultural_meaning', 'N/A') if meme_context else 'N/A',
                'original_mood': meme_context.get('mood', 'N/A') if meme_context else 'N/A'
            }

            # Display results
            logger.info(f"✅ 轉化成功！")
            logger.info(f"\n原始 Meme: {meme_keyword}")
            logger.info(f"目標角色: {character_name} ({character_species})")
            logger.info(f"\n移除詞彙: {', '.join(result['removed_terms'])}")
            logger.info(f"提取情緒: {result['extracted_emotion']}")
            logger.info(f"\n轉化描述:")
            logger.info(f'"{adapted_description}"')
            logger.info(f"\n{'='*60}\n")

            return result

        except Exception as e:
            logger.error(f"❌ GPT API 調用失敗: {e}")
            return None

    def _extract_emotion(self, meme_keyword: str) -> str:
        """提取 meme 關鍵字中的情緒詞。"""
        emotion_words = {
            'chill': 'relaxed, calm',
            'happy': 'joyful, cheerful',
            'sad': 'melancholic, down',
            'angry': 'frustrated, upset',
            'excited': 'energetic, enthusiastic',
            'tired': 'sleepy, low-energy',
            'unbothered': 'nonchalant, indifferent'
        }

        keyword_lower = meme_keyword.lower()
        for emotion, description in emotion_words.items():
            if emotion in keyword_lower:
                return description

        return 'neutral'

    def _identify_removed_terms(self, meme_keyword: str) -> List[str]:
        """識別需要移除的物種特定詞彙。"""
        species_terms = ['guy', 'dog', 'cat', 'owl', 'frog', 'humanoid', 'man', 'woman', 'person']

        removed = []
        keyword_lower = meme_keyword.lower()
        for term in species_terms:
            if term in keyword_lower:
                removed.append(term)

        return removed if removed else ['(none)']

    def generate_enhanced_prompt(
        self,
        base_character_description: str,
        scene_description: str,
        adapted_meme_description: str,
        style: str
    ) -> str:
        """
        生成完整的增強 prompt。

        Args:
            base_character_description: 基礎角色描述
            scene_description: 場景描述
            adapted_meme_description: 轉化後的 meme 描述
            style: 風格描述

        Returns:
            完整的 prompt
        """
        prompt = (
            f"{base_character_description} "
            f"{scene_description}, "
            f"{adapted_meme_description}, "
            f"{style}"
        )
        return prompt


def main():
    """測試 Meme-to-Character Adapter."""

    # Initialize adapter
    adapter = MemeToCharacterAdapter()

    # Load meme context from previous analysis
    import json
    meme_results_path = Path('data/trends_seasonal/meme_analysis_results.json')

    with open(meme_results_path, 'r', encoding='utf-8') as f:
        meme_data = json.load(f)

    # Get Chill Guy meme context
    chill_guy_meme = meme_data['detected_memes'][0]  # First is Chill Guy

    meme_context = {
        'cultural_meaning': chill_guy_meme['visual_features']['cultural_meaning'],
        'mood': chill_guy_meme['visual_features']['mood'],
        'visual_keywords': chill_guy_meme['visual_features']['visual_keywords']
    }

    # Test adaptation
    logger.info("測試案例：將 'chill guy' 轉化為適合 Lulu豬 的描述\n")

    result = adapter.adapt_meme_to_character(
        meme_keyword="chill guy",
        character_name="Lulu豬",
        character_species="piglet",
        meme_context=meme_context
    )

    if result:
        # Generate full prompt
        base_character = (
            "Lulu豬, chubby pastel piglet mascot, super-round head and torso, "
            "short stubby limbs, pill-shaped body, tiny feet and hands, "
            "soft velvet flocked surface, matte finish, no shine. "
            "eyes: very small bead-like black dots, slightly downturned, "
            "tired and listless, no catchlights, no reflections, no eyelids crease, "
            "wide eye spacing. "
            "expression: blank, calm, mildly sleepy, low energy, mouth absent. "
            "snout: small oval peach nose plate with two oval nostrils, soft edges. "
            "ears: short triangular, softly folded, pale pink with subtle gradient to warm pink at tips. "
            "cheeks: faint blush circles. "
            "color palette: milky pastel pink skin, peach snout, soft rose blush."
        )

        scene = (
            "in a festive Christmas scene with twinkling lights and wreaths, "
            "wearing a cozy Santa hat and scarf, surrounded by whimsical snowflakes "
            "and presents"
        )

        style = "kawaii art style, vibrant colors, soft lighting"

        enhanced_prompt = adapter.generate_enhanced_prompt(
            base_character_description=base_character,
            scene_description=scene,
            adapted_meme_description=result['adapted_description'],
            style=style
        )

        logger.info(f"{'='*60}")
        logger.info("生成的完整 Prompt")
        logger.info(f"{'='*60}\n")
        logger.info(f'"{enhanced_prompt}"')
        logger.info(f"\n字數: {len(enhanced_prompt.split())} words")
        logger.info(f"{'='*60}\n")

        # Save result
        output = {
            'adaptation_result': result,
            'enhanced_prompt': enhanced_prompt,
            'word_count': len(enhanced_prompt.split())
        }

        output_dir = Path('data/prompts_test')
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / 'meme_adapted_prompt.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 結果已儲存: {output_file}")


if __name__ == '__main__':
    main()
