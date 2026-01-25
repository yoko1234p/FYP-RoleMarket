"""
Prompt Variation Generator - 生成多樣化的 Prompt 變化

支援三種模式：
1. Single Mode: 同一個 prompt，微小變化（原有模式）
2. Preset Scenes: 預設場景配置（基於主題庫）
3. AI Creative: 使用 Gemini LLM 自動生成場景變化

Author: Developer (James)
Date: 2026-01-25
Version: 1.0
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 預設場景庫（Preset Scenes Mode）
SCENE_LIBRARY = {
    "Christmas": {
        "name": "聖誕節",
        "scenes": [
            "indoor family gathering with Christmas tree and presents, warm lighting",
            "outdoor snowy scene with snowman and winter decorations",
            "opening presents by fireplace, cozy atmosphere",
            "decorating Christmas tree together, festive mood"
        ]
    },
    "Halloween": {
        "name": "萬聖節",
        "scenes": [
            "trick-or-treating in neighborhood with Halloween decorations",
            "pumpkin carving scene with jack-o'-lanterns",
            "costume party celebration with spooky decorations",
            "haunted house adventure with friendly ghosts"
        ]
    },
    "Chinese New Year": {
        "name": "春節",
        "scenes": [
            "indoor family reunion dinner with traditional dishes",
            "outdoor lion dance performance, festive atmosphere",
            "giving red envelopes, joyful celebration",
            "watching fireworks display, bright night sky"
        ]
    },
    "Valentine's Day": {
        "name": "情人節",
        "scenes": [
            "romantic dinner date with candles and roses",
            "exchanging gifts and chocolates, sweet moment",
            "holding heart-shaped balloons in a park",
            "writing love letters together, cozy setting"
        ]
    },
    "Easter": {
        "name": "復活節",
        "scenes": [
            "Easter egg hunt in garden with colorful eggs",
            "decorating Easter eggs with pastel colors",
            "Easter bunny costume celebration",
            "spring picnic with Easter basket"
        ]
    },
    "Thanksgiving": {
        "name": "感恩節",
        "scenes": [
            "family Thanksgiving dinner with turkey",
            "preparing pumpkin pie together in kitchen",
            "autumn harvest scene with corn and pumpkins",
            "giving thanks together, warm atmosphere"
        ]
    },
    "New Year": {
        "name": "新年",
        "scenes": [
            "countdown celebration with confetti",
            "fireworks display at midnight",
            "champagne toast with friends",
            "New Year party with festive decorations"
        ]
    },
    "Birthday": {
        "name": "生日",
        "scenes": [
            "birthday cake with candles, making a wish",
            "opening birthday presents with excitement",
            "birthday party with balloons and decorations",
            "blowing birthday candles with friends"
        ]
    },
    "Summer": {
        "name": "夏天",
        "scenes": [
            "beach scene with palm trees and ocean waves",
            "pool party with inflatable toys",
            "eating ice cream on a sunny day",
            "surfing adventure with bright sunshine"
        ]
    },
    "Winter": {
        "name": "冬天",
        "scenes": [
            "building snowman in snowy landscape",
            "ice skating on frozen lake",
            "drinking hot chocolate by fireplace",
            "skiing adventure in snowy mountains"
        ]
    },
    "Spring": {
        "name": "春天",
        "scenes": [
            "cherry blossom viewing with pink flowers",
            "spring picnic in blooming garden",
            "flying kite in spring breeze",
            "planting flowers together"
        ]
    },
    "Autumn": {
        "name": "秋天",
        "scenes": [
            "playing in fallen leaves",
            "apple picking in autumn orchard",
            "hiking in colorful autumn forest",
            "cozy reading with autumn colors"
        ]
    }
}

# 細微變化配置（用於 Single Mode 添加變化）
MICRO_VARIATIONS = {
    "angles": [
        "front view",
        "slightly angled view",
        "three-quarter view",
        "side profile view"
    ],
    "actions": [
        "standing cheerfully",
        "jumping with joy",
        "waving happily",
        "sitting comfortably"
    ],
    "atmospheres": [
        "bright and cheerful",
        "warm and cozy",
        "vibrant and energetic",
        "soft and gentle"
    ],
    "lighting": [
        "natural daylight",
        "golden hour lighting",
        "soft ambient lighting",
        "dramatic lighting"
    ]
}


class PromptVariationGenerator:
    """
    Prompt 變化生成器 - 支援三種生成模式

    Modes:
        1. single: 微小變化（原有模式）
        2. preset: 預設場景配置
        3. creative: AI 自動生成場景變化
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        初始化 PromptVariationGenerator.

        Args:
            gemini_api_key: Gemini API Key (for creative mode, optional)
        """
        load_dotenv()
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_OPENAI_API_KEY")

        if not self.gemini_api_key:
            logger.warning("⚠️ GEMINI_OPENAI_API_KEY not found - Creative mode will be unavailable")

        logger.info("PromptVariationGenerator initialized")

    def generate_variations(
        self,
        base_prompt: str,
        mode: str = "single",
        num_variations: int = 4,
        theme: Optional[str] = None,
        character_name: Optional[str] = None,
        character_desc: Optional[str] = None
    ) -> List[str]:
        """
        生成 Prompt 變化。

        Args:
            base_prompt: 基礎 prompt
            mode: 生成模式 ("single", "preset", "creative")
            num_variations: 變化數量 (1-4)
            theme: 主題（用於 preset/creative mode）
            character_name: 角色名稱（用於 creative mode）
            character_desc: 角色描述（用於 creative mode）

        Returns:
            List[str]: Prompt 變化列表

        Raises:
            ValueError: 如果參數無效

        Example:
            >>> gen = PromptVariationGenerator()
            >>> # Single mode
            >>> variations = gen.generate_variations(
            ...     base_prompt="Lulu Pig celebrating Christmas",
            ...     mode="single",
            ...     num_variations=4
            ... )
            >>> # Preset mode
            >>> variations = gen.generate_variations(
            ...     base_prompt="Lulu Pig in Christmas scene",
            ...     mode="preset",
            ...     theme="Christmas",
            ...     num_variations=4
            ... )
            >>> # Creative mode
            >>> variations = gen.generate_variations(
            ...     base_prompt="Lulu Pig celebrating Chinese New Year",
            ...     mode="creative",
            ...     theme="Chinese New Year",
            ...     character_name="Lulu Pig",
            ...     character_desc="Cute pink pig",
            ...     num_variations=4
            ... )
        """
        if num_variations < 1:
            raise ValueError("num_variations must be at least 1")
        if num_variations > 20:
            logger.warning(f"⚠️ num_variations={num_variations} is large, limiting to 20")
            num_variations = 20

        if mode not in ["single", "preset", "creative"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'single', 'preset', or 'creative'")

        logger.info(f"Generating {num_variations} variations in {mode} mode")

        if mode == "single":
            return self._generate_single_variations(base_prompt, num_variations)
        elif mode == "preset":
            if not theme:
                raise ValueError("theme is required for preset mode")
            return self._generate_preset_variations(base_prompt, theme, num_variations)
        elif mode == "creative":
            if not self.gemini_api_key:
                logger.warning("⚠️ GEMINI_OPENAI_API_KEY not found - Falling back to preset/single mode")
                if theme:
                    return self._generate_preset_variations(base_prompt, theme, num_variations)
                else:
                    return self._generate_single_variations(base_prompt, num_variations)
            if not theme:
                logger.warning("⚠️ theme not provided for creative mode - Falling back to single mode")
                return self._generate_single_variations(base_prompt, num_variations)
            return self._generate_creative_variations(
                base_prompt, theme, num_variations, character_name, character_desc
            )

    def _generate_single_variations(self, base_prompt: str, num_variations: int) -> List[str]:
        """
        Single Mode: 同一個 prompt，添加微小變化。

        Args:
            base_prompt: 基礎 prompt
            num_variations: 變化數量

        Returns:
            List[str]: Prompt 變化列表
        """
        variations = []

        for i in range(num_variations):
            # 隨機選擇微小變化
            angle = random.choice(MICRO_VARIATIONS["angles"])
            action = random.choice(MICRO_VARIATIONS["actions"])
            atmosphere = random.choice(MICRO_VARIATIONS["atmospheres"])
            lighting = random.choice(MICRO_VARIATIONS["lighting"])

            # 組合變化
            variation = f"{base_prompt}, {angle}, {action}, {atmosphere}, {lighting}"
            variations.append(variation)

        logger.info(f"✅ Generated {len(variations)} single mode variations")
        return variations

    def _generate_preset_variations(
        self,
        base_prompt: str,
        theme: str,
        num_variations: int
    ) -> List[str]:
        """
        Preset Mode: 從預設場景庫生成變化。

        Args:
            base_prompt: 基礎 prompt
            theme: 主題名稱
            num_variations: 變化數量

        Returns:
            List[str]: Prompt 變化列表
        """
        # 尋找主題（模糊匹配）
        theme_key = None
        for key in SCENE_LIBRARY.keys():
            if key.lower() in theme.lower() or theme.lower() in key.lower():
                theme_key = key
                break

        if not theme_key:
            logger.warning(f"⚠️ Theme '{theme}' not found in library, using single mode fallback")
            return self._generate_single_variations(base_prompt, num_variations)

        # 取得場景配置
        theme_config = SCENE_LIBRARY[theme_key]
        scenes = theme_config["scenes"]

        # 如果場景數量少於需求，重複使用
        if len(scenes) < num_variations:
            logger.warning(f"⚠️ Only {len(scenes)} scenes available, some will be reused")
            selected_scenes = scenes * (num_variations // len(scenes) + 1)
            selected_scenes = selected_scenes[:num_variations]
        else:
            # 隨機選擇場景（不重複）
            selected_scenes = random.sample(scenes, num_variations)

        # 組合成完整 prompt
        variations = []
        for scene in selected_scenes:
            # 從 base_prompt 提取角色描述（假設格式為 "Character description..."）
            variation = f"{base_prompt}, {scene}"
            variations.append(variation)

        logger.info(f"✅ Generated {len(variations)} preset variations from theme '{theme_key}'")
        return variations

    def _generate_creative_variations(
        self,
        base_prompt: str,
        theme: str,
        num_variations: int,
        character_name: Optional[str] = None,
        character_desc: Optional[str] = None
    ) -> List[str]:
        """
        Creative Mode: 使用 Gemini LLM 自動生成場景變化。

        Args:
            base_prompt: 基礎 prompt
            theme: 主題名稱
            num_variations: 變化數量
            character_name: 角色名稱
            character_desc: 角色描述

        Returns:
            List[str]: Prompt 變化列表
        """
        logger.info(f"Using Gemini LLM to generate {num_variations} creative variations...")

        # 構建 LLM prompt
        llm_prompt = f"""You are a creative AI assistant helping to generate diverse image prompts for character design.

**Task:** Generate {num_variations} different scene variations for the following character and theme.

**Character:** {character_name if character_name else 'A character'}
**Character Description:** {character_desc if character_desc else 'Cute character'}
**Theme:** {theme}
**Base Prompt:** {base_prompt}

**Requirements:**
1. Each variation should describe a DIFFERENT scene/scenario related to the theme
2. Keep the character description consistent
3. Make each scene visually distinct and interesting
4. Include details about: setting, action, mood, lighting
5. Each prompt should be suitable for image generation (Midjourney/Gemini style)

**Output Format:** Return ONLY a JSON array of {num_variations} strings, each being a complete prompt.

Example format:
[
  "Character in scene 1, action, mood, lighting",
  "Character in scene 2, action, mood, lighting",
  ...
]

Generate {num_variations} creative and diverse prompts now:"""

        try:
            # 調用 Gemini API（text generation）
            response = requests.post(
                "https://newapi.aisonnet.org/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.gemini_api_key}"
                },
                json={
                    "model": "gemini-2.0-flash-exp",  # 使用 text generation model
                    "messages": [
                        {"role": "user", "content": llm_prompt}
                    ],
                    "temperature": 0.9,  # 較高創意度
                    "max_tokens": 1000
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            # 提取 LLM 回應
            content = result['choices'][0]['message']['content']

            # 解析 JSON 陣列
            # 移除可能的 markdown 格式
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 解析 JSON
            variations = json.loads(content)

            if not isinstance(variations, list):
                raise ValueError("LLM response is not a JSON array")

            # 確保數量正確
            if len(variations) < num_variations:
                logger.warning(f"⚠️ LLM only generated {len(variations)} variations, expected {num_variations}")
                # 補足到需要的數量（重複最後一個）
                while len(variations) < num_variations:
                    variations.append(variations[-1])
            elif len(variations) > num_variations:
                variations = variations[:num_variations]

            logger.info(f"✅ Generated {len(variations)} creative variations using Gemini LLM")
            return variations

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Gemini API request failed: {e}")
            logger.warning("Falling back to preset mode...")
            return self._generate_preset_variations(base_prompt, theme, num_variations)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Failed to parse LLM response: {e}")
            logger.warning("Falling back to preset mode...")
            return self._generate_preset_variations(base_prompt, theme, num_variations)

        except Exception as e:
            logger.error(f"❌ Unexpected error in creative mode: {e}")
            logger.warning("Falling back to single mode...")
            return self._generate_single_variations(base_prompt, num_variations)

    def get_available_themes(self) -> Dict[str, str]:
        """
        取得所有可用主題（用於 preset mode）。

        Returns:
            Dict[str, str]: {theme_key: theme_name_chinese}
        """
        return {key: config["name"] for key, config in SCENE_LIBRARY.items()}

    def get_theme_scenes(self, theme: str) -> Optional[List[str]]:
        """
        取得特定主題的所有場景。

        Args:
            theme: 主題名稱

        Returns:
            Optional[List[str]]: 場景列表，如果主題不存在則返回 None
        """
        # 模糊匹配主題
        for key in SCENE_LIBRARY.keys():
            if key.lower() in theme.lower() or theme.lower() in key.lower():
                return SCENE_LIBRARY[key]["scenes"]

        return None


def demo():
    """Demo function to test PromptVariationGenerator."""
    print("\n" + "="*80)
    print("Prompt Variation Generator Demo")
    print("="*80 + "\n")

    # Initialize generator
    try:
        generator = PromptVariationGenerator()
        print("✅ Generator initialized\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    base_prompt = "Lulu Pig, chubby pastel piglet mascot, cute kawaii style"

    # Test 1: Single Mode
    print("-" * 80)
    print("Test 1: Single Mode (Micro Variations)")
    print("-" * 80)

    try:
        variations = generator.generate_variations(
            base_prompt=base_prompt,
            mode="single",
            num_variations=4
        )

        print(f"✅ Generated {len(variations)} variations:\n")
        for i, var in enumerate(variations, 1):
            print(f"{i}. {var[:100]}...")
        print()

    except Exception as e:
        print(f"❌ Error: {e}\n")

    # Test 2: Preset Mode
    print("-" * 80)
    print("Test 2: Preset Mode (Christmas Scenes)")
    print("-" * 80)

    try:
        variations = generator.generate_variations(
            base_prompt=base_prompt,
            mode="preset",
            theme="Christmas",
            num_variations=4
        )

        print(f"✅ Generated {len(variations)} variations:\n")
        for i, var in enumerate(variations, 1):
            print(f"{i}. {var[:100]}...")
        print()

    except Exception as e:
        print(f"❌ Error: {e}\n")

    # Test 3: Creative Mode
    print("-" * 80)
    print("Test 3: Creative Mode (AI Generated)")
    print("-" * 80)

    try:
        variations = generator.generate_variations(
            base_prompt=base_prompt,
            mode="creative",
            theme="Chinese New Year",
            character_name="Lulu Pig",
            character_desc="Cute pink pig with big eyes",
            num_variations=4
        )

        print(f"✅ Generated {len(variations)} variations:\n")
        for i, var in enumerate(variations, 1):
            print(f"{i}. {var[:100]}...")
        print()

    except Exception as e:
        print(f"❌ Error: {e}\n")

    # List available themes
    print("-" * 80)
    print("Available Themes")
    print("-" * 80)

    themes = generator.get_available_themes()
    print(f"Total: {len(themes)} themes\n")
    for key, name in themes.items():
        print(f"  • {key}: {name}")

    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo()
