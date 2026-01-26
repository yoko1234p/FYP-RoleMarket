"""
場景變化生成器 - 使用參考圖片生成多個季節/主題場景

使用已生成的聖誕圖片作為角色參考，生成不同場景變化：
- Halloween (萬聖節)
- Spring Festival (春節)
- Birthday (生日)
- Summer Vacation (夏日)

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from obj2_midjourney_api.google_gemini_client import GoogleGeminiImageClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 場景配置
SCENES = {
    'halloween': {
        'name': '萬聖節',
        'prompt_addition': (
            "in a Halloween scene with pumpkins, jack-o'-lanterns, and spooky decorations, "
            "wearing a cute witch hat and cape, surrounded by friendly ghosts and bats, "
            "kawaii art style, vibrant orange and purple colors, soft mysterious lighting"
        ),
        'filename': 'lulu_halloween.png'
    },
    'spring_festival': {
        'name': '春節',
        'prompt_addition': (
            "in a Chinese Spring Festival scene with red lanterns and golden decorations, "
            "wearing a traditional Chinese festive outfit with red and gold accents, "
            "surrounded by firecrackers, lucky coins, and plum blossoms, "
            "kawaii art style, vibrant red and gold colors, warm celebratory lighting"
        ),
        'filename': 'lulu_spring_festival.png'
    },
    'birthday': {
        'name': '生日派對',
        'prompt_addition': (
            "in a birthday party scene with colorful balloons and confetti, "
            "wearing a cute party hat, surrounded by birthday cake, presents, and streamers, "
            "kawaii art style, vibrant rainbow colors, soft celebratory lighting"
        ),
        'filename': 'lulu_birthday.png'
    },
    'summer': {
        'name': '夏日度假',
        'prompt_addition': (
            "in a summer beach scene with palm trees and ocean waves, "
            "wearing sunglasses and a sun hat, surrounded by beach balls, ice cream, and seashells, "
            "kawaii art style, vibrant tropical colors, bright sunny lighting"
        ),
        'filename': 'lulu_summer.png'
    }
}


def generate_scene_variations(
    reference_image_path: str,
    base_character_description: str = None,
    output_dir: str = 'data/generated_images/scene_variations'
):
    """
    生成場景變化圖片。

    Args:
        reference_image_path: 參考圖片路徑
        base_character_description: 基礎角色描述（可選）
        output_dir: 輸出目錄
    """
    logger.info(f"\n{'='*80}")
    logger.info("場景變化生成器")
    logger.info(f"使用參考圖片: {reference_image_path}")
    logger.info(f"{'='*80}\n")

    # Verify reference image exists
    ref_path = Path(reference_image_path)
    if not ref_path.exists():
        logger.error(f"❌ 參考圖片不存在: {reference_image_path}")
        return None

    # Initialize Google Gemini client
    try:
        client = GoogleGeminiImageClient(output_dir=output_dir)
        logger.info("✅ Google Gemini Image Client initialized\n")
    except ValueError as e:
        logger.error(f"❌ Error: {e}")
        logger.error("Please set GEMINI_API_KEY in .env file")
        return None

    # Base character description (可以從參考圖片推斷，或使用預設)
    if base_character_description is None:
        base_character_description = (
            "Lulu豬, chubby pastel piglet mascot, super-round head and torso, "
            "short stubby limbs, pill-shaped body, tiny feet and hands, "
            "soft velvet flocked surface, matte finish, no shine. "
            "eyes: very small bead-like black dots, slightly downturned, "
            "tired and listless, no catchlights, no reflections, wide eye spacing. "
            "expression: blank, calm, mildly sleepy, low energy, mouth absent. "
            "snout: small oval peach nose plate with two oval nostrils, soft edges. "
            "ears: short triangular, softly folded, pale pink with subtle gradient. "
            "cheeks: faint blush circles. "
            "color palette: milky pastel pink skin, peach snout, soft rose blush. "
        )

    results = {}

    # Generate each scene variation
    for scene_id, scene_config in SCENES.items():
        logger.info(f"\n{'─'*80}")
        logger.info(f"【生成場景: {scene_config['name']}】")
        logger.info(f"{'─'*80}")

        # Build prompt with reference instruction
        prompt = (
            f"Generate an image of the same character from the reference image. "
            f"Maintain the exact character design, style, and proportions. "
            f"{base_character_description}"
            f"{scene_config['prompt_addition']}"
        )

        logger.info(f"Prompt length: {len(prompt.split())} words")
        logger.info(f"Prompt preview: {prompt[:150]}...\n")

        try:
            result = client.generate(
                prompt=prompt,
                image_filename=scene_config['filename'],
                reference_images=[reference_image_path],
                max_retries=5,
                retry_delay=50
            )

            results[scene_id] = result

            logger.info(f"✅ {scene_config['name']} 圖片生成成功！")
            logger.info(f"   Local Path: {result.get('local_path', 'N/A')}")
            logger.info(f"   Duration: {result['duration']:.2f}s")
            logger.info(f"   Cost: ${result['cost']}\n")

        except Exception as e:
            logger.error(f"❌ Failed to generate {scene_config['name']}: {e}\n")
            results[scene_id] = {'error': str(e)}

    # === Summary ===
    logger.info(f"\n{'='*80}")
    logger.info("場景變化生成總結")
    logger.info(f"{'='*80}\n")

    cost_summary = client.get_cost_summary()
    logger.info(f"Images Generated: {cost_summary['images_generated']}")
    logger.info(f"Total Cost: ${cost_summary['total_cost']}")

    success_count = sum(1 for r in results.values() if 'error' not in r)
    logger.info(f"\n成功生成: {success_count}/{len(SCENES)} 個場景")

    for scene_id, result in results.items():
        scene_name = SCENES[scene_id]['name']
        if 'error' not in result:
            logger.info(f"  ✅ {scene_name}: {result.get('local_path', 'N/A')}")
        else:
            logger.info(f"  ❌ {scene_name}: {result['error']}")

    # Save results
    output_path = Path(output_dir) / 'scene_variations_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'reference_image': reference_image_path,
            'scenes': SCENES,
            'results': results,
            'cost_summary': cost_summary,
            'api': 'Google Gemini 2.5 Flash Image (Nano Banana)'
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 結果已儲存: {output_path}")

    logger.info(f"\n{'='*80}")
    logger.info("下一步：使用 CLIP 驗證角色一致性")
    logger.info(f"{'='*80}\n")

    return results


def main():
    """執行場景變化生成。"""

    # 使用 GPT-Enhanced 版本作為參考（更具有 chill 文化特徵）
    reference_image = 'data/generated_images/comparison_test/comparison_2_gpt_enhanced.png'

    # 也可以使用 Standard 版本
    # reference_image = 'data/generated_images/comparison_test/comparison_1_standard.png'

    logger.info("="*80)
    logger.info("開始生成場景變化")
    logger.info(f"參考圖片: {reference_image}")
    logger.info("="*80)

    results = generate_scene_variations(
        reference_image_path=reference_image
    )

    if results:
        logger.info("\n✅ 場景變化生成完成！")
        logger.info("請檢查生成的圖片，確認角色一致性。")
    else:
        logger.error("\n❌ 場景變化生成失敗")


if __name__ == '__main__':
    main()
