"""
優化版場景變化生成器 - 強調 Reference 角色完整性

核心改進：
1. 不詳細描述角色特徵
2. 強調「在 ref 圖角色上加上 {場景/服裝/趨勢}」
3. 目標：CLIP 相似度 >= 0.8

Author: Product Manager (John)
Date: 2025-10-27
Version: 2.0 - Optimized prompt strategy for character consistency
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


# 場景配置 - 優化版 Prompts
SCENES_V2 = {
    'halloween': {
        'name': '萬聖節',
        'prompt': (
            "Generate the exact same character from the reference image. "
            "Keep the character identical - same face, body, proportions, and style. "
            "Only add Halloween theme: "
            "Add a cute witch hat on the head, "
            "add a small cape or collar around the neck (do not cover the body), "
            "place the character in a Halloween scene with pumpkins and jack-o'-lanterns, "
            "add friendly ghosts and bats in the background, "
            "use vibrant orange and purple colors with soft mysterious lighting. "
            "Maintain the same kawaii art style."
        ),
        'filename': 'lulu_halloween_v2.png'
    },
    'spring_festival': {
        'name': '春節',
        'prompt': (
            "Generate the exact same character from the reference image. "
            "Keep the character identical - same face, body, proportions, and style. "
            "Only add Chinese Spring Festival theme: "
            "Add a traditional Chinese festive collar or small decorative element (do not cover the body), "
            "place the character in a Spring Festival scene with red lanterns and golden decorations, "
            "add firecrackers, lucky coins, and plum blossoms in the background, "
            "use vibrant red and gold colors with warm celebratory lighting. "
            "Maintain the same kawaii art style."
        ),
        'filename': 'lulu_spring_festival_v2.png'
    },
}


def generate_scene_variations_v2(
    reference_image_path: str,
    output_dir: str = 'data/generated_images/scene_variations_v2'
):
    """
    生成場景變化圖片（優化版 - 強調角色一致性）。

    優化策略：
    1. 不描述角色特徵，依賴 reference image
    2. 明確指示「Keep the character identical」
    3. 只描述要添加的場景/服飾元素
    4. 避免遮擋關鍵特徵（眼睛、耳朵、身體）

    Args:
        reference_image_path: 參考圖片路徑
        output_dir: 輸出目錄
    """
    logger.info(f"\n{'='*80}")
    logger.info("場景變化生成器 V2.0 - 優化角色一致性")
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

    results = {}

    # Generate each scene variation
    for scene_id, scene_config in SCENES_V2.items():
        logger.info(f"\n{'─'*80}")
        logger.info(f"【生成場景: {scene_config['name']}】")
        logger.info(f"{'─'*80}")

        prompt = scene_config['prompt']

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
    logger.info("場景變化生成總結 V2.0")
    logger.info(f"{'='*80}\n")

    cost_summary = client.get_cost_summary()
    logger.info(f"Images Generated: {cost_summary['images_generated']}")
    logger.info(f"Total Cost: ${cost_summary['total_cost']}")

    success_count = sum(1 for r in results.values() if 'error' not in r)
    logger.info(f"\n成功生成: {success_count}/{len(SCENES_V2)} 個場景")

    for scene_id, result in results.items():
        scene_name = SCENES_V2[scene_id]['name']
        if 'error' not in result:
            logger.info(f"  ✅ {scene_name}: {result.get('local_path', 'N/A')}")
        else:
            logger.info(f"  ❌ {scene_name}: {result['error']}")

    # Save results
    output_path = Path(output_dir) / 'scene_variations_v2_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': '2.0',
            'strategy': 'Emphasize reference character identity, only change scene/costume/props',
            'reference_image': reference_image_path,
            'scenes': SCENES_V2,
            'results': results,
            'cost_summary': cost_summary,
            'api': 'Google Gemini 2.5 Flash Image (Nano Banana)'
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 結果已儲存: {output_path}")

    logger.info(f"\n{'='*90}")
    logger.info("下一步：使用 CLIP 驗證角色一致性（目標: >= 0.8）")
    logger.info(f"{'='*90}\n")

    return results


def main():
    """執行場景變化生成 V2.0。"""

    # 使用官方 IP reference image（推薦 ref_3，帶場景和道具）
    reference_image = 'data/reference_images/lulu_pig_ref_3.jpg'

    logger.info("="*80)
    logger.info("開始生成場景變化 V2.0 - 優化角色一致性")
    logger.info(f"參考圖片: {reference_image}")
    logger.info("="*80)

    results = generate_scene_variations_v2(
        reference_image_path=reference_image
    )

    if results:
        logger.info("\n✅ 場景變化生成完成！")
        logger.info("請檢查生成的圖片，確認角色一致性是否提升至 >= 0.8。")
    else:
        logger.error("\n❌ 場景變化生成失敗")


if __name__ == '__main__':
    main()
