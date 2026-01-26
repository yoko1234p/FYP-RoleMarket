"""
驗證 V2 生成圖片的角色一致性

使用 CLIP 模型驗證優化後的場景圖片，目標：相似度 >= 0.8

Author: Product Manager (John)
Date: 2025-10-27
Version: 2.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from obj2_midjourney_api.clip_validator import CLIPValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_v2_images():
    """驗證 V2 生成圖片的角色一致性。"""
    logger.info(f"\n{'='*80}")
    logger.info("驗證 V2 生成圖片角色一致性（目標：>= 0.8）")
    logger.info(f"{'='*80}\n")

    # Initialize CLIP validator
    validator = CLIPValidator()

    # 參考圖片
    reference_image = 'data/generated_images/comparison_test/comparison_2_gpt_enhanced.png'

    # V2 場景圖片
    v2_images = {
        'Halloween V2': 'data/generated_images/scene_variations_v2/lulu_halloween_v2.png',
        'Spring Festival V2': 'data/generated_images/scene_variations_v2/lulu_spring_festival_v2.png',
        'Birthday V2': 'data/generated_images/scene_variations_v2/lulu_birthday_v2.png',
        'Summer V2': 'data/generated_images/scene_variations_v2/lulu_summer_v2.png',
    }

    # 驗證圖片存在性
    all_images = [reference_image] + list(v2_images.values())
    missing_images = []
    for img_path in all_images:
        if not Path(img_path).exists():
            missing_images.append(img_path)

    if missing_images:
        logger.error(f"❌ 以下圖片不存在：")
        for img in missing_images:
            logger.error(f"   - {img}")
        return None

    logger.info(f"✅ 找到所有圖片\n")

    # 計算參考圖片的 embedding
    logger.info("計算參考圖片 embedding...")
    ref_embedding = validator.compute_embedding(reference_image)
    logger.info(f"  ✓ Reference (GPT-Enhanced)\n")

    # 計算各場景圖片與參考圖片的相似度
    logger.info(f"{'─'*80}")
    logger.info("計算與參考圖片的相似度...")
    logger.info(f"{'─'*80}\n")

    results = {
        'reference_image': reference_image,
        'v2_images': {},
        'statistics': {}
    }

    similarities = []

    for scene_name, scene_path in v2_images.items():
        scene_embedding = validator.compute_embedding(scene_path)
        similarity = validator.compute_similarity(ref_embedding, scene_embedding)

        results['v2_images'][scene_name] = {
            'path': scene_path,
            'similarity': round(similarity, 4)
        }

        similarities.append(similarity)

        # 檢查是否達到目標 >= 0.8
        if similarity >= 0.80:
            status = "✅ PASS (>= 0.8)"
            emoji = "🌟"
        elif similarity >= 0.75:
            status = "✅ PASS (>= 0.75)"
            emoji = "👍"
        else:
            status = "⚠️ WARN (< 0.75)"
            emoji = "⚠️"

        logger.info(f"{emoji} {scene_name:25s}: {similarity:.4f} {status}")

    # 統計數據
    avg_similarity = sum(similarities) / len(similarities)
    min_similarity = min(similarities)
    max_similarity = max(similarities)

    passed_08 = sum(1 for s in similarities if s >= 0.80)
    passed_075 = sum(1 for s in similarities if s >= 0.75)

    results['statistics'] = {
        'total_scenes': len(similarities),
        'avg_similarity': round(avg_similarity, 4),
        'min_similarity': round(min_similarity, 4),
        'max_similarity': round(max_similarity, 4),
        'target_threshold': 0.80,
        'core_threshold': 0.75,
        'passed_0.8': passed_08,
        'passed_0.75': passed_075,
        'failed': sum(1 for s in similarities if s < 0.75)
    }

    # 輸出統計結果
    logger.info(f"\n{'='*80}")
    logger.info("統計結果")
    logger.info(f"{'='*80}\n")

    logger.info(f"總場景數量: {results['statistics']['total_scenes']}")
    logger.info(f"平均相似度: {results['statistics']['avg_similarity']:.4f}")
    logger.info(f"最低相似度: {results['statistics']['min_similarity']:.4f}")
    logger.info(f"最高相似度: {results['statistics']['max_similarity']:.4f}")
    logger.info(f"\n目標 Threshold (>= 0.8): {results['statistics']['passed_0.8']}/{results['statistics']['total_scenes']} 通過")
    logger.info(f"Core Threshold (>= 0.75): {results['statistics']['passed_0.75']}/{results['statistics']['total_scenes']} 通過")

    # 評估改進效果
    logger.info(f"\n{'─'*80}")
    logger.info("改進效果評估")
    logger.info(f"{'─'*80}\n")

    # 與 V1 結果比較（從之前的結果）
    v1_avg = 0.7949  # V1 平均相似度（包括場景之間的相似度）
    v1_ref_to_scenes = [0.7171, 0.8105, 0.8380, 0.7198]  # V1 中 GPT-Enhanced 到各場景的相似度
    v1_avg_ref_to_scenes = sum(v1_ref_to_scenes) / len(v1_ref_to_scenes)

    improvement = avg_similarity - v1_avg_ref_to_scenes

    logger.info(f"V1 平均相似度（ref → scenes）: {v1_avg_ref_to_scenes:.4f}")
    logger.info(f"V2 平均相似度（ref → scenes）: {avg_similarity:.4f}")
    logger.info(f"改進幅度: {improvement:+.4f} ({improvement/v1_avg_ref_to_scenes*100:+.2f}%)")

    if avg_similarity >= 0.80:
        consistency_level = "優秀 (Excellent) - 達到目標！"
        emoji = "🌟"
    elif avg_similarity >= 0.75:
        consistency_level = "良好 (Good) - 接近目標"
        emoji = "✅"
    else:
        consistency_level = "需要改進 (Needs Improvement)"
        emoji = "⚠️"

    results['statistics']['consistency_level'] = consistency_level
    results['statistics']['improvement_vs_v1'] = round(improvement, 4)

    logger.info(f"\n{emoji} 角色一致性等級: {consistency_level}")

    # 保存結果
    output_path = Path('data/generated_images/clip_validation_v2_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 驗證結果已儲存: {output_path}")

    logger.info(f"\n{'='*80}")
    logger.info("驗證完成！")
    logger.info(f"{'='*80}\n")

    return results


def main():
    """執行驗證。"""
    results = validate_v2_images()

    if results:
        logger.info("✅ V2 角色一致性驗證完成！")

        # 總結建議
        avg = results['statistics']['avg_similarity']
        if avg >= 0.80:
            logger.info("\n🎉 恭喜！已達到目標相似度 >= 0.8")
        else:
            logger.info(f"\n💡 建議：平均相似度 {avg:.4f}，距離目標 0.8 還差 {0.8 - avg:.4f}")
            logger.info("   可以進一步優化 prompt，減少遮擋關鍵特徵的服飾")
    else:
        logger.error("❌ 驗證失敗")


if __name__ == '__main__':
    main()
