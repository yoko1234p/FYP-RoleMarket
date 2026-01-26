"""
驗證生成圖片的角色一致性

使用 CLIP 模型驗證所有生成圖片（對比圖 + 場景變化圖）的角色一致性。

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
from obj2_midjourney_api.clip_validator import CLIPValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_all_generated_images():
    """驗證所有生成圖片的角色一致性。"""
    logger.info(f"\n{'='*80}")
    logger.info("驗證生成圖片角色一致性")
    logger.info(f"{'='*80}\n")

    # Initialize CLIP validator
    validator = CLIPValidator()

    # 定義所有要驗證的圖片
    images = {
        'comparison': [
            'data/generated_images/comparison_test/comparison_1_standard.png',
            'data/generated_images/comparison_test/comparison_2_gpt_enhanced.png',
        ],
        'scenes': [
            'data/generated_images/scene_variations/lulu_halloween.png',
            'data/generated_images/scene_variations/lulu_spring_festival.png',
            'data/generated_images/scene_variations/lulu_birthday.png',
            'data/generated_images/scene_variations/lulu_summer.png',
        ]
    }

    # 驗證圖片存在性
    all_images = images['comparison'] + images['scenes']
    missing_images = []
    for img_path in all_images:
        if not Path(img_path).exists():
            missing_images.append(img_path)

    if missing_images:
        logger.error(f"❌ 以下圖片不存在：")
        for img in missing_images:
            logger.error(f"   - {img}")
        return None

    logger.info(f"✅ 找到所有 {len(all_images)} 張圖片\n")

    # 為每張圖片分配標籤
    image_labels = {
        images['comparison'][0]: 'Standard',
        images['comparison'][1]: 'GPT-Enhanced',
        images['scenes'][0]: 'Halloween',
        images['scenes'][1]: 'Spring Festival',
        images['scenes'][2]: 'Birthday',
        images['scenes'][3]: 'Summer',
    }

    # 初始化結果字典
    results = {
        'images': {},
        'similarity_matrix': {},
        'statistics': {}
    }

    # 計算所有圖片的 embeddings（一次性計算，避免重複）
    logger.info("計算所有圖片的 embeddings...")
    embeddings = {}
    for img_path in all_images:
        img_label = image_labels[img_path]
        embeddings[img_label] = validator.compute_embedding(img_path)
        logger.info(f"  ✓ {img_label}")

    logger.info(f"\n{'─'*80}")
    logger.info("計算相似度矩陣...")
    logger.info(f"{'─'*80}\n")

    # 計算所有圖片之間的相似度
    similarities = []

    for i, img1_path in enumerate(all_images):
        img1_label = image_labels[img1_path]
        results['similarity_matrix'][img1_label] = {}

        for j, img2_path in enumerate(all_images):
            img2_label = image_labels[img2_path]

            if i <= j:  # 只計算上三角矩陣（包括對角線）
                similarity = validator.compute_similarity(
                    embeddings[img1_label],
                    embeddings[img2_label]
                )
                results['similarity_matrix'][img1_label][img2_label] = round(similarity, 4)

                if i != j:  # 不包括自相似度
                    similarities.append(similarity)

                    # 檢查是否符合 core threshold
                    status = "✅ PASS" if similarity >= validator.core_threshold else "⚠️ WARN"
                    logger.info(f"{img1_label:20s} ↔ {img2_label:20s}: {similarity:.4f} {status}")

    # 統計數據
    avg_similarity = sum(similarities) / len(similarities)
    min_similarity = min(similarities)
    max_similarity = max(similarities)

    results['statistics'] = {
        'total_comparisons': len(similarities),
        'avg_similarity': round(avg_similarity, 4),
        'min_similarity': round(min_similarity, 4),
        'max_similarity': round(max_similarity, 4),
        'core_threshold': validator.core_threshold,
        'passed': sum(1 for s in similarities if s >= validator.core_threshold),
        'failed': sum(1 for s in similarities if s < validator.core_threshold)
    }

    # 輸出統計結果
    logger.info(f"\n{'='*80}")
    logger.info("統計結果")
    logger.info(f"{'='*80}\n")

    logger.info(f"總比較次數: {results['statistics']['total_comparisons']}")
    logger.info(f"平均相似度: {results['statistics']['avg_similarity']:.4f}")
    logger.info(f"最低相似度: {results['statistics']['min_similarity']:.4f}")
    logger.info(f"最高相似度: {results['statistics']['max_similarity']:.4f}")
    logger.info(f"Core Threshold: {results['statistics']['core_threshold']}")
    logger.info(f"通過數量: {results['statistics']['passed']}/{results['statistics']['total_comparisons']}")
    logger.info(f"未通過數量: {results['statistics']['failed']}/{results['statistics']['total_comparisons']}")

    # 評估角色一致性
    logger.info(f"\n{'─'*80}")
    logger.info("角色一致性評估")
    logger.info(f"{'─'*80}\n")

    if avg_similarity >= 0.80:
        consistency_level = "優秀 (Excellent)"
        emoji = "🌟"
    elif avg_similarity >= 0.75:
        consistency_level = "良好 (Good)"
        emoji = "✅"
    elif avg_similarity >= 0.70:
        consistency_level = "合格 (Pass)"
        emoji = "👍"
    else:
        consistency_level = "需要改進 (Needs Improvement)"
        emoji = "⚠️"

    results['statistics']['consistency_level'] = consistency_level

    logger.info(f"{emoji} 角色一致性等級: {consistency_level}")
    logger.info(f"   平均相似度: {avg_similarity:.4f}")

    # 找出最相似和最不相似的圖片對
    logger.info(f"\n{'─'*80}")
    logger.info("極端值分析")
    logger.info(f"{'─'*80}\n")

    # 找出最相似的圖片對
    max_pair = None
    max_sim = 0
    for img1 in image_labels.values():
        for img2 in image_labels.values():
            if img1 != img2 and img1 in results['similarity_matrix'] and img2 in results['similarity_matrix'][img1]:
                sim = results['similarity_matrix'][img1][img2]
                if sim > max_sim:
                    max_sim = sim
                    max_pair = (img1, img2)

    # 找出最不相似的圖片對
    min_pair = None
    min_sim = 1.0
    for img1 in image_labels.values():
        for img2 in image_labels.values():
            if img1 != img2 and img1 in results['similarity_matrix'] and img2 in results['similarity_matrix'][img1]:
                sim = results['similarity_matrix'][img1][img2]
                if sim < min_sim:
                    min_sim = sim
                    min_pair = (img1, img2)

    logger.info(f"✨ 最相似圖片對: {max_pair[0]} ↔ {max_pair[1]} ({max_sim:.4f})")
    logger.info(f"⚠️  最不相似圖片對: {min_pair[0]} ↔ {min_pair[1]} ({min_sim:.4f})")

    # 保存結果
    output_path = Path('data/generated_images/clip_validation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 驗證結果已儲存: {output_path}")

    logger.info(f"\n{'='*80}")
    logger.info("驗證完成！")
    logger.info(f"{'='*80}\n")

    return results


def main():
    """執行驗證。"""
    results = validate_all_generated_images()

    if results:
        logger.info("✅ 角色一致性驗證完成！")
    else:
        logger.error("❌ 驗證失敗")


if __name__ == '__main__':
    main()
