"""
角色專注驗證器 (Character-Focused Validator)

使用背景移除技術，讓 CLIP 只針對角色本身進行比較，排除場景影響。

方案 1: 使用 rembg 移除背景
方案 2: 裁剪中心區域（簡單快速）
方案 3: 使用白背景 reference

Author: Product Manager (John)
Date: 2025-10-27
Version: 1.0
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharacterFocusedValidator:
    """
    角色專注驗證器 - 只比較角色，不受場景影響。

    支援多種策略：
    1. background_removal: 使用 rembg 移除背景
    2. center_crop: 裁剪中心區域
    3. white_bg_ref: 使用白背景 reference
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None
    ):
        """初始化驗證器。"""
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        logger.info("CharacterFocusedValidator initialized")

    def remove_background(self, image_path: str) -> Image.Image:
        """
        移除背景，只保留角色。

        使用 rembg 庫進行背景移除。
        需要安裝：pip install rembg

        Args:
            image_path: 圖片路徑

        Returns:
            移除背景後的 PIL Image
        """
        try:
            from rembg import remove
        except ImportError:
            raise ImportError(
                "rembg not installed. Install with: pip install rembg"
            )

        logger.info(f"Removing background from: {image_path}")

        # 讀取圖片
        with open(image_path, 'rb') as f:
            input_image = f.read()

        # 移除背景
        output_image = remove(input_image)

        # 轉換為 PIL Image
        from io import BytesIO
        image = Image.open(BytesIO(output_image)).convert('RGB')

        logger.info("Background removed successfully")
        return image

    def center_crop(
        self,
        image_path: str,
        crop_ratio: float = 0.6
    ) -> Image.Image:
        """
        裁剪中心區域（角色通常在中心）。

        這是一個簡單快速的方法，不需要額外依賴。

        Args:
            image_path: 圖片路徑
            crop_ratio: 裁剪比例（0.6 = 保留中心 60% 區域）

        Returns:
            裁剪後的 PIL Image
        """
        logger.info(f"Center cropping: {image_path} (ratio: {crop_ratio})")

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # 計算裁剪區域
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        cropped = image.crop((left, top, right, bottom))

        logger.info(f"Cropped to: {crop_width}x{crop_height}")
        return cropped

    def compute_embedding(
        self,
        image: Image.Image
    ) -> np.ndarray:
        """
        計算圖片的 CLIP embedding。

        Args:
            image: PIL Image

        Returns:
            Normalized embedding vector
        """
        # 預處理圖片
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 計算 embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """計算相似度。"""
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def validate_with_strategy(
        self,
        generated_image_path: str,
        reference_image_path: str,
        strategy: str = "background_removal"
    ) -> Dict:
        """
        使用指定策略驗證角色一致性。

        Args:
            generated_image_path: 生成圖片路徑
            reference_image_path: 參考圖片路徑
            strategy: 策略選擇
                - "background_removal": 移除背景（最準確，需要 rembg）
                - "center_crop": 裁剪中心（快速，無額外依賴）
                - "original": 原始圖片（不處理）

        Returns:
            Dictionary with validation result
        """
        logger.info(f"\n{'─'*60}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"{'─'*60}")

        # 根據策略處理圖片
        if strategy == "background_removal":
            try:
                gen_image = self.remove_background(generated_image_path)
                ref_image = self.remove_background(reference_image_path)
            except ImportError as e:
                logger.warning(f"Background removal failed: {e}")
                logger.warning("Falling back to center_crop strategy")
                strategy = "center_crop"
                gen_image = self.center_crop(generated_image_path)
                ref_image = self.center_crop(reference_image_path)

        elif strategy == "center_crop":
            gen_image = self.center_crop(generated_image_path, crop_ratio=0.6)
            ref_image = self.center_crop(reference_image_path, crop_ratio=0.6)

        elif strategy == "original":
            gen_image = Image.open(generated_image_path).convert('RGB')
            ref_image = Image.open(reference_image_path).convert('RGB')

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 計算 embeddings
        gen_embedding = self.compute_embedding(gen_image)
        ref_embedding = self.compute_embedding(ref_image)

        # 計算相似度
        similarity = self.compute_similarity(gen_embedding, ref_embedding)

        result = {
            'strategy': strategy,
            'similarity': round(similarity, 4),
            'generated_image': generated_image_path,
            'reference_image': reference_image_path
        }

        logger.info(f"Similarity: {similarity:.4f}")

        return result

    def compare_strategies(
        self,
        generated_image_path: str,
        reference_image_path: str,
        strategies: List[str] = None
    ) -> Dict[str, Dict]:
        """
        對比多種策略的結果。

        Args:
            generated_image_path: 生成圖片路徑
            reference_image_path: 參考圖片路徑
            strategies: 要測試的策略列表

        Returns:
            Dictionary mapping strategy -> result
        """
        if strategies is None:
            strategies = ["original", "center_crop", "background_removal"]

        logger.info(f"\n{'='*60}")
        logger.info("對比不同策略")
        logger.info(f"{'='*60}\n")

        results = {}

        for strategy in strategies:
            try:
                result = self.validate_with_strategy(
                    generated_image_path,
                    reference_image_path,
                    strategy=strategy
                )
                results[strategy] = result
            except Exception as e:
                logger.error(f"Strategy '{strategy}' failed: {e}")
                results[strategy] = {'error': str(e)}

        # 總結對比
        logger.info(f"\n{'='*60}")
        logger.info("策略對比總結")
        logger.info(f"{'='*60}\n")

        for strategy, result in results.items():
            if 'error' not in result:
                sim = result['similarity']
                logger.info(f"{strategy:20s}: {sim:.4f}")
            else:
                logger.info(f"{strategy:20s}: ❌ {result['error']}")

        return results


def demo():
    """Demo function."""
    print("\n" + "="*80)
    print("Character-Focused Validator Demo")
    print("="*80 + "\n")

    # Initialize validator
    validator = CharacterFocusedValidator()

    # Test images
    ref_image = "data/reference_images/lulu_pig_ref_3.jpg"
    gen_image = "data/generated_images/scene_variations_v2/lulu_halloween_v2.png"

    # 對比不同策略
    results = validator.compare_strategies(gen_image, ref_image)

    # 分析結果
    print("\n" + "="*80)
    print("策略效果分析")
    print("="*80 + "\n")

    if 'original' in results and 'background_removal' in results:
        original_sim = results['original']['similarity']
        bg_removed_sim = results['background_removal']['similarity']
        improvement = bg_removed_sim - original_sim

        print(f"原始比較:     {original_sim:.4f}")
        print(f"移除背景後:   {bg_removed_sim:.4f}")
        print(f"改進幅度:     {improvement:+.4f}")

        if improvement > 0:
            print(f"\n✅ 移除背景後相似度提升！更準確地反映角色一致性。")
        else:
            print(f"\n⚠️ 移除背景後相似度下降，可能角色本身有變化。")


if __name__ == '__main__':
    demo()
