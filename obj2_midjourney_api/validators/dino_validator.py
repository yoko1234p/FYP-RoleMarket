"""
DINO Validator - Self-supervised Vision Transformer 驗證器

使用 Facebook DINO (Self-supervised Vision Transformers) 進行角色一致性驗證。
DINO 擅長物體級特徵提取，相比 CLIP 更專注於物體本身而非場景。

論文: "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)
GitHub: https://github.com/facebookresearch/dino

Author: Product Manager (John)
Date: 2025-10-27
"""

import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DINOValidator:
    """
    DINO-based 角色一致性驗證器。

    優勢:
    - 物體級特徵提取（vs CLIP 的場景級）
    - Self-attention 自動關注重要區域
    - Zero-shot，無需訓練
    - 研究證明優於 CLIP 的物體識別能力
    """

    def __init__(
        self,
        model_name: str = "facebook/dino-vitb16",
        device: Optional[str] = None
    ):
        """
        初始化 DINO 驗證器。

        Args:
            model_name: DINO 模型名稱
                - "facebook/dino-vitb16": ViT-Base (86M params, 推薦)
                - "facebook/dino-vits16": ViT-Small (22M params, 更快)
            device: 運算裝置
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading DINO model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # Set to evaluation mode
        self.model.eval()

        logger.info("DINOValidator initialized")

    def compute_embedding(
        self,
        image_path: str,
        pooling: str = "cls"
    ) -> np.ndarray:
        """
        計算圖片的 DINO embedding。

        Args:
            image_path: 圖片路徑
            pooling: 特徵池化方式
                - "cls": 使用 [CLS] token（推薦，最常用）
                - "mean": Mean pooling over all tokens
                - "max": Max pooling over all tokens

        Returns:
            Normalized embedding vector
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pooling
        if pooling == "cls":
            # [CLS] token (first token)
            embedding = outputs.last_hidden_state[:, 0]
        elif pooling == "mean":
            # Mean pooling over all tokens
            embedding = outputs.last_hidden_state.mean(dim=1)
        elif pooling == "max":
            # Max pooling over all tokens
            embedding = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()[0]

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        計算兩個 embedding 的餘弦相似度。

        Args:
            embedding1: 第一個 embedding
            embedding2: 第二個 embedding

        Returns:
            相似度分數 (0-1)
        """
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def validate(
        self,
        generated_image_path: str,
        reference_image_path: str,
        pooling: str = "cls"
    ) -> Dict:
        """
        驗證生成圖片與參考圖片的角色一致性。

        Args:
            generated_image_path: 生成圖片路徑
            reference_image_path: 參考圖片路徑
            pooling: 特徵池化方式

        Returns:
            Dictionary with validation result:
            {
                'method': 'DINO',
                'model': model_name,
                'pooling': pooling,
                'similarity': float,
                'generated_image': str,
                'reference_image': str
            }
        """
        logger.info(f"\n{'─'*60}")
        logger.info(f"DINO Validation")
        logger.info(f"{'─'*60}")
        logger.info(f"Generated: {generated_image_path}")
        logger.info(f"Reference: {reference_image_path}")
        logger.info(f"Pooling: {pooling}")

        # Compute embeddings
        gen_embedding = self.compute_embedding(generated_image_path, pooling)
        ref_embedding = self.compute_embedding(reference_image_path, pooling)

        # Compute similarity
        similarity = self.compute_similarity(gen_embedding, ref_embedding)

        result = {
            'method': 'DINO',
            'model': self.model.config._name_or_path,
            'pooling': pooling,
            'similarity': round(similarity, 4),
            'generated_image': generated_image_path,
            'reference_image': reference_image_path
        }

        logger.info(f"Similarity: {similarity:.4f}")

        return result

    def validate_batch(
        self,
        image_pairs: list,
        pooling: str = "cls"
    ) -> Dict[str, Dict]:
        """
        批量驗證多組圖片。

        Args:
            image_pairs: List of tuples (generated_path, reference_path)
            pooling: 特徵池化方式

        Returns:
            Dictionary mapping test_name -> result
        """
        results = {}

        for i, (gen_path, ref_path) in enumerate(image_pairs):
            test_name = f"test_{i+1}"
            result = self.validate(gen_path, ref_path, pooling)
            results[test_name] = result

        return results


def demo():
    """Demo function."""
    print("\n" + "="*80)
    print("DINO Validator Demo")
    print("="*80 + "\n")

    # Initialize validator
    validator = DINOValidator()

    # Test images
    ref_image = "data/reference_images/lulu_pig_ref_3.jpg"
    gen_image = "data/generated_images/e2e_test/e2e_20251027_170132_kawaii_Valentines_Day.png"

    # Validate
    result = validator.validate(gen_image, ref_image, pooling="cls")

    print("\n" + "="*80)
    print("Result")
    print("="*80 + "\n")
    print(f"Method: {result['method']}")
    print(f"Model: {result['model']}")
    print(f"Similarity: {result['similarity']}")


if __name__ == '__main__':
    demo()
