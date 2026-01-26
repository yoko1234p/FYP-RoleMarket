"""
SSIM Validator - Structural Similarity Index

使用 SSIM (Structural Similarity Index) 進行角色一致性驗證。
SSIM 是傳統電腦視覺方法，測量結構相似度，快速但僅比較像素級結構。

論文: "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)

Author: Product Manager (John)
Date: 2025-10-27
"""

from PIL import Image
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSIMValidator:
    """
    SSIM-based 角色一致性驗證器。

    優勢:
    - 極快速 - 無需深度學習模型
    - 簡單穩定
    - 適合結構相似的圖片

    劣勢:
    - 只比較像素結構，無語義理解
    - 對顏色/光照變化敏感
    - 不適合場景變化大的圖片
    """

    def __init__(self):
        """初始化 SSIM 驗證器。"""
        try:
            from skimage.metrics import structural_similarity
            self.ssim_func = structural_similarity
        except ImportError:
            raise ImportError(
                "scikit-image not installed. Install with: pip install scikit-image"
            )

        logger.info("SSIMValidator initialized")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        載入圖片並轉換為 numpy array。

        Args:
            image_path: 圖片路徑

        Returns:
            Image as numpy array
        """
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def compute_similarity(
        self,
        image1_path: str,
        image2_path: str,
        multichannel: bool = True
    ) -> float:
        """
        計算兩張圖片的 SSIM 相似度。

        Args:
            image1_path: 第一張圖片路徑
            image2_path: 第二張圖片路徑
            multichannel: 是否使用多通道（彩色圖片）

        Returns:
            SSIM similarity score (-1 to 1, typically 0 to 1)
        """
        # Load images
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)

        # Resize to same size if needed
        if img1.shape != img2.shape:
            logger.warning(f"Images have different sizes: {img1.shape} vs {img2.shape}")
            logger.warning(f"Resizing image2 to match image1")
            img2_pil = Image.fromarray(img2)
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]))
            img2 = np.array(img2_pil)

        # Compute SSIM
        similarity = self.ssim_func(
            img1,
            img2,
            multichannel=multichannel,
            channel_axis=2 if multichannel else None,
            data_range=255
        )

        return float(similarity)

    def validate(
        self,
        generated_image_path: str,
        reference_image_path: str,
        multichannel: bool = True
    ) -> Dict:
        """
        驗證生成圖片與參考圖片的角色一致性。

        Args:
            generated_image_path: 生成圖片路徑
            reference_image_path: 參考圖片路徑
            multichannel: 是否使用多通道

        Returns:
            Dictionary with validation result:
            {
                'method': 'SSIM',
                'similarity': float,
                'multichannel': bool,
                'generated_image': str,
                'reference_image': str
            }
        """
        logger.info(f"\n{'─'*60}")
        logger.info(f"SSIM Validation")
        logger.info(f"{'─'*60}")
        logger.info(f"Generated: {generated_image_path}")
        logger.info(f"Reference: {reference_image_path}")
        logger.info(f"Multichannel: {multichannel}")

        # Compute similarity
        similarity = self.compute_similarity(
            generated_image_path,
            reference_image_path,
            multichannel
        )

        result = {
            'method': 'SSIM',
            'similarity': round(similarity, 4),
            'multichannel': multichannel,
            'generated_image': generated_image_path,
            'reference_image': reference_image_path
        }

        logger.info(f"Similarity: {similarity:.4f}")

        return result

    def validate_batch(
        self,
        image_pairs: list,
        multichannel: bool = True
    ) -> Dict[str, Dict]:
        """
        批量驗證多組圖片。

        Args:
            image_pairs: List of tuples (generated_path, reference_path)
            multichannel: 是否使用多通道

        Returns:
            Dictionary mapping test_name -> result
        """
        results = {}

        for i, (gen_path, ref_path) in enumerate(image_pairs):
            test_name = f"test_{i+1}"
            result = self.validate(gen_path, ref_path, multichannel)
            results[test_name] = result

        return results


def demo():
    """Demo function."""
    print("\n" + "="*80)
    print("SSIM Validator Demo")
    print("="*80 + "\n")

    # Initialize validator
    validator = SSIMValidator()

    # Test images
    ref_image = "data/reference_images/lulu_pig_ref_3.jpg"
    gen_image = "data/generated_images/e2e_test/e2e_20251027_170132_kawaii_Valentines_Day.png"

    # Validate
    result = validator.validate(gen_image, ref_image)

    print("\n" + "="*80)
    print("Result")
    print("="*80 + "\n")
    print(f"Method: {result['method']}")
    print(f"Similarity: {result['similarity']}")


if __name__ == '__main__':
    demo()
