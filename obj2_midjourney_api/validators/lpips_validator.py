"""
LPIPS Validator - Learned Perceptual Image Patch Similarity

使用 LPIPS (Learned Perceptual Image Patch Similarity) 進行角色一致性驗證。
LPIPS 測量感知相似度，更符合人類視覺感知，比 CLIP 更專注於視覺而非語義。

論文: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)
GitHub: https://github.com/richzhang/PerceptualSimilarity

Author: Product Manager (John)
Date: 2025-10-27
"""

from PIL import Image
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LPIPSValidator:
    """
    LPIPS-based 角色一致性驗證器。

    優勢:
    - Perceptual distance - 符合人類視覺感知
    - 成熟穩定 - 廣泛用於 GAN/Diffusion 評估
    - 快速 - 比 CLIP 推理更快
    - 專注視覺相似度 - 不受文字語義影響
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[str] = None
    ):
        """
        初始化 LPIPS 驗證器。

        Args:
            net: 使用的網絡
                - "alex": AlexNet (fastest, 推薦)
                - "vgg": VGG16 (balanced)
                - "squeeze": SqueezeNet (smallest)
            device: 運算裝置
        """
        try:
            import lpips
            import torch
        except ImportError:
            raise ImportError(
                "lpips not installed. Install with: pip install lpips"
            )

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading LPIPS model: {net}")
        logger.info(f"Device: {self.device}")

        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()
        self.net = net

        logger.info("LPIPSValidator initialized")

    def load_image(self, image_path: str, target_size=None):
        """
        載入圖片並預處理為 LPIPS 格式。

        Args:
            image_path: 圖片路徑
            target_size: Optional target size (width, height)

        Returns:
            Preprocessed tensor [-1, 1] range
        """
        import lpips
        import torch

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize if target size specified
        if target_size is not None:
            image = image.resize(target_size, Image.LANCZOS)

        # Convert to tensor [-1, 1] range
        img_tensor = lpips.im2tensor(np.array(image))

        return img_tensor.to(self.device)

    def compute_distance(
        self,
        image1_path: str,
        image2_path: str
    ) -> float:
        """
        計算兩張圖片的 LPIPS 距離。

        Args:
            image1_path: 第一張圖片路徑
            image2_path: 第二張圖片路徑

        Returns:
            LPIPS distance (0 = identical, larger = more different)
        """
        import torch

        # Get image sizes
        img1_pil = Image.open(image1_path)
        img2_pil = Image.open(image2_path)

        # Use the smaller dimensions to avoid upscaling
        target_width = min(img1_pil.width, img2_pil.width)
        target_height = min(img1_pil.height, img2_pil.height)
        target_size = (target_width, target_height)

        logger.info(f"Resizing images to: {target_size}")

        # Load images with unified size
        img1 = self.load_image(image1_path, target_size=target_size)
        img2 = self.load_image(image2_path, target_size=target_size)

        # Compute distance
        with torch.no_grad():
            distance = self.loss_fn(img1, img2)

        return float(distance.item())

    def compute_similarity(
        self,
        image1_path: str,
        image2_path: str
    ) -> float:
        """
        計算兩張圖片的相似度（1 - distance）。

        Args:
            image1_path: 第一張圖片路徑
            image2_path: 第二張圖片路徑

        Returns:
            Similarity score (1 = identical, 0 = completely different)
        """
        distance = self.compute_distance(image1_path, image2_path)

        # Convert distance to similarity
        # LPIPS distance is typically in [0, 1] range
        # but can be slightly > 1 for very different images
        similarity = max(0.0, 1.0 - distance)

        return similarity

    def validate(
        self,
        generated_image_path: str,
        reference_image_path: str
    ) -> Dict:
        """
        驗證生成圖片與參考圖片的角色一致性。

        Args:
            generated_image_path: 生成圖片路徑
            reference_image_path: 參考圖片路徑

        Returns:
            Dictionary with validation result:
            {
                'method': 'LPIPS',
                'net': net_name,
                'distance': float,
                'similarity': float,
                'generated_image': str,
                'reference_image': str
            }
        """
        logger.info(f"\n{'─'*60}")
        logger.info(f"LPIPS Validation")
        logger.info(f"{'─'*60}")
        logger.info(f"Generated: {generated_image_path}")
        logger.info(f"Reference: {reference_image_path}")
        logger.info(f"Network: {self.net}")

        # Compute distance and similarity
        distance = self.compute_distance(generated_image_path, reference_image_path)
        similarity = max(0.0, 1.0 - distance)

        result = {
            'method': 'LPIPS',
            'net': self.net,
            'distance': round(distance, 4),
            'similarity': round(similarity, 4),
            'generated_image': generated_image_path,
            'reference_image': reference_image_path
        }

        logger.info(f"Distance: {distance:.4f}")
        logger.info(f"Similarity: {similarity:.4f}")

        return result

    def validate_batch(
        self,
        image_pairs: list
    ) -> Dict[str, Dict]:
        """
        批量驗證多組圖片。

        Args:
            image_pairs: List of tuples (generated_path, reference_path)

        Returns:
            Dictionary mapping test_name -> result
        """
        results = {}

        for i, (gen_path, ref_path) in enumerate(image_pairs):
            test_name = f"test_{i+1}"
            result = self.validate(gen_path, ref_path)
            results[test_name] = result

        return results


def demo():
    """Demo function."""
    print("\n" + "="*80)
    print("LPIPS Validator Demo")
    print("="*80 + "\n")

    # Initialize validator
    validator = LPIPSValidator(net="alex")

    # Test images
    ref_image = "data/reference_images/lulu_pig_ref_3.jpg"
    gen_image = "data/generated_images/e2e_test/e2e_20251027_170132_kawaii_Valentines_Day.png"

    # Validate
    result = validator.validate(gen_image, ref_image)

    print("\n" + "="*80)
    print("Result")
    print("="*80 + "\n")
    print(f"Method: {result['method']}")
    print(f"Network: {result['net']}")
    print(f"Distance: {result['distance']}")
    print(f"Similarity: {result['similarity']}")


if __name__ == '__main__':
    demo()
