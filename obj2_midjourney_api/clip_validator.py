"""
CLIP Similarity Validator

Validates character consistency between generated images and reference images
using OpenAI's CLIP (Contrastive Language-Image Pre-training) model.

Model: CLIP ViT-Large/14
Thresholds:
- Core similarity: > 0.75 (character identity)
- Style similarity: > 0.60 (artistic style)

Author: Product Manager (John)
Epic: 3 - Objective 2: Midjourney API Integration
Story: 3.4 - CLIP Similarity Validation

Usage:
    from obj2_midjourney_api.clip_validator import CLIPValidator

    validator = CLIPValidator()
    result = validator.validate_image(
        generated_image_path='data/generated_images/halloween_var1.png',
        reference_image_paths=['data/reference_images/lulu_pig_ref_1.png']
    )

    if result['is_valid']:
        print(f"✅ Valid: similarity={result['similarity']:.3f}")
    else:
        print(f"❌ Invalid: similarity={result['similarity']:.3f}")
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPValidator:
    """
    CLIP-based image similarity validator for character consistency.

    Uses OpenAI's CLIP ViT-Large/14 model to compute cosine similarity
    between generated images and reference images.

    Features:
    - Multi-reference support (average similarity)
    - Configurable thresholds for core and style validation
    - GPU acceleration if available
    - Embedding caching for efficiency
    """

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-large-patch14',
        core_threshold: Optional[float] = None,
        style_threshold: Optional[float] = None,
        device: Optional[str] = None,
        cache_dir: str = 'data/clip_embeddings'
    ):
        """
        Initialize CLIP validator.

        Args:
            model_name: HuggingFace model identifier
            core_threshold: Minimum similarity for core character features (default from .env)
            style_threshold: Minimum similarity for style consistency (default from .env)
            device: Device to use ('cuda', 'mps', 'cpu'), auto-detect if None
            cache_dir: Directory to cache computed embeddings
        """
        load_dotenv()

        # Load thresholds from environment or use defaults
        self.core_threshold = core_threshold or float(os.getenv('CLIP_THRESHOLD_CORE', 0.75))
        self.style_threshold = style_threshold or float(os.getenv('CLIP_THRESHOLD_STYLE', 0.60))

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        # Load CLIP model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        logger.info(f"Device: {device}")

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Set to eval mode
        self.model.eval()

        # Cache directory for embeddings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CLIP Validator initialized")
        logger.info(f"Core threshold: {self.core_threshold}")
        logger.info(f"Style threshold: {self.style_threshold}")

    def compute_embedding(self, image_path: str, use_cache: bool = True) -> np.ndarray:
        """
        Compute CLIP image embedding.

        Args:
            image_path: Path to image file
            use_cache: Whether to use cached embedding if available

        Returns:
            Image embedding as numpy array (normalized)
        """
        image_path = Path(image_path)

        # Check cache
        cache_file = self.cache_dir / f"{image_path.stem}.npy"
        if use_cache and cache_file.exists():
            logger.debug(f"Loading cached embedding: {cache_file.name}")
            return np.load(cache_file)

        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Compute embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # Convert to numpy
            embedding = image_features.cpu().numpy()[0]

        # Cache embedding
        if use_cache:
            np.save(cache_file, embedding)
            logger.debug(f"Cached embedding: {cache_file.name}")

        return embedding

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0 to 1)
        """
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def validate_image(
        self,
        generated_image_path: str,
        reference_image_paths: List[str],
        validation_type: str = 'core',
        return_details: bool = True
    ) -> Dict:
        """
        Validate a generated image against reference images.

        Args:
            generated_image_path: Path to generated image
            reference_image_paths: List of reference image paths
            validation_type: 'core' (character identity) or 'style' (artistic style)
            return_details: Whether to return detailed similarity scores

        Returns:
            Dictionary with validation result:
                - is_valid: bool
                - similarity: float (average if multiple references)
                - threshold: float (threshold used)
                - validation_type: str
                - reference_similarities: list (if return_details=True)
        """
        # Select threshold
        threshold = self.core_threshold if validation_type == 'core' else self.style_threshold

        # Compute generated image embedding
        generated_embedding = self.compute_embedding(generated_image_path)

        # Compute similarities with all references
        similarities = []
        for ref_path in reference_image_paths:
            ref_embedding = self.compute_embedding(ref_path)
            sim = self.compute_similarity(generated_embedding, ref_embedding)
            similarities.append(sim)

        # Average similarity
        avg_similarity = np.mean(similarities)

        # Validation result
        is_valid = avg_similarity >= threshold

        result = {
            'is_valid': is_valid,
            'similarity': float(avg_similarity),
            'threshold': threshold,
            'validation_type': validation_type
        }

        if return_details:
            result['reference_similarities'] = [float(s) for s in similarities]
            result['min_similarity'] = float(np.min(similarities))
            result['max_similarity'] = float(np.max(similarities))

        return result

    def validate_batch(
        self,
        generated_images: List[str],
        reference_image_paths: List[str],
        validation_type: str = 'core'
    ) -> List[Dict]:
        """
        Validate multiple generated images.

        Args:
            generated_images: List of generated image paths
            reference_image_paths: List of reference image paths
            validation_type: 'core' or 'style'

        Returns:
            List of validation results (one per image)
        """
        logger.info(f"Validating {len(generated_images)} images...")

        results = []
        for i, img_path in enumerate(generated_images, 1):
            logger.info(f"Validating {i}/{len(generated_images)}: {Path(img_path).name}")

            result = self.validate_image(
                img_path,
                reference_image_paths,
                validation_type=validation_type
            )

            result['image_path'] = img_path
            result['image_index'] = i - 1

            status = "✅" if result['is_valid'] else "❌"
            logger.info(f"  {status} Similarity: {result['similarity']:.3f} (threshold: {result['threshold']})")

            results.append(result)

        # Summary
        valid_count = sum(1 for r in results if r['is_valid'])
        logger.info(f"\nValidation Summary: {valid_count}/{len(results)} passed ({valid_count/len(results)*100:.1f}%)")

        return results

    def clear_cache(self):
        """Clear all cached embeddings."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")


def demo():
    """
    Demo function to test CLIP validator.

    Usage:
        python obj2_midjourney_api/clip_validator.py
    """
    print("\n" + "="*80)
    print("CLIP Similarity Validator Demo")
    print("="*80 + "\n")

    # Check if reference images exist
    ref_dir = Path("data/reference_images")
    ref_images = list(ref_dir.glob("lulu_pig_ref_*.png"))

    if not ref_images:
        print("❌ No reference images found in data/reference_images/")
        print("   Please add reference images to test the validator")
        return

    print(f"✅ Found {len(ref_images)} reference images:")
    for img in ref_images:
        print(f"   - {img.name}")

    # Initialize validator
    print(f"\nInitializing CLIP Validator...")
    validator = CLIPValidator()

    # Test: Compare reference image to itself (should be ~1.0)
    print(f"\n{'-'*80}")
    print("Test 1: Self-similarity (reference vs itself)")
    print(f"{'-'*80}")

    result = validator.validate_image(
        generated_image_path=str(ref_images[0]),
        reference_image_paths=[str(ref_images[0])],
        validation_type='core'
    )

    print(f"Image: {ref_images[0].name}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Expected: ~1.0 (perfect match)")
    print(f"Status: {'✅ PASS' if 0.95 <= result['similarity'] <= 1.0 else '❌ FAIL'}")

    # Test: Compare two reference images (should be high similarity)
    if len(ref_images) >= 2:
        print(f"\n{'-'*80}")
        print("Test 2: Inter-reference similarity")
        print(f"{'-'*80}")

        result = validator.validate_image(
            generated_image_path=str(ref_images[0]),
            reference_image_paths=[str(ref_images[1])],
            validation_type='core'
        )

        print(f"Image 1: {ref_images[0].name}")
        print(f"Image 2: {ref_images[1].name}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Threshold: {result['threshold']} (core)")
        print(f"Status: {'✅ PASS' if result['is_valid'] else '❌ FAIL'}")

    print(f"\n{'='*80}")
    print("Demo completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    demo()
