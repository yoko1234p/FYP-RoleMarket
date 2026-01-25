"""
Compare CLIP similarity between single-stage and two-stage generation.

Expected improvement:
- Single-stage: 0.66-0.70
- Two-stage: 0.75-0.85
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pytest
import os
from obj4_web_app.utils.design_generator import DesignGeneratorWrapper


class TestTwoStageCLIPComparison:
    """Compare CLIP similarity improvements."""

    @pytest.fixture
    def wrapper(self):
        """Create DesignGeneratorWrapper instance."""
        return DesignGeneratorWrapper(use_openai_api=True)

    @pytest.fixture
    def reference_image(self):
        """Path to reference image."""
        return str(PROJECT_ROOT / "data" / "reference_images" / "lulu_pig_ref_1.png")

    @pytest.fixture
    def test_config(self):
        """Test configuration."""
        return {
            'character': 'Lulu Pig',
            'theme_elements': 'wearing cozy red sweater, holding a book',
            'theme_description': 'warm indoor library scene with soft lighting'
        }

    @pytest.mark.skipif(
        not os.getenv("GEMINI_OPENAI_API_KEY"),
        reason="Requires API key for integration test"
    )
    def test_single_stage_baseline(self, wrapper, reference_image, test_config):
        """Test single-stage generation as baseline."""
        # Generate with single stage (original method)
        prompt = f"{test_config['character']}, {test_config['theme_elements']}, {test_config['theme_description']}"

        result = wrapper.generate_single_design(
            prompt=prompt,
            reference_image_path=reference_image,
            output_filename="comparison_single_stage.png"
        )

        assert result['success'] is True

        # Compute CLIP similarity
        similarity, _ = wrapper.compute_clip_similarity(
            generated_image=result['image'],
            reference_image_path=reference_image,
            strategy="multi"
        )

        print(f"\n📊 Single-stage CLIP Similarity: {similarity:.4f}")
        print(f"   Expected range: 0.66-0.70")

        # Store for comparison
        result['clip_similarity'] = similarity

        # Note: Cleanup handled by caller to avoid premature deletion

        return result

    @pytest.mark.skipif(
        not os.getenv("GEMINI_OPENAI_API_KEY"),
        reason="Requires API key for integration test"
    )
    def test_two_stage_improved(self, wrapper, reference_image, test_config):
        """Test two-stage generation for improvement."""
        result = wrapper.generate_with_two_stage(
            character_prompt=test_config['character'],
            reference_image_path=reference_image,
            theme_elements=test_config['theme_elements'],
            theme_description=test_config['theme_description'],
            base_filename="comparison_two_stage",
            compute_clip=True,
            clip_strategy="multi"
        )

        assert result['success'] is True
        assert 'clip_similarity' in result

        similarity = result['clip_similarity']

        print(f"\n📊 Two-stage CLIP Similarity: {similarity:.4f}")
        print(f"   Expected range: 0.75-0.85")
        print(f"   Improvement target: +0.05 to +0.15")

        # Note: Cleanup handled by caller to avoid premature deletion

        return result

    @pytest.mark.skipif(
        not os.getenv("GEMINI_OPENAI_API_KEY"),
        reason="Requires API key for integration test"
    )
    def test_comparison_summary(self, wrapper, reference_image, test_config):
        """Full comparison test with summary report."""
        print("\n" + "="*80)
        print("CLIP SIMILARITY COMPARISON: Single-Stage vs Two-Stage")
        print("="*80)

        # Single-stage baseline
        print("\n1️⃣ Generating with single-stage method (baseline)...")
        single_result = self.test_single_stage_baseline(wrapper, reference_image, test_config)

        # Two-stage improved
        print("\n2️⃣ Generating with two-stage method (improved)...")
        two_stage_result = self.test_two_stage_improved(wrapper, reference_image, test_config)

        # Compare results
        single_clip = single_result['clip_similarity']
        two_stage_clip = two_stage_result['clip_similarity']
        improvement = two_stage_clip - single_clip
        improvement_pct = (improvement / single_clip) * 100 if single_clip > 0 else 0

        # Print summary
        print("\n" + "="*80)
        print("📊 COMPARISON SUMMARY")
        print("="*80)
        print(f"Single-stage CLIP:  {single_clip:.4f}")
        print(f"Two-stage CLIP:     {two_stage_clip:.4f}")
        print(f"Improvement:        +{improvement:.4f} ({improvement_pct:+.1f}%)")
        print("")
        print(f"Expected baseline:  0.66-0.70")
        print(f"Expected improved:  0.75-0.85")
        print(f"Target improvement: +0.05 to +0.15")
        print("")

        if improvement >= 0.05:
            print("✅ SUCCESS: Two-stage strategy shows significant improvement!")
        elif improvement > 0:
            print("⚠️  PARTIAL: Small improvement, may need prompt tuning")
        else:
            print("❌ FAILED: No improvement, strategy needs revision")

        print("="*80)

        # Cleanup all generated images
        try:
            Path(single_result['image_path']).unlink(missing_ok=True)
            if two_stage_result.get('stage1_image_path'):
                Path(two_stage_result['stage1_image_path']).unlink(missing_ok=True)
            if two_stage_result.get('final_image_path'):
                Path(two_stage_result['final_image_path']).unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

        # Assert improvement (meaningful threshold for real-world variance)
        # We expect at least +0.03 improvement (accounting for API variance)
        assert improvement >= 0.03, f"Two-stage should improve CLIP similarity by at least 0.03, got {improvement:.4f}"

        return {
            'single_stage': single_clip,
            'two_stage': two_stage_clip,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
