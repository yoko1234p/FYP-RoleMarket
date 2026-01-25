"""
Real API integration test for two-stage generation.

This test actually calls the Gemini API and generates real images.
Use sparingly to avoid API rate limits and costs.

Run with:
    pytest tests/integration/test_two_stage_real_api.py -v -s --api
"""

import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import pytest
import os
from obj4_web_app.utils.design_generator import DesignGeneratorWrapper


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "api: tests that call real APIs (deselect with '-m \"not api\"')"
    )


@pytest.mark.api
class TestTwoStageRealAPI:
    """Real API integration tests."""

    @pytest.fixture(scope="class")
    def wrapper(self):
        """Create DesignGeneratorWrapper instance."""
        if not os.getenv("GEMINI_OPENAI_API_KEY"):
            pytest.skip("GEMINI_OPENAI_API_KEY not set")
        return DesignGeneratorWrapper(use_openai_api=True)

    @pytest.fixture(scope="class")
    def reference_image(self):
        """Path to reference image."""
        path = PROJECT_ROOT / "data" / "reference_images" / "lulu_pig_ref_1.png"
        if not path.exists():
            pytest.skip(f"Reference image not found: {path}")
        return str(path)

    @pytest.fixture(scope="class")
    def output_dir(self):
        """Create output directory for test results."""
        output_path = PROJECT_ROOT / "data" / "test_results" / "two_stage_integration"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def test_christmas_scene(self, wrapper, reference_image, output_dir):
        """Test two-stage generation with Christmas theme."""
        print("\n" + "="*80)
        print("TEST: Christmas Scene Generation")
        print("="*80)

        result = wrapper.generate_with_two_stage(
            character_prompt="Lulu Pig",
            reference_image_path=reference_image,
            theme_elements="wearing cozy red Christmas sweater, holding a gift box",
            theme_description="warm indoor Christmas scene with Christmas tree and decorations, soft golden lighting",
            base_filename="integration_test_christmas",
            compute_clip=True,
            clip_strategy="multi"
        )

        # Print results
        print(f"\n✅ Generation complete!")
        print(f"   Success: {result['success']}")
        print(f"   Total time: {result['total_time']:.2f}s")
        print(f"   CLIP Similarity: {result['clip_similarity']:.4f}")
        print(f"   Stage 1 image: {result['stage1_image_path']}")
        print(f"   Final image: {result['final_image_path']}")

        # Save metadata
        metadata = {
            'test': 'christmas_scene',
            'timestamp': time.time(),
            'success': result['success'],
            'clip_similarity': float(result['clip_similarity']),
            'total_time': result['total_time'],
            'stage1_prompt': result['stage1_prompt'],
            'stage2_prompt': result['stage2_prompt'],
            'stage1_image_path': result['stage1_image_path'],
            'final_image_path': result['final_image_path']
        }

        metadata_path = output_dir / "christmas_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Metadata saved to: {metadata_path}")

        # Assertions
        assert result['success'] is True
        assert result['clip_similarity'] > 0.0
        assert Path(result['stage1_image_path']).exists()
        assert Path(result['final_image_path']).exists()

        # Expected improvement threshold
        print(f"\n📊 Quality Check:")
        print(f"   Current CLIP: {result['clip_similarity']:.4f}")
        print(f"   Target range: 0.75-0.85")

        if result['clip_similarity'] >= 0.75:
            print(f"   ✅ Excellent! Exceeds target threshold")
        elif result['clip_similarity'] >= 0.70:
            print(f"   ⚠️  Good, approaching target")
        else:
            print(f"   ❌ Below target, needs improvement")

        return result

    def test_summer_scene(self, wrapper, reference_image, output_dir):
        """Test two-stage generation with Summer theme."""
        print("\n" + "="*80)
        print("TEST: Summer Scene Generation")
        print("="*80)

        result = wrapper.generate_with_two_stage(
            character_prompt="Lulu Pig",
            reference_image_path=reference_image,
            theme_elements="wearing sunglasses and beach hat, holding ice cream",
            theme_description="sunny beach scene with blue sky and ocean, bright summer day",
            base_filename="integration_test_summer",
            compute_clip=True,
            clip_strategy="multi"
        )

        print(f"\n✅ Generation complete!")
        print(f"   CLIP Similarity: {result['clip_similarity']:.4f}")

        # Save metadata
        metadata_path = output_dir / "summer_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test': 'summer_scene',
                'clip_similarity': float(result['clip_similarity']),
                'total_time': result['total_time']
            }, f, indent=2)

        assert result['success'] is True
        assert result['clip_similarity'] > 0.0

        return result

    def test_comparison_report(self, wrapper, reference_image, output_dir):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TWO-STAGE vs SINGLE-STAGE COMPARISON")
        print("="*80)

        # Single-stage baseline
        print("\n1️⃣ Single-stage generation (baseline)...")
        single_start = time.time()
        single_result = wrapper.generate_single_design(
            prompt="Lulu Pig, wearing cozy sweater, reading a book, warm library scene",
            reference_image_path=reference_image,
            output_filename="integration_single_baseline.png"
        )
        single_time = time.time() - single_start

        single_clip, _ = wrapper.compute_clip_similarity(
            generated_image=single_result['image'],
            reference_image_path=reference_image,
            strategy="multi"
        )

        print(f"   CLIP: {single_clip:.4f}, Time: {single_time:.2f}s")

        # Two-stage improved
        print("\n2️⃣ Two-stage generation (improved)...")
        two_stage_result = wrapper.generate_with_two_stage(
            character_prompt="Lulu Pig",
            reference_image_path=reference_image,
            theme_elements="wearing cozy sweater, reading a book",
            theme_description="warm library scene with bookshelves",
            base_filename="integration_two_stage_improved",
            compute_clip=True,
            clip_strategy="multi"
        )

        print(f"   CLIP: {two_stage_result['clip_similarity']:.4f}, Time: {two_stage_result['total_time']:.2f}s")

        # Generate report
        improvement = two_stage_result['clip_similarity'] - single_clip
        improvement_pct = (improvement / single_clip * 100) if single_clip > 0 else 0

        report = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'single_stage': {
                'clip_similarity': float(single_clip),
                'generation_time': single_time,
                'image_path': single_result['image_path']
            },
            'two_stage': {
                'clip_similarity': float(two_stage_result['clip_similarity']),
                'generation_time': two_stage_result['total_time'],
                'stage1_image_path': two_stage_result['stage1_image_path'],
                'final_image_path': two_stage_result['final_image_path']
            },
            'comparison': {
                'clip_improvement': float(improvement),
                'clip_improvement_pct': float(improvement_pct),
                'time_overhead': two_stage_result['total_time'] - single_time,
                'time_overhead_pct': ((two_stage_result['total_time'] - single_time) / single_time * 100)
            },
            'verdict': 'SUCCESS' if improvement >= 0.05 else 'PARTIAL' if improvement > 0 else 'FAILED'
        }

        # Save report
        report_path = output_dir / "comparison_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "="*80)
        print("📊 FINAL COMPARISON REPORT")
        print("="*80)
        print(f"Single-stage CLIP:  {single_clip:.4f}")
        print(f"Two-stage CLIP:     {two_stage_result['clip_similarity']:.4f}")
        print(f"Improvement:        +{improvement:.4f} ({improvement_pct:+.1f}%)")
        print(f"\nTime overhead:      +{report['comparison']['time_overhead']:.2f}s ({report['comparison']['time_overhead_pct']:+.1f}%)")
        print(f"\nVerdict:            {report['verdict']}")
        print(f"\n📄 Full report saved to: {report_path}")
        print("="*80)

        # Cleanup
        Path(single_result['image_path']).unlink(missing_ok=True)

        # Assertion
        assert improvement >= 0.0, "Two-stage should not decrease similarity"

        return report
