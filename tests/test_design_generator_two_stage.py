"""
Tests for DesignGeneratorWrapper two-stage generation integration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pytest
import os
from obj4_web_app.utils.design_generator import DesignGeneratorWrapper


class TestDesignGeneratorTwoStage:
    """Test two-stage generation in DesignGeneratorWrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create DesignGeneratorWrapper instance."""
        return DesignGeneratorWrapper(use_openai_api=True)

    @pytest.fixture
    def reference_image(self):
        """Path to reference image."""
        ref_path = PROJECT_ROOT / "data" / "reference_images" / "lulu_pig_ref_1.png"
        if not ref_path.exists():
            pytest.skip("Reference image not available")
        return str(ref_path)

    def test_two_stage_generator_available(self, wrapper):
        """Test that two_stage_generator is available."""
        assert hasattr(wrapper, 'two_stage_generator')
        assert wrapper.two_stage_generator is not None

    def test_generate_two_stage_method_exists(self, wrapper):
        """Test that generate_with_two_stage method exists."""
        assert hasattr(wrapper, 'generate_with_two_stage')
        assert callable(wrapper.generate_with_two_stage)

    def test_generate_with_two_stage_basic(self, wrapper, reference_image):
        """Test basic two-stage generation (integration test)."""
        if not os.getenv("GEMINI_OPENAI_API_KEY"):
            pytest.skip("No API key for integration test")

        result = wrapper.generate_with_two_stage(
            character_prompt="Lulu Pig",
            reference_image_path=reference_image,
            theme_elements="wearing Christmas sweater",
            theme_description="cozy Christmas scene"
        )

        # Verify success
        assert result['success'] is True
        assert 'final_image' in result
        assert 'final_image_path' in result
        assert 'clip_similarity' in result

        # Verify CLIP similarity exists
        assert result['clip_similarity'] >= 0.0

        # Cleanup
        if result.get('stage1_image_path'):
            Path(result['stage1_image_path']).unlink(missing_ok=True)
        if result.get('final_image_path'):
            Path(result['final_image_path']).unlink(missing_ok=True)
