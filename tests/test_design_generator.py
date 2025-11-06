"""
Unit Tests for DesignGeneratorWrapper

測試 Obj 2 API Wrapper 功能。

Author: Developer (James)
Date: 2025-11-06
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.design_generator import (
    DesignGeneratorWrapper,
    DesignGenerationError,
    CLIPValidationError
)


class TestDesignGeneratorWrapper(unittest.TestCase):
    """Test cases for DesignGeneratorWrapper."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock Google Gemini client to avoid API calls
        self.mock_client_patch = patch(
            'obj4_web_app.utils.design_generator.GoogleGeminiImageClient'
        )
        self.mock_client = self.mock_client_patch.start()

        # Mock CLIP validator
        self.mock_validator_patch = patch(
            'obj4_web_app.utils.design_generator.CharacterFocusedValidator'
        )
        self.mock_validator = self.mock_validator_patch.start()

    def tearDown(self):
        """Clean up after tests."""
        self.mock_client_patch.stop()
        self.mock_validator_patch.stop()

    def test_initialization(self):
        """Test DesignGeneratorWrapper initialization."""
        wrapper = DesignGeneratorWrapper()
        self.assertIsNotNone(wrapper)
        self.assertIsNotNone(wrapper.client)

    def test_initialization_without_api_key(self):
        """Test initialization without API key raises error."""
        self.mock_client.side_effect = ValueError("API key not found")

        with self.assertRaises(DesignGenerationError):
            DesignGeneratorWrapper()

    def test_image_to_bytes(self):
        """Test image_to_bytes conversion."""
        wrapper = DesignGeneratorWrapper()

        # Create dummy image
        test_image = Image.new('RGB', (100, 100), color='red')

        # Convert to bytes
        img_bytes = wrapper.image_to_bytes(test_image)

        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 0)

    def test_get_average_similarity_with_valid_results(self):
        """Test get_average_similarity with valid results."""
        wrapper = DesignGeneratorWrapper()

        results = [
            {'success': True, 'clip_similarity': 0.85},
            {'success': True, 'clip_similarity': 0.90},
            {'success': True, 'clip_similarity': 0.80},
        ]

        avg_similarity = wrapper.get_average_similarity(results)

        expected = (0.85 + 0.90 + 0.80) / 3
        self.assertAlmostEqual(avg_similarity, expected, places=4)

    def test_get_average_similarity_with_failures(self):
        """Test get_average_similarity ignores failed results."""
        wrapper = DesignGeneratorWrapper()

        results = [
            {'success': True, 'clip_similarity': 0.85},
            {'success': False, 'error': 'API error'},
            {'success': True, 'clip_similarity': 0.90},
        ]

        avg_similarity = wrapper.get_average_similarity(results)

        expected = (0.85 + 0.90) / 2
        self.assertAlmostEqual(avg_similarity, expected, places=4)

    def test_get_average_similarity_empty_results(self):
        """Test get_average_similarity with empty results."""
        wrapper = DesignGeneratorWrapper()

        results = []
        avg_similarity = wrapper.get_average_similarity(results)

        self.assertEqual(avg_similarity, 0.0)

    @patch('obj4_web_app.utils.design_generator.Image')
    def test_generate_single_design_success(self, mock_image):
        """Test generate_single_design successful generation."""
        wrapper = DesignGeneratorWrapper()

        # Mock client.generate()
        wrapper.client.generate = Mock(return_value={
            'local_path': '/tmp/test_image.png',
            'prompt': 'test prompt',
            'cost': 0.0
        })

        # Mock Image.open()
        mock_img = Mock(spec=Image.Image)
        mock_img.convert = Mock(return_value=mock_img)
        mock_image.open = Mock(return_value=mock_img)

        result = wrapper.generate_single_design(
            prompt="Test prompt",
            reference_image_path="/path/to/ref.png",
            max_retries=1
        )

        self.assertTrue(result['success'])
        self.assertIsNotNone(result['image'])
        self.assertGreater(result['generation_time'], 0)

    def test_generate_single_design_failure(self):
        """Test generate_single_design handles errors."""
        wrapper = DesignGeneratorWrapper()

        # Mock client.generate() to raise error
        wrapper.client.generate = Mock(side_effect=Exception("API error"))

        result = wrapper.generate_single_design(
            prompt="Test prompt",
            reference_image_path="/path/to/ref.png",
            max_retries=1
        )

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIsNone(result['image'])


class TestDesignGeneratorWrapperIntegration(unittest.TestCase):
    """Integration tests (requires API keys)."""

    def test_generate_designs_validation(self):
        """Test generate_designs validates num_images parameter."""
        with patch('obj4_web_app.utils.design_generator.GoogleGeminiImageClient'):
            wrapper = DesignGeneratorWrapper()

            with self.assertRaises(ValueError):
                wrapper.generate_designs(
                    prompt="Test",
                    reference_image_path="/path/to/ref.png",
                    num_images=0  # Invalid
                )

            with self.assertRaises(ValueError):
                wrapper.generate_designs(
                    prompt="Test",
                    reference_image_path="/path/to/ref.png",
                    num_images=5  # Invalid (> 4)
                )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
