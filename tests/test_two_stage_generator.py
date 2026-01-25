"""
測試 TwoStageGenerator 兩階段生成策略

Author: Developer
Date: 2026-01-25
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class TestTwoStageGenerator(unittest.TestCase):
    """Test cases for TwoStageGenerator."""

    def test_two_stage_generator_initialization(self):
        """測試 TwoStageGenerator 初始化"""
        # 直接導入模組避免 __init__.py 的依賴問題
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "two_stage_generator",
            PROJECT_ROOT / "obj2_midjourney_api" / "two_stage_generator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TwoStageGenerator = module.TwoStageGenerator

        generator = TwoStageGenerator(
            gemini_client=Mock(),
            validator=Mock()
        )

        self.assertIsNotNone(generator)
        self.assertTrue(hasattr(generator, 'gemini_client'))
        self.assertTrue(hasattr(generator, 'validator'))

    def test_two_stage_generator_has_required_methods(self):
        """測試 TwoStageGenerator 有必要的方法"""
        # 直接導入模組避免 __init__.py 的依賴問題
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "two_stage_generator",
            PROJECT_ROOT / "obj2_midjourney_api" / "two_stage_generator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TwoStageGenerator = module.TwoStageGenerator

        generator = TwoStageGenerator(
            gemini_client=Mock(),
            validator=Mock()
        )

        # 必須有這些方法
        self.assertTrue(hasattr(generator, 'generate_stage1'))
        self.assertTrue(hasattr(generator, 'generate_stage2'))
        self.assertTrue(hasattr(generator, 'generate_two_stage'))
        self.assertTrue(callable(generator.generate_stage1))
        self.assertTrue(callable(generator.generate_stage2))
        self.assertTrue(callable(generator.generate_two_stage))


    def test_generate_stage1(self):
        """測試 Stage 1 基礎角色生成"""
        # 直接導入模組
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "two_stage_generator",
            PROJECT_ROOT / "obj2_midjourney_api" / "two_stage_generator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TwoStageGenerator = module.TwoStageGenerator

        # Mock Gemini client
        mock_gemini = Mock()
        mock_gemini.generate.return_value = {
            'local_path': '/fake/path/stage1.png',
            'cost': 0.0,
            'duration': 1.5
        }

        # Mock validator
        mock_validator = Mock()

        generator = TwoStageGenerator(
            gemini_client=mock_gemini,
            validator=mock_validator
        )

        # 執行 Stage 1 生成
        result = generator.generate_stage1(
            character_prompt="Lulu Pig, pink pig character",
            reference_image_path="/fake/ref.jpg",
            image_filename="test_stage1.png"
        )

        # 驗證結果
        self.assertIsNotNone(result)
        self.assertIn('local_path', result)
        self.assertEqual(result['local_path'], '/fake/path/stage1.png')

        # 驗證 Gemini client 被正確調用
        mock_gemini.generate.assert_called_once()
        call_args = mock_gemini.generate.call_args

        # 驗證 prompt 是極簡的（minimal style）
        self.assertIn('minimal style', call_args[1]['prompt'])
        self.assertIn('simple clean background', call_args[1]['prompt'])
        self.assertIn('no extra decorations', call_args[1]['prompt'])


if __name__ == '__main__':
    unittest.main()
