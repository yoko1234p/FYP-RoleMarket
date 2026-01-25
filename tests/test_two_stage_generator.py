"""
測試 TwoStageGenerator 兩階段生成策略

Author: Developer
Date: 2026-01-25
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pytest


class TestTwoStageGeneratorInit:
    """Test TwoStageGenerator initialization."""

    def test_init_default(self):
        """測試預設初始化"""
        with patch('obj2_midjourney_api.two_stage_generator.GeminiOpenAIImageClient'):
            from obj2_midjourney_api.two_stage_generator import TwoStageGenerator

            generator = TwoStageGenerator()
            assert generator is not None
            assert generator.client is not None

    def test_init_with_api_key(self):
        """測試使用自定義 API key 初始化"""
        with patch('obj2_midjourney_api.two_stage_generator.GeminiOpenAIImageClient') as mock_client:
            from obj2_midjourney_api.two_stage_generator import TwoStageGenerator

            generator = TwoStageGenerator(api_key="test_key")
            assert generator is not None
            mock_client.assert_called_once_with(api_key="test_key", use_preview=True)


class TestTwoStageGeneration:
    """Test two-stage generation workflow."""

    @pytest.fixture
    def generator(self):
        """Create TwoStageGenerator instance."""
        with patch('obj2_midjourney_api.two_stage_generator.GeminiOpenAIImageClient'):
            from obj2_midjourney_api.two_stage_generator import TwoStageGenerator
            return TwoStageGenerator()

    @pytest.fixture
    def mock_image(self):
        """Create a mock PIL Image."""
        # 創建一個簡單的 1x1 像素圖像
        img = Image.new('RGB', (1, 1), color='red')
        return img

    def test_generate_stage1_method_exists(self, generator):
        """測試 generate_stage1 方法存在"""
        assert hasattr(generator, 'generate_stage1')
        assert callable(generator.generate_stage1)

    def test_generate_stage2_method_exists(self, generator):
        """測試 generate_stage2 方法存在"""
        assert hasattr(generator, 'generate_stage2')
        assert callable(generator.generate_stage2)

    def test_generate_two_stage_method_exists(self, generator):
        """測試 generate_two_stage 方法存在"""
        assert hasattr(generator, 'generate_two_stage')
        assert callable(generator.generate_two_stage)

    def test_generate_stage1_success(self, generator, mock_image, tmp_path):
        """測試 Stage 1 成功生成"""
        # 創建臨時圖片文件
        temp_image_path = tmp_path / "stage1.png"
        mock_image.save(temp_image_path)

        # Mock Gemini client
        generator.client.generate = Mock(return_value={
            'local_path': str(temp_image_path),
            'cost': 0.0,
            'duration': 1.5
        })

        # 執行 Stage 1 生成
        result = generator.generate_stage1(
            character_prompt="Lulu Pig, pink pig character",
            reference_image_path="/fake/ref.jpg",
            output_filename="test_stage1.png"
        )

        # 驗證結果
        assert result is not None
        assert result['success'] is True
        assert 'image_path' in result
        assert 'image' in result
        assert 'prompt_used' in result
        assert 'generation_time' in result
        assert isinstance(result['image'], Image.Image)

        # 驗證 Gemini client 被正確調用
        generator.client.generate.assert_called_once()
        call_kwargs = generator.client.generate.call_args[1]

        # 驗證 prompt 是極簡的（minimal style）
        assert 'minimal style' in call_kwargs['prompt']
        assert 'simple clean background' in call_kwargs['prompt']
        assert 'no extra decorations' in call_kwargs['prompt']

    def test_generate_stage1_failure(self, generator):
        """測試 Stage 1 失敗處理"""
        # Mock Gemini client 拋出異常
        generator.client.generate = Mock(side_effect=Exception("API Error"))

        # 執行 Stage 1 生成
        result = generator.generate_stage1(
            character_prompt="Lulu Pig",
            reference_image_path="/fake/ref.jpg"
        )

        # 驗證失敗結果
        assert result is not None
        assert result['success'] is False
        assert result['error'] == "API Error"
        assert result['image_path'] is None
        assert result['image'] is None

    def test_generate_stage2_success(self, generator, mock_image, tmp_path):
        """測試 Stage 2 成功生成"""
        # 創建臨時圖片文件
        temp_image_path = tmp_path / "stage2.png"
        mock_image.save(temp_image_path)

        # Mock Gemini client
        generator.client.generate = Mock(return_value={
            'local_path': str(temp_image_path),
            'cost': 0.0,
            'duration': 2.0
        })

        # Stage 1 成功結果
        stage1_result = {
            'image_path': '/fake/stage1.png',
            'image': mock_image,
            'prompt_used': 'test prompt',
            'generation_time': 1.5,
            'success': True
        }

        # 執行 Stage 2 生成
        result = generator.generate_stage2(
            stage1_result=stage1_result,
            theme_elements="Santa hat, Christmas tree, gift boxes",
            theme_description="Cozy Christmas celebration at home",
            output_filename="test_stage2.png"
        )

        # 驗證結果
        assert result is not None
        assert result['success'] is True
        assert 'image_path' in result
        assert 'image' in result
        assert isinstance(result['image'], Image.Image)

        # 驗證 Gemini client 被正確調用
        generator.client.generate.assert_called_once()
        call_kwargs = generator.client.generate.call_args[1]

        # 驗證使用 Stage 1 圖片作為 reference
        assert call_kwargs['reference_images'] == ['/fake/stage1.png']

        # 驗證 prompt 包含主題元素和場景描述
        prompt = call_kwargs['prompt']
        assert 'Santa hat' in prompt
        assert 'Christmas tree' in prompt
        assert 'Cozy Christmas celebration' in prompt

        # 驗證 prompt 強調保持角色一致性
        assert 'keep the character appearance EXACTLY the same' in prompt
        assert 'Do not change' in prompt

    def test_generate_stage2_stage1_failed(self, generator):
        """測試 Stage 2 在 Stage 1 失敗時的處理"""
        # Stage 1 失敗結果
        stage1_result = {
            'image_path': None,
            'image': None,
            'prompt_used': 'test prompt',
            'generation_time': 1.5,
            'success': False,
            'error': 'Stage 1 failed'
        }

        # 執行 Stage 2 生成
        result = generator.generate_stage2(
            stage1_result=stage1_result,
            theme_elements="Santa hat",
            theme_description="Christmas"
        )

        # 驗證結果
        assert result is not None
        assert result['success'] is False
        assert 'Stage 1 failed' in result['error']
        assert result['image_path'] is None
        assert result['image'] is None

    def test_generate_stage2_failure(self, generator, mock_image):
        """測試 Stage 2 失敗處理"""
        # Mock Gemini client 拋出異常
        generator.client.generate = Mock(side_effect=Exception("API Error"))

        # Stage 1 成功結果
        stage1_result = {
            'image_path': '/fake/stage1.png',
            'image': mock_image,
            'success': True
        }

        # 執行 Stage 2 生成
        result = generator.generate_stage2(
            stage1_result=stage1_result,
            theme_elements="Santa hat",
            theme_description="Christmas"
        )

        # 驗證失敗結果
        assert result is not None
        assert result['success'] is False
        assert result['error'] == "API Error"
        assert result['image_path'] is None
        assert result['image'] is None


class TestTwoStageGenerationIntegration:
    """Integration tests (requires API key)."""

    @pytest.fixture
    def reference_image(self):
        """Path to reference image."""
        ref_path = PROJECT_ROOT / "data" / "reference_images" / "lulu_pig_ref_1.jpg"
        if not ref_path.exists():
            pytest.skip("Reference image not available")
        return str(ref_path)

    def test_generate_stage1_integration(self, reference_image):
        """測試 Stage 1 整合（需要真實 API）"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        if not os.getenv("GEMINI_OPENAI_API_KEY"):
            pytest.skip("No API key for integration test")

        from obj2_midjourney_api.two_stage_generator import TwoStageGenerator

        generator = TwoStageGenerator()
        result = generator.generate_stage1(
            character_prompt="Lulu Pig",
            reference_image_path=reference_image,
            output_filename="test_integration_stage1.png"
        )

        # 驗證結果
        assert result['success'] is True
        assert Path(result['image_path']).exists()
        assert isinstance(result['image'], Image.Image)
        assert result['generation_time'] > 0

    def test_generate_stage2_integration(self, reference_image):
        """測試 Stage 2 整合（需要真實 API）"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        if not os.getenv("GEMINI_OPENAI_API_KEY"):
            pytest.skip("No API key for integration test")

        from obj2_midjourney_api.two_stage_generator import TwoStageGenerator

        generator = TwoStageGenerator()

        # 先生成 Stage 1
        stage1_result = generator.generate_stage1(
            character_prompt="Lulu Pig",
            reference_image_path=reference_image,
            output_filename="test_integration_stage1.png"
        )

        assert stage1_result['success'] is True

        # 再生成 Stage 2
        stage2_result = generator.generate_stage2(
            stage1_result=stage1_result,
            theme_elements="Santa hat, Christmas decorations",
            theme_description="Cozy Christmas indoor scene",
            output_filename="test_integration_stage2.png"
        )

        # 驗證結果
        assert stage2_result['success'] is True
        assert Path(stage2_result['image_path']).exists()
        assert isinstance(stage2_result['image'], Image.Image)
        assert stage2_result['generation_time'] > 0
