"""
測試 Gemini OpenAI API 以調試錯誤
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from obj2_midjourney_api.gemini_openai_client import GeminiOpenAIImageClient

def test_simple_generation():
    """測試簡單的圖像生成"""
    print("="*80)
    print("測試 Gemini OpenAI API 圖像生成")
    print("="*80)

    # Initialize client
    client = GeminiOpenAIImageClient(use_preview=True)
    print(f"\n✅ Client initialized (Model: {client.model})")

    # Test prompt
    prompt = "A cute pink pig wearing a Santa hat, festive Christmas theme"
    print(f"\n📝 Test Prompt: {prompt}")

    # Reference image
    ref_image = PROJECT_ROOT / "data" / "reference_images" / "lulu_pig_ref_1.png"
    if not ref_image.exists():
        print(f"\n⚠️ Reference image not found: {ref_image}")
        print("Trying alternative reference images...")
        ref_image = PROJECT_ROOT / "data" / "reference_images" / "lulu_pig_ref_1.jpg"

    if not ref_image.exists():
        print(f"\n❌ No reference image found!")
        return

    print(f"\n🖼️ Reference Image: {ref_image}")

    try:
        # Generate
        print("\n🚀 Starting generation...")
        result = client.generate(
            prompt=prompt,
            reference_images=[str(ref_image)],
            image_filename="test_debug.png",
            max_retries=1
        )

        print("\n✅ Generation successful!")
        print(f"📁 Saved to: {result['local_path']}")
        print(f"⏱️ Duration: {result['duration']:.2f}s")
        print(f"💰 Cost: ${result['cost']:.4f}")

    except Exception as e:
        print(f"\n❌ Generation failed!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == '__main__':
    test_simple_generation()
