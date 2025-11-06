"""
測試 Streamlit App 模組導入

驗證所有必要模組能正確導入。
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

print("=" * 60)
print("Testing Streamlit App Imports")
print("=" * 60)

# Test 1: Basic imports
print("\n1️⃣ Testing basic imports...")
try:
    import streamlit as st
    print("✅ streamlit imported")
except Exception as e:
    print(f"❌ streamlit import failed: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("✅ dotenv imported")
except Exception as e:
    print(f"❌ dotenv import failed: {e}")
    sys.exit(1)

# Test 2: Config module
print("\n2️⃣ Testing config module...")
try:
    from obj4_web_app import config
    print(f"✅ config imported")
    print(f"   - DEFAULT_REGION: {config.DEFAULT_REGION}")
    print(f"   - DEFAULT_LANG: {config.DEFAULT_LANG}")
    print(f"   - GPT_API_TOKEN: {'✅ Set' if config.GPT_API_TOKEN else '❌ Not set'}")
except Exception as e:
    print(f"❌ config import failed: {e}")
    sys.exit(1)

# Test 3: TrendsAPIWrapper
print("\n3️⃣ Testing TrendsAPIWrapper...")
try:
    from obj4_web_app.utils.trends_api import TrendsAPIWrapper
    print("✅ TrendsAPIWrapper imported")

    # Try to initialize
    wrapper = TrendsAPIWrapper()
    print("✅ TrendsAPIWrapper initialized")

    # Test keyword extraction
    keywords = wrapper.extract_keywords_simple("春節, 紅色, 喜慶")
    assert keywords == ["春節", "紅色", "喜慶"]
    print(f"✅ extract_keywords_simple works: {keywords}")

except Exception as e:
    print(f"❌ TrendsAPIWrapper test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: PromptGenerator (Obj 1)
print("\n4️⃣ Testing PromptGenerator (Obj 1)...")
try:
    from obj1_nlp_prompt.prompt_generator import PromptGenerator
    print("✅ PromptGenerator imported")

    template_path = PROJECT_ROOT / 'obj1_nlp_prompt' / 'templates' / 'prompt_template.txt'
    character_desc_path = PROJECT_ROOT / 'data' / 'character_descriptions' / 'lulu_pig.txt'

    if template_path.exists() and character_desc_path.exists():
        print(f"✅ Required files exist")
    else:
        print(f"❌ Missing files:")
        print(f"   - template: {template_path.exists()}")
        print(f"   - character desc: {character_desc_path.exists()}")

except Exception as e:
    print(f"❌ PromptGenerator test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All import tests completed!")
print("=" * 60)
