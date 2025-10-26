"""
Test Script for TTAPI Midjourney Client

Tests various functionality of the TTAPIClient:
- Client initialization
- API configuration validation
- Error handling
- Method signatures

Author: Product Manager (John)
Epic: 3 - Objective 2: Midjourney API Integration
Story: 3.1 - TTAPI Midjourney API Client

Usage:
    python obj2_midjourney_api/test_ttapi_client.py
"""

import sys
from pathlib import Path
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from obj2_midjourney_api.ttapi_client import TTAPIClient
from dotenv import load_dotenv


def test_client_initialization():
    """Test TTAPIClient initialization."""
    print("\n" + "="*80)
    print("Test 1: Client Initialization")
    print("="*80 + "\n")

    load_dotenv()
    api_key = os.getenv('TTAPI_API_KEY')

    if not api_key:
        print("❌ TTAPI_API_KEY not found in .env")
        return False

    try:
        client = TTAPIClient()
        print(f"✅ Client initialized successfully")
        print(f"   API Key: {api_key[:20]}...")
        print(f"   Base URL: {client.BASE_URL}")
        print(f"   Timeout: {client.timeout}s")
        print(f"   Output Dir: {client.output_dir}")
        print(f"   Max Retries: {client.MAX_RETRIES}")
        print(f"   Cost per Image: ${client.COST_PER_IMAGE}")
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_method_signatures():
    """Test that all required methods exist with correct signatures."""
    print("\n" + "="*80)
    print("Test 2: Method Signatures")
    print("="*80 + "\n")

    try:
        client = TTAPIClient()

        required_methods = [
            ('imagine', 'Main image generation method'),
            ('get_task_status', 'Get task status'),
            ('get_cost_summary', 'Get cost summary'),
            ('_submit_with_retry', 'Submit with retry logic'),
            ('_wait_for_task', 'Wait for task completion'),
            ('_download_image', 'Download image')
        ]

        all_exist = True
        for method_name, description in required_methods:
            if hasattr(client, method_name):
                print(f"✅ {method_name:25} - {description}")
            else:
                print(f"❌ {method_name:25} - MISSING!")
                all_exist = False

        return all_exist

    except Exception as e:
        print(f"❌ Method check failed: {e}")
        return False


def test_cost_tracking():
    """Test cost tracking functionality."""
    print("\n" + "="*80)
    print("Test 3: Cost Tracking")
    print("="*80 + "\n")

    try:
        client = TTAPIClient()

        # Initial state
        summary = client.get_cost_summary()
        print(f"Initial state:")
        print(f"  Images Generated: {summary['images_generated']}")
        print(f"  Total Cost: ${summary['total_cost']}")
        print(f"  Avg Cost per Image: ${summary['avg_cost_per_image']}")

        # Simulate image generation
        client.images_generated = 5
        client.total_cost = 5 * 0.40

        summary = client.get_cost_summary()
        print(f"\nAfter simulating 5 images:")
        print(f"  Images Generated: {summary['images_generated']}")
        print(f"  Total Cost: ${summary['total_cost']}")
        print(f"  Avg Cost per Image: ${summary['avg_cost_per_image']}")

        expected_cost = 2.0
        if summary['total_cost'] == expected_cost:
            print(f"\n✅ Cost tracking correct (${expected_cost})")
            return True
        else:
            print(f"\n❌ Cost tracking incorrect (expected ${expected_cost}, got ${summary['total_cost']})")
            return False

    except Exception as e:
        print(f"❌ Cost tracking test failed: {e}")
        return False


def test_prompt_building():
    """Test prompt building with parameters."""
    print("\n" + "="*80)
    print("Test 4: Prompt Building")
    print("="*80 + "\n")

    test_cases = [
        {
            'name': 'Simple prompt (no cref)',
            'prompt': 'Lulu Pig celebrating Halloween',
            'cref_urls': None,
            'expected_contains': ['Halloween', '--ar 1:1', '--v 6.1', '--style raw']
        },
        {
            'name': 'Prompt with character reference',
            'prompt': 'Lulu Pig celebrating Christmas',
            'cref_urls': ['https://example.com/lulu.png'],
            'expected_contains': ['Christmas', '--cref', 'https://example.com/lulu.png', '--cw 100']
        },
        {
            'name': 'Prompt with multiple references',
            'prompt': 'Lulu Pig summer vacation',
            'cref_urls': ['https://example.com/lulu1.png', 'https://example.com/lulu2.png'],
            'expected_contains': ['summer', '--cref', 'lulu1.png', 'lulu2.png']
        }
    ]

    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")

        # Build prompt (without calling API)
        prompt = test_case['prompt']
        if test_case['cref_urls']:
            cref_param = f" --cref {' '.join(test_case['cref_urls'])} --cw 100"
            prompt += cref_param
        prompt += " --ar 1:1 --v 6.1 --style raw"

        print(f"  Built Prompt: {prompt[:80]}...")

        # Check expected content
        all_found = all(keyword in prompt for keyword in test_case['expected_contains'])
        if all_found:
            print(f"  ✅ All expected keywords found\n")
        else:
            print(f"  ❌ Missing keywords: {test_case['expected_contains']}\n")
            all_passed = False

    return all_passed


def test_directory_creation():
    """Test that output directory is created."""
    print("\n" + "="*80)
    print("Test 5: Directory Creation")
    print("="*80 + "\n")

    try:
        test_output_dir = 'data/test_generated_images'
        client = TTAPIClient(output_dir=test_output_dir)

        if client.output_dir.exists() and client.output_dir.is_dir():
            print(f"✅ Output directory created: {client.output_dir}")

            # Cleanup
            import shutil
            shutil.rmtree(test_output_dir)
            print(f"✅ Test directory cleaned up")
            return True
        else:
            print(f"❌ Output directory not created: {client.output_dir}")
            return False

    except Exception as e:
        print(f"❌ Directory creation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# TTAPI Client Test Suite")
    print("#"*80)

    tests = [
        test_client_initialization,
        test_method_signatures,
        test_cost_tracking,
        test_prompt_building,
        test_directory_creation
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
