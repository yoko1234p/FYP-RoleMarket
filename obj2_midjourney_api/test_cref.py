"""
Test Script for --cref (Character Reference) Parameter

Tests TTAPI client with character reference images to ensure
character consistency in generated images.

Test Cases:
1. Baseline: Generate without --cref
2. Low weight: --cref with weight 50
3. Medium weight: --cref with weight 75
4. High weight: --cref with weight 100

Author: Product Manager (John)
Epic: 3 - Objective 2: Midjourney API Integration
Story: 3.2 - Test --cref with Reference Images

Usage:
    python obj2_midjourney_api/test_cref.py [--actual-run]

    Without --actual-run flag: Dry run (no API calls)
    With --actual-run flag: Execute actual API calls (costs $0.40 per image)
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from obj2_midjourney_api.ttapi_client import TTAPIClient


def check_reference_images():
    """Check if reference images exist."""
    print("\n" + "="*80)
    print("Checking Reference Images")
    print("="*80 + "\n")

    ref_dir = Path("data/reference_images")
    ref_images = list(ref_dir.glob("lulu_pig_ref_*.png"))

    if not ref_images:
        print("❌ No reference images found in data/reference_images/")
        print("   Expected: lulu_pig_ref_1.png, lulu_pig_ref_2.png")
        return False

    for img in ref_images:
        size_kb = img.stat().st_size / 1024
        print(f"✅ Found: {img.name} ({size_kb:.1f} KB)")

    print(f"\nTotal: {len(ref_images)} reference images\n")
    return True


def upload_reference_images_info():
    """
    Display information about uploading reference images.

    Note: TTAPI requires publicly accessible URLs for --cref.
    Reference images need to be uploaded to:
    - Discord (recommended by Midjourney)
    - TTAPI image hosting (if available)
    - Public image hosting service (imgur, etc.)
    """
    print("\n" + "="*80)
    print("Reference Image Upload Requirements")
    print("="*80 + "\n")

    print("⚠️  IMPORTANT: --cref requires publicly accessible image URLs")
    print("\nOptions for hosting reference images:")
    print("1. Discord CDN (recommended by Midjourney)")
    print("   - Upload to Discord channel")
    print("   - Right-click → Copy Image Address")
    print("   - Use the CDN URL (starts with https://cdn.discordapp.com/)")
    print("\n2. TTAPI Image Upload API (if available)")
    print("   - Check TTAPI docs for upload endpoint")
    print("\n3. Public Image Hosting")
    print("   - Imgur, Imgbb, or similar services")
    print("\n4. Your own server/CDN")
    print("   - Ensure HTTPS and publicly accessible")

    print("\n" + "-"*80)
    print("For this test, you need to:")
    print("1. Upload data/reference_images/lulu_pig_ref_1.png")
    print("2. Upload data/reference_images/lulu_pig_ref_2.png")
    print("3. Update CREF_URLS in this script with the URLs")
    print("-"*80 + "\n")


def test_cref_variations(dry_run=True):
    """
    Test different --cref weight values.

    Args:
        dry_run: If True, only show what would be done (no API calls)
    """
    print("\n" + "="*80)
    print("Testing --cref Weight Variations")
    print("="*80 + "\n")

    # Imgur URLs for Lulu Pig reference images (permanent, free hosting)
    # Uploaded: 2025-10-26, Status: Active (HTTP 200 OK)
    CREF_URLS = [
        "https://i.imgur.com/m0syInf.png",  # lulu_pig_ref_1.png (241KB)
        "https://i.imgur.com/t7zZotG.png"   # lulu_pig_ref_2.png (196KB)
    ]

    # Check if URLs are placeholder
    if any("REPLACE" in url or "..." in url for url in CREF_URLS):
        print("❌ Reference image URLs not configured!")
        print("   Update CREF_URLS in this script with actual public URLs")
        print("\nSkipping API tests (dry run mode)")
        dry_run = True

    # Test prompt
    base_prompt = "Lulu Pig celebrating Halloween, wearing witch costume, holding pumpkin, kawaii style"

    # Test cases
    test_cases = [
        {
            'name': 'Baseline (No --cref)',
            'cref_urls': None,
            'cref_weight': 100,
            'filename': 'test_baseline_no_cref.png'
        },
        {
            'name': 'Low Weight (--cw 50)',
            'cref_urls': CREF_URLS[:1],  # Use first reference only
            'cref_weight': 50,
            'filename': 'test_cref_weight_50.png'
        },
        {
            'name': 'Medium Weight (--cw 75)',
            'cref_urls': CREF_URLS[:1],
            'cref_weight': 75,
            'filename': 'test_cref_weight_75.png'
        },
        {
            'name': 'High Weight (--cw 100)',
            'cref_urls': CREF_URLS[:1],
            'cref_weight': 100,
            'filename': 'test_cref_weight_100.png'
        },
        {
            'name': 'Multiple References (--cw 100)',
            'cref_urls': CREF_URLS,  # Use both references
            'cref_weight': 100,
            'filename': 'test_cref_multiple_refs.png'
        }
    ]

    if dry_run:
        print("🔍 DRY RUN MODE (No actual API calls)\n")
        print("Test cases that would be executed:")
        print("-" * 80)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print(f"  Prompt: {base_prompt}")

            if test_case['cref_urls']:
                print(f"  --cref: {', '.join(test_case['cref_urls'])}")
                print(f"  --cw: {test_case['cref_weight']}")
            else:
                print(f"  --cref: None (baseline)")

            print(f"  Output: data/generated_images/{test_case['filename']}")
            print(f"  Estimated Cost: $0.40")

        total_cost = len(test_cases) * 0.40
        print(f"\n" + "-" * 80)
        print(f"Total Test Cases: {len(test_cases)}")
        print(f"Estimated Total Cost: ${total_cost:.2f}")
        print("\nTo execute actual tests: python obj2_midjourney_api/test_cref.py --actual-run")

    else:
        print("🚀 EXECUTING ACTUAL API CALLS\n")
        print("⚠️  This will cost approximately $0.40 per image")

        try:
            client = TTAPIClient()
            results = []

            for i, test_case in enumerate(test_cases, 1):
                print(f"\n{'-' * 80}")
                print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
                print(f"{'-' * 80}\n")

                try:
                    result = client.imagine(
                        prompt=base_prompt,
                        cref_urls=test_case['cref_urls'],
                        cref_weight=test_case['cref_weight'],
                        wait_for_completion=True,
                        save_image=True,
                        image_filename=test_case['filename']
                    )

                    print(f"✅ Test {i} completed successfully")
                    print(f"   Task ID: {result['task_id']}")
                    print(f"   Duration: {result.get('duration', 0):.2f}s")
                    print(f"   Image: {result.get('local_path', 'N/A')}")
                    print(f"   Cost: ${result.get('cost', 0)}")

                    results.append({
                        'test_name': test_case['name'],
                        'success': True,
                        'result': result
                    })

                except Exception as e:
                    print(f"❌ Test {i} failed: {e}")
                    results.append({
                        'test_name': test_case['name'],
                        'success': False,
                        'error': str(e)
                    })

            # Summary
            print("\n" + "=" * 80)
            print("Test Summary")
            print("=" * 80 + "\n")

            passed = sum(1 for r in results if r['success'])
            failed = len(results) - passed

            for i, result in enumerate(results, 1):
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"{status}  Test {i}: {result['test_name']}")

            print(f"\nPassed: {passed}/{len(results)}")
            print(f"Failed: {failed}/{len(results)}")

            # Cost summary
            cost_summary = client.get_cost_summary()
            print(f"\nTotal Cost: ${cost_summary['total_cost']}")
            print(f"Images Generated: {cost_summary['images_generated']}")

        except Exception as e:
            print(f"\n❌ Test execution failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(
        description='Test TTAPI --cref parameter with different weights'
    )
    parser.add_argument(
        '--actual-run',
        action='store_true',
        help='Execute actual API calls (costs $0.40 per image)'
    )
    args = parser.parse_args()

    print("\n" + "#"*80)
    print("# TTAPI --cref Parameter Test Suite")
    print("#"*80)

    # Check reference images
    if not check_reference_images():
        return

    # Display upload info
    upload_reference_images_info()

    # Run tests
    test_cref_variations(dry_run=not args.actual_run)

    print("\n" + "#"*80)
    print("# Test Completed")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
