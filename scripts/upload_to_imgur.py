"""
Upload Reference Images to Imgur

Simple script to upload Lulu Pig reference images to Imgur
and get permanent public URLs.

Author: Product Manager (John)
Usage: python scripts/upload_to_imgur.py
"""

import requests
import base64
from pathlib import Path
import json

# Imgur API (Anonymous Upload)
# Note: For anonymous upload, we use Imgur's public API
# Rate limit: ~50 uploads per hour
IMGUR_UPLOAD_URL = "https://api.imgur.com/3/image"

# Imgur Client ID (public, for anonymous uploads)
# This is a demo client ID, replace with your own if needed
CLIENT_ID = "546c25a59c58ad7"  # Public demo client ID


def upload_to_imgur(image_path: str, title: str = None, description: str = None) -> dict:
    """
    Upload an image to Imgur.

    Args:
        image_path: Path to image file
        title: Optional image title
        description: Optional image description

    Returns:
        Dictionary with upload result:
            - success: bool
            - link: str (direct image URL)
            - deletehash: str (for deletion)
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return {"success": False, "error": f"File not found: {image_path}"}

    # Read image as base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Prepare request
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }

    payload = {
        'image': image_data,
        'type': 'base64'
    }

    if title:
        payload['title'] = title
    if description:
        payload['description'] = description

    # Upload
    try:
        print(f"Uploading {image_path.name} to Imgur...")
        response = requests.post(IMGUR_UPLOAD_URL, headers=headers, data=payload)
        response.raise_for_status()

        data = response.json()

        if data.get('success'):
            result = {
                'success': True,
                'link': data['data']['link'],
                'deletehash': data['data']['deletehash'],
                'id': data['data']['id']
            }
            print(f"✅ Upload successful!")
            print(f"   Link: {result['link']}")
            print(f"   Delete hash: {result['deletehash']} (save this to delete later)")
            return result
        else:
            return {"success": False, "error": "Imgur API returned success=false"}

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def main():
    """Upload reference images to Imgur."""
    print("\n" + "="*80)
    print("Upload Reference Images to Imgur")
    print("="*80 + "\n")

    ref_dir = Path("data/reference_images")
    results = []

    # Find reference images
    ref_images = sorted(ref_dir.glob("lulu_pig_ref_*.png"))

    if not ref_images:
        print("❌ No reference images found in data/reference_images/")
        return

    print(f"Found {len(ref_images)} reference images:\n")

    # Upload each image
    for img_path in ref_images:
        title = f"Lulu Pig Reference - {img_path.stem}"
        description = "Reference image for Lulu Pig character IP design (FYP-RoleMarket)"

        result = upload_to_imgur(str(img_path), title=title, description=description)
        results.append({
            'filename': img_path.name,
            **result
        })
        print()

    # Summary
    print("="*80)
    print("Upload Summary")
    print("="*80 + "\n")

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}\n")

    if successful:
        print("Imgur URLs (copy these to config/reference_images.py):\n")
        print("CREF_URLS_CLOUD = [")
        for r in successful:
            print(f'    "{r["link"]}",')
        print("]\n")

        # Save to file
        output_file = Path("config/imgur_urls.txt")
        with open(output_file, 'w') as f:
            f.write("# Imgur URLs for Reference Images\n\n")
            for r in successful:
                f.write(f"{r['filename']}: {r['link']}\n")
                f.write(f"  Delete hash: {r['deletehash']}\n\n")

        print(f"URLs saved to: {output_file}")

        # Also save as JSON for programmatic access
        json_file = Path("config/imgur_upload_result.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Full results saved to: {json_file}")

    if failed:
        print("\nFailed uploads:")
        for r in failed:
            print(f"  ❌ {r['filename']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
