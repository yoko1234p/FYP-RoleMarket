"""
Reference Images 配置

定義 Lulu豬 官方 IP reference images 路徑。

Author: Product Manager (John)
Date: 2025-10-27
"""

from pathlib import Path
from typing import Dict

# Reference Images 基礎路徑
REF_BASE_DIR = Path("data/reference_images")

# 官方 IP Reference Images
REFERENCE_IMAGES: Dict[str, str] = {
    # ref_1: 白色背景，正面站立，純淨角色展示
    # 適合：需要極簡背景的場景
    'ref_1': str(REF_BASE_DIR / "lulu_pig_ref_1.png"),

    # ref_2: （如果有）
    'ref_2': str(REF_BASE_DIR / "lulu_pig_ref_2.png"),

    # ref_3: 藍色背景，戶外場景，帶 LuLu 小包包
    # 適合：需要場景感的生成，推薦作為主要 reference
    'ref_3': str(REF_BASE_DIR / "lulu_pig_ref_3.jpg"),
}

# 預設 Reference Image（推薦使用 ref_3，因為有場景和道具）
DEFAULT_REFERENCE = REFERENCE_IMAGES['ref_3']

# 用於特定用途的 Reference
SIMPLE_REFERENCE = REFERENCE_IMAGES['ref_1']  # 純淨背景
SCENE_REFERENCE = REFERENCE_IMAGES['ref_3']    # 帶場景


def get_reference_image(ref_type: str = 'default') -> str:
    """
    獲取 reference image 路徑。

    Args:
        ref_type: 'default', 'simple', 'scene', 'ref_1', 'ref_2', 'ref_3'

    Returns:
        Reference image 路徑
    """
    if ref_type == 'default':
        return DEFAULT_REFERENCE
    elif ref_type == 'simple':
        return SIMPLE_REFERENCE
    elif ref_type == 'scene':
        return SCENE_REFERENCE
    elif ref_type in REFERENCE_IMAGES:
        return REFERENCE_IMAGES[ref_type]
    else:
        raise ValueError(f"Unknown ref_type: {ref_type}. Available: {list(REFERENCE_IMAGES.keys())}")


def validate_references() -> Dict[str, bool]:
    """
    驗證 reference images 是否存在。

    Returns:
        Dictionary mapping ref_name -> exists (bool)
    """
    return {
        name: Path(path).exists()
        for name, path in REFERENCE_IMAGES.items()
    }


if __name__ == '__main__':
    print("Lulu豬 Reference Images 配置")
    print("="*60)

    validation = validate_references()
    for name, exists in validation.items():
        status = "✅" if exists else "❌"
        path = REFERENCE_IMAGES[name]
        print(f"{status} {name:10s}: {path}")

    print(f"\n預設 Reference: {DEFAULT_REFERENCE}")
    print(f"簡單背景 Reference: {SIMPLE_REFERENCE}")
    print(f"場景 Reference: {SCENE_REFERENCE}")
