"""
Objective 2: Commercial-Grade Design Generation

This module handles:
- TTAPI Midjourney API integration (PPU mode)
- Character consistency via --cref parameter
- CLIP ViT-Large/14 similarity validation (>0.75 core, >0.60 style)
- Error handling & image caching
- Cost tracking

Target: Generate 28 character-consistent designs (~$10-30 total cost)
"""

from .ttapi_client import TTAPIClient

__all__ = ['TTAPIClient']
__version__ = "1.0.0"
