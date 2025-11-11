"""
Configuration for Streamlit Web Application

管理 API keys、常數和應用配置。
支援 Streamlit Secrets 和 .env 檔案。

Author: Developer (James)
Date: 2025-11-06
Updated: 2025-11-07 (Added Streamlit Secrets support)
Version: 1.1
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file (local development)
load_dotenv()


def get_secret(key: str, default=None):
    """
    Get secret from Streamlit secrets or environment variable.

    Priority:
    1. Streamlit Secrets (st.secrets) - for production deployment
    2. Environment variable (.env) - for local development

    Args:
        key: Secret key name
        default: Default value if not found

    Returns:
        Secret value or default

    Example:
        >>> api_key = get_secret("GOOGLE_API_KEY")
        >>> api_key = get_secret("OPTIONAL_KEY", default="fallback_value")
    """
    # Try Streamlit secrets first (production)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        # st.secrets may not be available in non-Streamlit contexts
        pass

    # Fallback to environment variable (local dev)
    return os.getenv(key, default)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Detect Cloud environment
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") == "true" or \
                     os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud" or \
                     hasattr(st, 'secrets')

# API Keys (supports both .env and Streamlit Secrets)
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GEMINI_OPENAI_API_KEY = get_secret("GEMINI_OPENAI_API_KEY")
# Support both GPT_API_TOKEN and GPT_API_FREE_KEY
GPT_API_TOKEN = get_secret("GPT_API_TOKEN") or get_secret("GPT_API_FREE_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
TTAPI_API_KEY = get_secret("TTAPI_API_KEY")

# Validate required API keys
if not GPT_API_TOKEN:
    raise ValueError(
        "GPT_API_TOKEN or GPT_API_FREE_KEY not found in environment. "
        "Please set it in .env file or Streamlit Secrets."
    )

# Optional API keys (warnings only)
if not GOOGLE_API_KEY and not GEMINI_OPENAI_API_KEY:
    import warnings
    warnings.warn(
        "Neither GOOGLE_API_KEY nor GEMINI_OPENAI_API_KEY found. "
        "Image generation features will be unavailable."
    )

# Application constants
APP_TITLE = "AI 角色設計與需求預測系統"
APP_VERSION = "1.0"

# Obj 1 - Trend Analysis
DEFAULT_REGION = "HK"
DEFAULT_LANG = "zh-TW"
MAX_KEYWORDS = 10
TRENDS_CACHE_TTL = 3600  # 1 hour in seconds

# Obj 2 - Image Generation
CLIP_SIMILARITY_THRESHOLD = 0.80
REFERENCE_IMAGES_DIR = PROJECT_ROOT / "data" / "reference_images"

# Obj 3 - Forecasting
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "models" / "transformer_lulu" / "best_transformer_model.pth"
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_NHEAD = 8

# Streamlit cache settings
CACHE_TTL_TRENDS = 3600  # 1 hour
CACHE_TTL_MODEL = None  # Never expire (resource cache)

# Cloud environment optimizations
if IS_STREAMLIT_CLOUD:
    # Reduce resource usage in Cloud
    DEFAULT_NUM_IMAGES = 2  # Generate 2 images instead of 4
    ENABLE_MULTITHREADING = True  # Keep enabled (already optimized for Cloud)
    CLIP_MODEL_CACHE = True  # Cache CLIP model
else:
    # Local development settings
    DEFAULT_NUM_IMAGES = 4
    ENABLE_MULTITHREADING = True
    CLIP_MODEL_CACHE = True

# Error messages (Traditional Chinese)
ERROR_MESSAGES = {
    "empty_keywords": "請輸入至少一個關鍵字",
    "api_timeout": "API 請求超時，請稍後重試",
    "api_error": "API 調用失敗：{error}",
    "invalid_input": "輸入格式無效：{details}",
    "model_load_error": "模型載入失敗，請檢查模型檔案",
}

# Success messages
SUCCESS_MESSAGES = {
    "trends_fetched": "✅ 趨勢數據提取成功！",
    "prompt_generated": "✅ Prompt 生成成功！",
    "images_generated": "✅ 圖片生成成功！",
    "prediction_complete": "✅ 銷量預測完成！",
}
