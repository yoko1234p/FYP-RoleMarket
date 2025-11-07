# Image Generation API Alternatives

**Project:** FYP-RoleMarket
**Context:** Google Gemini 2.5 Flash Image API has regional restrictions (unavailable in HK/CN)
**Date Created:** 2025-11-07
**Last Updated:** 2025-11-07

---

## 📋 Executive Summary

### Problem
Google Gemini 2.5 Flash Image API 在香港/中國大陸地區不可用，需要 VPN 才能使用。為確保系統可用性，我們需要整合替代 API。

### Recommended Solution ⭐
**Hugging Face Inference API (FLUX.1-dev)**
- ✅ 有免費 tier（每月 credits）
- ✅ 冇地區限制（香港可用）
- ✅ 整合簡單（Python SDK）
- ✅ 成本低（$0.0012/圖）
- ✅ 速度快（~10s/圖）

---

## 🔍 Detailed API Comparison

### 1. Hugging Face Inference API (FLUX.1-dev) ⭐ **推薦**

**Status:** 🚀 Ready to implement

**Pros:**
- ✅ **Free Tier Available**: 每月提供 credits，適合測試和小規模使用
- ✅ **No Regional Restrictions**: 香港/中國大陸可直接使用
- ✅ **Cost-Effective**: $0.0012/圖（超過免費額度後）
- ✅ **Fast Generation**: ~10 seconds per image
- ✅ **High Quality**: FLUX.1-dev 是 2024-2025 頂級開源模型
- ✅ **Easy Integration**: Official Python SDK (huggingface_hub)
- ✅ **Same Infrastructure**: 已經用緊 HF Spaces 部署模型

**Cons:**
- ⚠️ Free tier 有 rate limits (~幾百 requests/hour)
- ⚠️ 超過免費額度後需要付費（但成本很低）

**Pricing:**
- **Free Tier**: 每月 credits（適合測試）
- **PRO Users** ($9/month): 20× 免費 credits
- **Pay-as-you-go**: $0.00012 per second of GPU compute time
  - FLUX.1-dev (10s generation): **$0.0012/image**
  - Stable Diffusion XL (5s generation): **$0.0006/image**

**Implementation Example:**

```python
"""
Hugging Face FLUX.1-dev Image Generation

Requirements:
    pip install huggingface_hub pillow
"""

import os
from huggingface_hub import InferenceClient
from PIL import Image

# Initialize client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Generate image
def generate_image_hf(prompt: str, model: str = "black-forest-labs/FLUX.1-dev"):
    """
    Generate image using Hugging Face Inference API.

    Args:
        prompt: Text prompt for image generation
        model: HF model ID (default: FLUX.1-dev)

    Returns:
        PIL Image object
    """
    try:
        image = client.text_to_image(
            prompt=prompt,
            model=model
        )
        return image
    except Exception as e:
        print(f"Image generation failed: {str(e)}")
        return None

# Example usage
prompt = """
A cute pink pig character (Lulu Pig) with big eyes and round body,
wearing a red Chinese New Year costume with traditional patterns,
surrounded by red lanterns and festive decorations,
--style cute --ar 1:1
"""

image = generate_image_hf(prompt)
if image:
    image.save("lulu_spring_festival_hf.png")
    print("✅ Image generated successfully!")
```

**Integration Steps:**
1. Sign up for Hugging Face account (free)
2. Get HF Token: https://huggingface.co/settings/tokens
3. Add to `.env`: `HF_TOKEN=hf_xxxxx`
4. Install SDK: `pip install huggingface_hub`
5. Replace Gemini API calls with HF client

**Expected Timeline:** 1-2 days

---

### 2. Hugging Face Inference API (Stable Diffusion XL)

**Status:** 📋 Backup option

**Same infrastructure as FLUX.1-dev, but:**
- ✅ Faster generation: ~5s/image
- ✅ Cheaper: $0.0006/image
- ⚠️ Lower quality than FLUX
- ✅ Good for bulk generation or cost optimization

**Model ID:** `stabilityai/stable-diffusion-xl-base-1.0`

---

### 3. Midjourney via TTAPI

**Status:** ✅ API Key already available

**Pros:**
- ✅ **API Key Already Configured**: `TTAPI_API_KEY` in `.env`
- ✅ **Highest Quality**: Industry-leading image generation
- ✅ **Proven Track Record**: Already used in Obj 2 development
- ✅ **Existing Code**: `obj2_midjourney_api/ttapi_client.py`

**Cons:**
- ❌ No free tier
- ❌ Higher cost: ~$0.02/image
- ❌ Slower: ~15s/image
- ⚠️ Requires Midjourney subscription + TTAPI credits

**Pricing:**
- **Cost per image**: ~$0.02
- **Generation time**: 15-20 seconds
- **TTAPI Docs**: https://ttapi.io/docs/apiReference/midjourney

**When to Use:**
- For high-quality marketing materials
- When image quality is critical
- For final production designs

**Code Location:**
- Existing implementation: `obj2_midjourney_api/ttapi_client.py`
- Already integrated with reference images (cref)

---

### 4. Replicate (FLUX/Stable Diffusion)

**Status:** 📋 Planned for mid-term

**Pros:**
- ✅ Stable commercial service
- ✅ Wide model selection (FLUX, SDXL, etc.)
- ✅ Good documentation
- ✅ Pay-per-use pricing

**Cons:**
- ❌ No free tier
- ⚠️ Requires credit card for signup

**Pricing:**
- **FLUX Pro**: ~$0.003/image
- **FLUX Dev**: ~$0.002/image
- **Generation time**: 3-5 seconds

**Implementation Example:**

```python
import replicate

# Generate image
output = replicate.run(
    "black-forest-labs/flux-schnell",
    input={
        "prompt": "A cute pink pig in Chinese New Year costume",
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "output_quality": 80
    }
)

print(output)
```

**API Docs:** https://replicate.com/docs

---

### 5. RunPod (Self-hosted SDXL)

**Status:** 📋 Long-term consideration

**Pros:**
- ✅ Lowest cost: ~$0.0005/image
- ✅ Full control over model and configuration
- ✅ Scalable GPU infrastructure

**Cons:**
- ❌ Requires infrastructure management
- ❌ Higher setup complexity
- ❌ Need to maintain model serving code

**Pricing:**
- **GPU rental**: ~$0.3-0.5/hour (RTX 3090/4090)
- **Cost per image**: ~$0.0005 (if generating many images)
- **Setup time**: 1-2 days for initial deployment

**When to Consider:**
- High volume generation (>10,000 images/month)
- Need custom model fine-tuning
- Long-term cost optimization

---

## 📊 Cost Analysis (1000 Images)

| API | Free Tier Covers | Cost for 1000 images | Total Cost |
|-----|------------------|---------------------|------------|
| **Google Gemini** | ✅ ~1000 (60/min) | $0 | **$0** (if no regional restriction) |
| **HF FLUX.1-dev** ⭐ | ✅ ~500 | $0.60 | **$0.60** |
| **HF SDXL** | ✅ ~500 | $0.30 | **$0.30** |
| **Midjourney TTAPI** | ❌ 0 | $20.00 | **$20.00** |
| **Replicate FLUX** | ❌ 0 | $3.00 | **$3.00** |
| **RunPod SDXL** | ❌ 0 | $0.50 | **$0.50** |

**Recommendation for 1000 images/month:**
1. **Best Value**: Hugging Face FLUX.1-dev ($0.60) ⭐
2. **Cheapest**: Hugging Face SDXL ($0.30)
3. **Highest Quality**: Midjourney ($20)

---

## 🚀 Implementation Priority

### Phase 1: Immediate (Week 1)
✅ **Implement Hugging Face FLUX.1-dev**
- **Timeline**: 1-2 days
- **Effort**: Low (simple SDK integration)
- **Benefits**:
  - Solves regional restriction issue
  - Free tier for testing
  - Low cost for production
  - No infrastructure changes needed

**Implementation Checklist:**
- [ ] Sign up for Hugging Face account
- [ ] Get HF_TOKEN
- [ ] Add token to `.env` and Streamlit Secrets
- [ ] Install `huggingface_hub` package
- [ ] Create `HuggingFaceImageGenerator` class
- [ ] Replace Gemini API calls in `design_generator.py`
- [ ] Add CLIP similarity validation
- [ ] Test with 3 themes (Spring Festival, Halloween, Christmas)
- [ ] Update documentation

### Phase 2: Backup Integration (Week 2)
📋 **Keep Midjourney TTAPI as fallback**
- Already have API key
- Already have code (`obj2_midjourney_api/ttapi_client.py`)
- Use for high-quality marketing materials

### Phase 3: Cost Optimization (Month 2+)
📋 **Evaluate Replicate or RunPod**
- If usage > 5000 images/month
- Compare actual costs
- Consider RunPod for > 10,000 images/month

---

## 🔧 Integration Guide: Hugging Face FLUX.1-dev

### Step 1: Setup (10 minutes)

1. **Get HF Token:**
   - Visit: https://huggingface.co/settings/tokens
   - Create new token with "Read" permission
   - Copy token

2. **Add to Environment:**
   ```bash
   # .env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

3. **Install Dependencies:**
   ```bash
   pip install huggingface_hub
   ```

### Step 2: Create Wrapper Class (1 hour)

Create new file: `obj4_web_app/utils/huggingface_image_generator.py`

```python
"""
Hugging Face Image Generator

Wrapper for HF Inference API (FLUX.1-dev).
"""

import os
from typing import List, Optional
from huggingface_hub import InferenceClient
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


class HuggingFaceImageGenerator:
    """
    Image generation using Hugging Face Inference API.
    """

    def __init__(
        self,
        model: str = "black-forest-labs/FLUX.1-dev",
        token: Optional[str] = None
    ):
        """
        Initialize HF image generator.

        Args:
            model: HF model ID
            token: HF API token (default: from HF_TOKEN env var)
        """
        self.model = model
        self.token = token or os.getenv("HF_TOKEN")

        if not self.token:
            raise ValueError("HF_TOKEN not found in environment")

        self.client = InferenceClient(token=self.token)
        logger.info(f"HuggingFaceImageGenerator initialized with model: {model}")

    def generate_images(
        self,
        prompt: str,
        num_images: int = 1,
        reference_image_path: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Generate images using HF Inference API.

        Args:
            prompt: Text prompt
            num_images: Number of images to generate
            reference_image_path: Path to reference image (for character consistency)

        Returns:
            List of PIL Image objects
        """
        images = []

        for i in range(num_images):
            try:
                logger.info(f"Generating image {i+1}/{num_images}...")

                # Generate image
                image = self.client.text_to_image(
                    prompt=prompt,
                    model=self.model
                )

                images.append(image)
                logger.info(f"✅ Image {i+1} generated successfully")

            except Exception as e:
                logger.error(f"❌ Image {i+1} generation failed: {str(e)}")
                # Continue to next image instead of failing completely
                continue

        if not images:
            raise RuntimeError(f"All {num_images} image generations failed")

        return images

    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3
    ) -> Optional[Image.Image]:
        """
        Generate single image with retry logic.

        Args:
            prompt: Text prompt
            max_retries: Maximum retry attempts

        Returns:
            PIL Image or None
        """
        import time

        for attempt in range(max_retries):
            try:
                image = self.client.text_to_image(
                    prompt=prompt,
                    model=self.model
                )
                return image

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    raise

        return None
```

### Step 3: Update Design Generator (30 minutes)

Modify `obj4_web_app/utils/design_generator.py`:

```python
from obj4_web_app.utils.huggingface_image_generator import HuggingFaceImageGenerator

class DesignGenerator:
    def __init__(self):
        # Try HF first, fallback to Gemini
        self.hf_available = os.getenv("HF_TOKEN") is not None
        self.gemini_available = os.getenv("GOOGLE_API_KEY") is not None

        if self.hf_available:
            self.generator = HuggingFaceImageGenerator()
            logger.info("Using Hugging Face FLUX.1-dev")
        elif self.gemini_available:
            self.generator = GeminiImageGenerator()
            logger.info("Using Google Gemini (fallback)")
        else:
            raise ValueError("No image generation API available")
```

### Step 4: Test (30 minutes)

Test script:

```python
# test_hf_image_generation.py

from obj4_web_app.utils.huggingface_image_generator import HuggingFaceImageGenerator

generator = HuggingFaceImageGenerator()

prompt = """
A cute pink pig character with big eyes and round body,
wearing a red Chinese New Year costume,
surrounded by red lanterns,
high quality, detailed, cute style
"""

images = generator.generate_images(prompt, num_images=2)

for i, img in enumerate(images):
    img.save(f"test_hf_lulu_{i+1}.png")
    print(f"✅ Saved test_hf_lulu_{i+1}.png")
```

---

## 📝 Migration Checklist

### Pre-Migration
- [ ] Review current Gemini API usage
- [ ] Estimate monthly image generation volume
- [ ] Calculate expected costs with HF

### Migration Steps
- [ ] Setup HF account and get token
- [ ] Add HF_TOKEN to environment
- [ ] Implement HuggingFaceImageGenerator class
- [ ] Update DesignGenerator to use HF
- [ ] Test locally with 3 scenarios
- [ ] Add CLIP validation
- [ ] Test from HK IP (verify no regional restriction)
- [ ] Deploy to Streamlit Cloud
- [ ] Configure HF_TOKEN in Streamlit Secrets
- [ ] Test deployed version
- [ ] Monitor usage and costs

### Post-Migration
- [ ] Update documentation
- [ ] Update manual testing report
- [ ] Monitor HF API performance
- [ ] Track actual costs
- [ ] Keep Gemini code as fallback (if VPN available)

---

## 🔍 Monitoring & Optimization

### Metrics to Track
1. **Success Rate**: Target >= 95%
2. **Average Generation Time**: Target < 15s
3. **CLIP Similarity Score**: Target >= 0.75
4. **Monthly Cost**: Budget $50-100/month for production
5. **Rate Limit Hits**: Monitor free tier usage

### Cost Optimization Tips
1. Use SDXL for less critical images (~50% cheaper)
2. Batch generate images to maximize GPU utilization
3. Cache generated images to avoid regeneration
4. Upgrade to PRO ($9/month) if hitting free tier limits frequently
5. Consider RunPod if usage > 10,000 images/month

---

## 📚 References

### Official Documentation
- **Hugging Face Inference API**: https://huggingface.co/docs/api-inference
- **FLUX.1-dev Model**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **Pricing**: https://huggingface.co/docs/inference-providers/pricing
- **Python SDK**: https://huggingface.co/docs/huggingface_hub

### Alternative APIs
- **Replicate**: https://replicate.com/docs
- **Midjourney TTAPI**: https://ttapi.io/docs
- **RunPod**: https://www.runpod.io/docs

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** Ready for Implementation
