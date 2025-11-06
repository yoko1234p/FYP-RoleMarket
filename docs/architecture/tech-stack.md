# Technology Stack

**Project:** AI-Driven Market-Informed Character IP Design & Demand Forecasting
**Version:** 1.0
**Last Updated:** 2025-11-06
**Status:** Production (Obj 1-3 Complete, Obj 4 Pending)

---

## Executive Summary

本系統為商業級 AI Pipeline，使用 Python 生態系統整合市場趨勢分析、AI 圖片生成和需求預測。技術選型優先考慮快速部署、成本效益和企業級穩定性。

**核心特點：**
- 100% Python (3.9+)
- 免費/低成本 API（Google Gemini Flash Image, GPT_API_free）
- 無需自建 GPU 基礎設施
- Kaggle/Colab 訓練環境
- Streamlit 快速原型開發

---

## Core Technology Stack

### Runtime Environment

| Category | Technology | Version | Purpose | Notes |
|----------|-----------|---------|---------|-------|
| **Language** | Python | 3.9+ | 主要開發語言 | 所有模組使用 Python |
| **Package Manager** | pip | Latest | 依賴管理 | requirements.txt |
| **Virtual Env** | venv / conda | - | 環境隔離 | 推薦 venv |
| **Container** | Docker | Latest (Optional) | 部署容器化 | docker-compose.yml 已配置 |

### AI/ML Frameworks

| Category | Technology | Version | Purpose | Critical Notes |
|----------|-----------|---------|---------|----------------|
| **Deep Learning** | PyTorch | 2.0+ | Transformer 模型訓練 | ⚠️ 核心依賴 - 所有 Obj 3 模型基於此 |
| **Transformers** | Transformers (HF) | 4.30+ | CLIP Model 載入 | 用於 `openai/clip-vit-large-patch14` |
| **ML Utils** | Scikit-learn | 1.3+ | 數據預處理、TF-IDF | Obj 1 關鍵字提取 |
| **Data Processing** | Pandas | 2.0+ | 數據操作 | 趨勢分析、數據集生成 |
| **Numerical** | NumPy | 1.24+ | 數值計算 | CLIP embeddings, 矩陣運算 |

### External APIs & Services

| Service | Provider | Purpose | Cost Model | Critical Info |
|---------|----------|---------|------------|---------------|
| **Image Generation** | Google Gemini 2.5 Flash | AI 設計圖生成 | **免費** (subject to quota) | ⚠️ 取代原先嘅 TTAPI Midjourney |
| **LLM Prompt Gen** | GPT_API_free (Llama 3.1) | Prompt 生成 | **免費** | 社區維護，有 rate limit |
| **Trends Data** | Google Trends (pytrends) | 市場趨勢提取 | **免費** | 無官方 API，使用 pytrends 庫 |
| **CLIP Validation** | OpenAI CLIP (HF) | 圖片相似度驗證 | **免費** (local inference) | ViT-Large/14, 1.7GB model |

### Web Application

| Category | Technology | Version | Purpose | Notes |
|----------|-----------|---------|---------|-------|
| **Framework** | Streamlit | 1.28+ | Web UI 框架 | ⚠️ Obj 4 核心技術 |
| **Visualization** | Plotly | 5.17+ | 互動圖表 | 銷量預測視覺化 |
| **Charts** | Matplotlib | 3.7+ | 靜態圖表 | 訓練曲線、實驗報告 |
| **Image Handling** | Pillow (PIL) | 10.0+ | 圖片處理 | 圖片下載、格式轉換 |

### Development & Testing

| Category | Technology | Version | Purpose | Notes |
|----------|-----------|---------|---------|-------|
| **Testing** | pytest | 7.4+ | 單元測試 | Optional - 測試覆蓋率待提升 |
| **Coverage** | pytest-cov | 4.1+ | 測試覆蓋率 | Optional |
| **Linting** | (未配置) | - | 程式碼風格 | ⚠️ 待補充 - 建議 Ruff/Black |
| **Type Checking** | (未配置) | - | 型別檢查 | ⚠️ 待補充 - 建議 mypy |
| **Notebooks** | Jupyter | 7.0+ | 探索性分析 | Optional - Docker 環境支援 |

### Training Infrastructure

| Category | Technology | Purpose | Cost | Notes |
|----------|-----------|---------|------|-------|
| **Training Env** | Kaggle Notebooks | Transformer 模型訓練 | **免費** | GPU P100, 9hrs/week |
| **Alternative** | Google Colab | 備案訓練環境 | **免費** (Pro: $9.99/mo) | GPU T4 |
| **Model Storage** | Hugging Face Hub | 模型權重託管 | **免費** | Optional - 目前使用 local storage |
| **Dataset Storage** | Local + Kaggle Datasets | 訓練數據儲存 | **免費** | Lulu Pig 數據集（1,075 records） |

---

## Technology Selection Rationale

### Why Python?

**選擇理由：**
1. ✅ AI/ML 生態系統最完善（PyTorch, Transformers, Scikit-learn）
2. ✅ Streamlit 快速原型開發（Obj 4）
3. ✅ 團隊技能匹配
4. ✅ 豐富的 API 客戶端庫

**限制：**
- ⚠️ 性能不及 compiled languages（但對當前規模足夠）
- ⚠️ 部署需要 Python runtime（Docker 解決）

### Why Google Gemini Flash Image (vs Midjourney)?

**選擇理由：**
1. ✅ **免費** - 無需支付 TTAPI quota（原預算 $10-30）
2. ✅ 快速生成（11.18s/圖）
3. ✅ 官方 Google API（穩定性高）
4. ✅ Reference Image 支援（角色一致性）

**Trade-offs：**
- ⚠️ 圖片質量略低於 Midjourney（但 CLIP ≥ 0.80 仍達標）
- ⚠️ 免費 quota 限制（需監控使用量）

**歷史決策：**
- 原計劃：TTAPI Midjourney API（PPU mode, $10-30 budget）
- 變更原因：Google Gemini 免費且效果可接受
- 變更日期：2025-10-27（v1.2 Enhancement）

### Why Transformer (vs LSTM)?

**選擇理由：**
1. ✅ 更強的長距離依賴捕捉能力
2. ✅ 並行計算（訓練更快）
3. ✅ 實驗結果優於 LSTM（R² 0.6788 vs 0.5127 baseline）

**實際配置（Exp #11v2）：**
- D_MODEL = 64
- NUM_LAYERS = 2
- NHEAD = 8
- Input: Time-series (4-quarter history) + Static (CLIP 768-dim + product type 4-dim)

### Why Streamlit (vs Flask/FastAPI)?

**選擇理由（Obj 4）：**
1. ✅ 快速原型開發（小時級完成 MVP）
2. ✅ 內建 UI 組件（不需寫 HTML/CSS/JS）
3. ✅ Python-native（無需學習前端技術）
4. ✅ 適合 FYP Demo（重視功能展示 > 生產級 UI）

**Trade-offs：**
- ⚠️ 客製化彈性較低
- ⚠️ 不適合高併發生產環境
- ⚠️ Session state 管理需謹慎

---

## Dependency Management

### Requirements.txt Structure

```plaintext
# Core AI/ML Libraries
torch>=2.0.0
transformers>=4.30.0

# NLP & Trend Analysis
pytrends>=4.9.0
jieba>=0.42.1
scikit-learn>=1.3.0

# LLM Integration
openai>=1.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0

# Web Application
streamlit>=1.28.0
plotly>=5.17.0
matplotlib>=3.7.0

# Utilities
python-dotenv>=1.0.0
requests>=2.31.0
tqdm>=4.66.0

# Testing (Optional)
pytest>=7.4.0
pytest-cov>=4.1.0

# Development (Optional)
jupyter>=1.0.0
notebook>=7.0.0
```

### Critical Dependencies

**⚠️ 必須版本要求：**
1. **PyTorch >= 2.0**
   - Reason: Transformer model 使用 2.0+ 的 API
   - Impact: 降級會導致 Obj 3 模型無法載入

2. **Transformers >= 4.30**
   - Reason: CLIP model loading
   - Impact: 舊版可能無法正確載入 `openai/clip-vit-large-patch14`

3. **Streamlit >= 1.28**
   - Reason: Obj 4 使用的 session_state API
   - Impact: 舊版 session state 行為可能不同

**🟡 建議版本（有彈性）：**
- Pandas, NumPy, Scikit-learn - 可使用較新版本
- Plotly, Matplotlib - 向下兼容性佳

---

## Environment Configuration

### Required Environment Variables

```bash
# API Keys
GOOGLE_API_KEY=<Google AI Studio API Key>  # For Gemini Image Generation
GPT_API_TOKEN=<GPT_API_free Token>         # For LLM Prompt Generation

# Optional
HUGGINGFACE_TOKEN=<HF Token>               # If uploading models to HF Hub
KAGGLE_USERNAME=<Kaggle Username>           # For Kaggle dataset access
KAGGLE_KEY=<Kaggle API Key>
```

### .env File Example

```bash
# Copy from .env.example
GOOGLE_API_KEY=AIzaSy...
GPT_API_TOKEN=sk-...

# Development Settings
DEBUG=True
LOG_LEVEL=INFO
```

---

## Known Issues & Technical Debt

### Critical Issues

1. **Google Gemini API Rate Limiting**
   - Issue: 免費 tier 有 rate limit（未明確公開）
   - Workaround: 實作 retry 機制 + 延遲
   - Status: Obj 4 需處理

2. **CLIP Model Size (1.7GB)**
   - Issue: 首次載入需 5-10 秒
   - Workaround: Streamlit `@st.cache_resource`
   - Status: Obj 4 需實作

3. **Transformer Model 權重儲存**
   - Issue: 目前使用 local storage（`models/transformer_lulu/`）
   - Risk: 版本控制困難、協作不便
   - TODO: 考慮上傳至 Hugging Face Hub

### Minor Issues

1. **Linting/Formatting 未配置**
   - Impact: 程式碼風格不一致
   - Recommendation: 使用 Ruff 或 Black

2. **Type Hints 不完整**
   - Impact: IDE 支援受限
   - Recommendation: 逐步添加 type hints

3. **測試覆蓋率低**
   - Current: < 20%（估計）
   - Target: > 60% for Obj 4

---

## Performance Characteristics

### Obj 1 - Trend Analysis & Prompt Generation
- Google Trends 查詢: ~2-3 秒
- TF-IDF 關鍵字提取: < 1 秒
- LLM Prompt 生成: ~3-5 秒
- **Total: ~5-8 秒**

### Obj 2 - Image Generation
- Google Gemini 生成: ~11.18 秒/張
- CLIP 相似度計算: ~0.5 秒/張
- 4 張變化: **~45-50 秒**

### Obj 3 - Sales Forecasting
- Transformer model 載入: ~3-5 秒（首次）
- 單次預測: < 1 秒
- **Total: ~3-5 秒（首次），< 1 秒（後續）**

### Obj 4 - Web Application (預估)
- Streamlit app 啟動: ~5-10 秒
- Page 切換: < 1 秒
- **完整流程（Obj 1 → 2 → 3）: ~1-2 分鐘**

---

## Upgrade Path & Versioning

### Current Version: 1.0 (Phase A Complete)

**Completed:**
- Obj 1: Trend Analysis ✅
- Obj 2: Image Generation ✅
- Obj 3: Sales Forecasting ✅

**Pending:**
- Obj 4: Web Application ⏳

### Future Considerations (v2.0)

**Potential Upgrades:**
1. **API 升級：**
   - Google Gemini Pro → 更高質量圖片
   - GPT-4 → 更好的 Prompt 生成

2. **架構升級：**
   - FastAPI backend + React frontend（生產級）
   - PostgreSQL 數據持久化
   - Redis cache layer

3. **ML 模型升級：**
   - 實際銷售數據訓練（取代模擬數據）
   - Ensemble models（Transformer + XGBoost）

---

## References

**Official Documentation:**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pytrends GitHub](https://github.com/GeneralMills/pytrends)

**Internal References:**
- PRD: `docs/prd.md`
- Implementation Roadmap: `docs/implementation-roadmap.md`
- Experiment Log: `docs/experiment-log-lulu-transformer.md`

---

**Document Owner:** Architect (Winston)
**Maintained By:** Development Team
**Review Cycle:** After each major milestone
