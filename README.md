# AI-Driven Market-Informed Character IP Design & Demand Forecasting

**FYP Project - ToyzeroPlus Commercial AI Pipeline**

## 專案概述

商業級 AI 系統，整合市場趨勢分析、Midjourney 設計生成、LSTM 需求預測，專為 character IP 設計公司提供即刻可部署嘅生產工具。

## 核心功能

- **Objective 1:** Google Trends 趨勢分析 + LLM Prompt 生成
- **Objective 2:** TTAPI Midjourney API 商業級設計生成（--cref character consistency）
- **Objective 3:** Hybrid LSTM 銷量預測（結合 Trends + CLIP embeddings）
- **Objective 4:** Streamlit 統一 Web 介面

## 專案結構

```
FYP-RoleMarket/
├── obj1_nlp_prompt/       # Trend Intelligence & Prompt Generation
├── obj2_midjourney_api/   # Midjourney API Design Generation
├── obj3_lstm_forecast/    # LSTM Demand Forecasting
├── obj4_web_app/          # Streamlit Web Application
├── data/                  # Data storage (cache, images, trends)
├── tests/                 # Integration & unit tests
├── docs/                  # PRD, reports, documentation
└── config/                # API keys & configuration
```

## 快速開始

1. **安裝依賴：**
   ```bash
   pip install -r requirements.txt
   ```

2. **設置 API Keys：**
   - TTAPI Midjourney API key
   - GPT_API_free access token

3. **執行系統：**
   ```bash
   streamlit run obj4_web_app/app.py
   ```

## 技術棧

- **AI Models:** CLIP ViT-Large/14, Hybrid LSTM, GPT (Llama 3.1)
- **APIs:** TTAPI Midjourney, Google Trends, GPT_API_free
- **Framework:** PyTorch 2.0+, Streamlit, Transformers
- **成本:** ~$10-30 (TTAPI quota for 28 images)

## 文檔

詳細技術規格請參考：`docs/prd.md`

---

**Version:** 1.0
**Author:** Product Manager
**Date:** 2025-01-26
