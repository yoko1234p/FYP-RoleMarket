# Epic 4: Streamlit Web Application Integration - Brownfield Enhancement

**Epic ID:** EPIC-004
**Status:** Draft
**Priority:** High
**Created:** 2025-11-06
**Owner:** Product Manager

---

## Epic Goal

建立一個統一的 Streamlit Web 應用程式，將 Objective 1（NLP Prompt 生成）、Objective 2（Google Gemini 圖片生成）和 Objective 3（Transformer 銷量預測）整合成一個完整的端到端商業解決方案，讓 ToyzeroPlus 設計團隊能夠透過友善的 Web 介面進行市場趨勢分析、角色設計生成和銷量預測。

---

## Epic Description

### Existing System Context

**當前相關功能：**
- **Obj 1 - NLP Prompt 生成：**
  - 核心模組：`obj1_nlp_prompt/enhanced_trends_pipeline.py`, `cultural_trend_adapter.py`
  - 功能：Google Trends 分析、Cultural Trend 轉化、LLM Prompt 生成

- **Obj 2 - Google Gemini 設計生成：**
  - 核心模組：`obj2_midjourney_api/google_gemini_client.py`, `character_focused_validator.py`
  - 功能：使用 Google Gemini 2.5 Flash Image 生成設計、CLIP 相似度驗證

- **Obj 3 - Transformer 預測模型：**
  - 核心模組：`obj3_lstm_forecast/hybrid_transformer_model.py`, `kaggle_train_lulu_exp11v2.py`
  - 功能：基於趨勢 + CLIP embeddings 的銷量預測（R² = 0.6788）

**技術棧：**
- Python 3.9+
- PyTorch 2.0+ (Transformer Model)
- Transformers 4.30+ (CLIP Model)
- Google Gemini API (圖片生成)
- Streamlit 1.28+ (Web Framework)
- Plotly 5.17+ (視覺化)

**整合點：**
- Web UI → Obj 1 NLP Pipeline (趨勢分析 + Prompt 生成)
- Web UI → Obj 2 Google Gemini Client (圖片生成 + CLIP 驗證)
- Web UI → Obj 3 Transformer Model (銷量預測)
- 所有模組透過 Python imports 和 API calls 整合

### Enhancement Details

**新增內容：**
1. **Streamlit Multi-Page Application**
   - Page 1: 趨勢分析與設計生成介面
   - Page 2: 銷量預測與市場洞察儀表板
   - 共享 sidebar 導航和配置

2. **核心整合層**
   - `obj4_web_app/utils/trends_api.py` - Obj 1 介面封裝
   - `obj4_web_app/utils/design_generator.py` - Obj 2 介面封裝
   - `obj4_web_app/utils/forecast_predictor.py` - Obj 3 介面封裝

3. **用戶體驗優化**
   - 即時進度顯示（Streamlit spinner/progress bar）
   - 錯誤處理和用戶友善提示
   - 結果快取（@st.cache_data, @st.cache_resource）

**整合方式：**
- 使用 Python module imports 整合現有 Obj 1-3 程式碼
- 透過 utility wrappers 統一 API 介面
- Streamlit session state 管理狀態和快取

**成功標準：**
- ✅ 用戶可在 5 分鐘內完成完整流程（趨勢輸入 → 設計生成 → 銷量預測）
- ✅ 所有 Obj 1-3 功能正常運作，無 regression
- ✅ Web UI 響應流暢，無阻塞性錯誤
- ✅ 生成結果可下載和保存

---

## Stories

### Story 4.1: Streamlit 基礎架構與 Obj 1 整合
**目標：** 建立 Streamlit app 基礎結構，整合 Obj 1 趨勢分析和 Prompt 生成功能

**任務：**
- 建立 `obj4_web_app/` 目錄結構（app.py, pages/, utils/, config.py）
- 實作 Page 1: 趨勢分析介面（Google Trends 關鍵字輸入、趨勢圖表顯示）
- 封裝 Obj 1 API (`utils/trends_api.py`)
- 實作 Prompt 生成功能並顯示結果

**驗收標準：**
- Streamlit app 可啟動並顯示 Page 1
- 用戶可輸入趨勢關鍵字並查看趨勢分析結果
- 生成的 Prompt 正確顯示在 UI 上
- Obj 1 原有功能正常運作（regression test）

**預估時間：** 4-6 小時

---

### Story 4.2: Obj 2 設計生成與 CLIP 驗證整合
**目標：** 整合 Google Gemini 圖片生成和 CLIP 相似度驗證功能至 Web UI

**任務：**
- 封裝 Obj 2 API (`utils/design_generator.py`)
- 實作圖片生成介面（4 張變化圖展示）
- 實作 CLIP 相似度顯示（參考圖 vs 生成圖）
- 實作圖片下載功能
- 優化圖片生成 loading 體驗（progress bar + 預計時間）

**驗收標準：**
- 用戶可基於 Prompt 生成 4 張設計圖
- CLIP 相似度分數正確顯示（目標 ≥ 0.80）
- 生成圖片可下載至本地
- Google Gemini API 錯誤處理完善（timeout, quota exceeded）
- Obj 2 原有功能正常運作（regression test）

**預估時間：** 5-7 小時

---

### Story 4.3: Obj 3 銷量預測與市場洞察儀表板
**目標：** 整合 Transformer 預測模型，建立市場洞察儀表板

**任務：**
- 封裝 Obj 3 API (`utils/forecast_predictor.py`)
- 實作 Page 2: 預測儀表板
  - 季節選擇器（Spring/Summer/Fall/Winter）
  - 設計選擇（從 Page 1 生成結果）
  - 銷量預測結果顯示（數字 + 信心區間）
- 實作歷史趨勢對比圖表（Plotly line chart）
- 實作市場洞察摘要（基於 Feature Importance）
- 實作模型載入快取（@st.cache_resource）

**驗收標準：**
- 用戶可選擇季節和設計，獲得銷量預測
- 預測結果顯示清晰（預測值 + MAE 誤差範圍）
- 歷史趨勢圖表正確顯示
- 市場洞察基於實際 Feature Importance 分析
- Transformer 模型載入時間 < 5 秒（透過快取）
- Obj 3 原有功能正常運作（regression test）

**預估時間：** 6-8 小時

---

## Compatibility Requirements

### 現有 API 兼容性
- [ ] Obj 1 `enhanced_trends_pipeline.py` 的 `generate_prompt()` API 保持不變
- [ ] Obj 2 `google_gemini_client.py` 的 `generate_image()` API 保持不變
- [ ] Obj 3 `hybrid_transformer_model.py` 的 `predict()` API 保持不變
- [ ] 所有現有 CLI 腳本仍可獨立運行（不依賴 Streamlit）

### 數據兼容性
- [ ] 不修改 Obj 3 訓練好的模型權重
- [ ] CLIP embeddings 提取方式保持一致
- [ ] 數據格式（CSV, JSON, NPY）與現有系統兼容

### UI/UX 兼容性
- [ ] 使用 Streamlit 預設主題（或輕量客製化）
- [ ] 響應式設計（支援 1280x720 以上解析度）
- [ ] 錯誤訊息使用繁體中文

### 性能兼容性
- [ ] 單次完整流程（趨勢分析 → 生成 → 預測）< 2 分鐘
- [ ] 模型載入使用 Streamlit cache，避免重複載入
- [ ] API 超時設定合理（Google Gemini: 60s, Transformer: 10s）

---

## Risk Mitigation

### Primary Risks

**Risk 1: Google Gemini API 不穩定或超時**
- **機率：** 中
- **影響：** 高（阻塞圖片生成功能）
- **緩解策略：**
  - 實作 retry 機制（最多 3 次）
  - 顯示清晰的錯誤訊息和重試按鈕
  - 超時設定為 60 秒
  - 在文檔中提供降級方案（手動使用 Google AI Studio）

**Risk 2: Streamlit session state 管理複雜度**
- **機率：** 中
- **影響：** 中（影響用戶體驗）
- **緩解策略：**
  - 明確定義 session state keys 和生命週期
  - 使用 `st.cache_data` 和 `st.cache_resource` 減少重複計算
  - 提供 "Reset" 按鈕清除 session state

**Risk 3: Obj 1-3 程式碼變更導致整合失敗**
- **機率：** 低
- **影響：** 高
- **緩解策略：**
  - 為 Obj 1-3 建立 wrapper 層，隔離直接依賴
  - 每個 Story 完成後執行 regression test
  - 使用 try-except 捕捉所有整合點錯誤

**Risk 4: Transformer 模型載入速度過慢**
- **機率：** 低
- **影響：** 中（影響首次使用體驗）
- **緩解策略：**
  - 使用 `@st.cache_resource` 快取模型
  - 顯示 loading spinner 和進度說明
  - 考慮模型量化（如需要）

### Rollback Plan

**如果整合失敗或阻塞：**
1. **Stage 1 Rollback (Story 4.1 失敗):**
   - 保留 Obj 1-3 獨立運行能力
   - 提供簡單的 CLI Demo 腳本展示整合

2. **Stage 2 Rollback (Story 4.2 失敗):**
   - 使用靜態圖片展示（預先生成的範例圖）
   - 文檔說明手動使用 Google Gemini 的流程

3. **Stage 3 Rollback (Story 4.3 失敗):**
   - 使用簡化版預測（不帶視覺化）
   - 提供 Jupyter Notebook 替代方案

**完全回退：**
- 所有 Obj 1-3 模組保持獨立可運行
- 提供詳細的 CLI 操作文檔
- 準備 PowerPoint 展示整合概念

---

## Definition of Done

### 功能完整性
- [ ] 所有 3 個 Stories 完成並通過驗收標準
- [ ] 端到端測試通過（3 個完整場景：春節、萬聖節、聖誕節）
- [ ] 所有 Obj 1-3 功能驗證無 regression

### 整合品質
- [ ] 整合點運作正確（Obj 1 → Obj 2 → Obj 3 流程順暢）
- [ ] 錯誤處理覆蓋所有整合點
- [ ] Session state 管理正確，無記憶體洩漏

### 文檔完整性
- [ ] README.md 更新（包含 Streamlit 啟動指令）
- [ ] 每個 Story 有詳細的 Dev Notes（在 story 文件中）
- [ ] API wrapper 函數有 docstrings

### 用戶體驗
- [ ] UI 響應流暢，無明顯卡頓
- [ ] 所有錯誤訊息清晰且友善（繁體中文）
- [ ] Loading 狀態顯示適當

### 測試覆蓋
- [ ] 單元測試（每個 util wrapper 函數）
- [ ] 整合測試（端到端流程）
- [ ] Regression 測試（Obj 1-3 獨立功能）

---

## Technical Dependencies

### 外部依賴
- Google Gemini API (需要 API key)
- OpenAI GPT API (GPT_API_free，用於 Prompt 生成)
- Google Trends (pytrends，無需 API key)

### Python 套件新增
```
# 已在 requirements.txt 中
streamlit>=1.28.0
plotly>=5.17.0
```

### 檔案結構
```
obj4_web_app/
├── app.py                          # Streamlit 主入口
├── pages/
│   ├── 1_🎨_設計生成.py            # Page 1: Obj 1 + Obj 2
│   └── 2_📊_銷量預測.py            # Page 2: Obj 3
├── utils/
│   ├── __init__.py
│   ├── trends_api.py               # Obj 1 Wrapper
│   ├── design_generator.py         # Obj 2 Wrapper
│   ├── forecast_predictor.py       # Obj 3 Wrapper
│   └── ui_helpers.py               # 共用 UI 函數
├── config.py                       # App 配置
└── README.md                       # Streamlit 使用說明
```

---

## Success Metrics

### 功能指標
- ✅ 完整流程成功率 > 95%（3/3 測試場景通過）
- ✅ CLIP 相似度維持 ≥ 0.80
- ✅ 預測誤差維持 MAE ≤ 330（Obj 3 原有水準）

### 性能指標
- ✅ Streamlit app 啟動時間 < 10 秒
- ✅ Prompt 生成時間 < 5 秒
- ✅ 圖片生成時間 < 15 秒/張（Google Gemini）
- ✅ 銷量預測時間 < 3 秒

### 用戶體驗指標
- ✅ 完整流程時間 < 2 分鐘
- ✅ 錯誤恢復時間 < 10 秒（retry 機制）
- ✅ UI 無阻塞性錯誤

---

## Timeline Estimate

- **Story 4.1:** 1 天（4-6 小時）
- **Story 4.2:** 1.5 天（5-7 小時）
- **Story 4.3:** 1.5 天（6-8 小時）
- **整合測試與優化：** 1 天
- **總計：** 5 天（包含緩衝）

---

## Notes

### 架構決策
- **為什麼選擇 Streamlit？**
  - 快速原型開發，適合 FYP Demo
  - 原生支援 Python ML 模型整合
  - 無需前後端分離，降低複雜度

- **為什麼使用 Wrapper 層？**
  - 隔離 Obj 1-3 直接依賴，降低耦合
  - 方便未來替換底層實作
  - 提供統一的錯誤處理介面

### 未來改進方向
- 使用者驗證和多用戶支援
- 結果數據庫持久化（SQLite/PostgreSQL）
- 批量生成功能（一次多個主題）
- 進階視覺化（Feature Importance 互動圖表）
- Docker 容器化部署

---

**Epic Status:** Draft - Ready for Story Development
**Next Step:** 開發詳細 User Stories（3 個 Stories）
