# Story 4.1: Streamlit 基礎架構與 Obj 1 整合

**Story ID:** STORY-4.1
**Epic:** EPIC-004 - Streamlit Web Application Integration
**Status:** ✅ Done
**Priority:** High
**Points:** 8
**Created:** 2025-11-06
**Completed:** 2025-11-06
**Assigned To:** Developer (James)

---

## User Story

**As a** ToyzeroPlus 設計團隊成員，
**I want** 透過 Web 介面輸入趨勢關鍵字並生成設計 Prompt，
**So that** 我可以快速獲得基於市場趨勢的角色設計靈感，無需操作複雜的命令列工具。

---

## Story Context

### Existing System Integration

**整合對象：** Objective 1 - NLP Prompt Generation Pipeline

**核心模組：**
- `obj1_nlp_prompt/enhanced_trends_pipeline.py` - 完整趨勢分析流程
- `obj1_nlp_prompt/cultural_trend_adapter.py` - 文化趨勢轉化
- `obj1_nlp_prompt/prompt_generator.py` - LLM Prompt 生成

**技術棧：**
- Python 3.9+
- Streamlit 1.28+ (Web Framework)
- PyTrends 4.9+ (Google Trends API)
- OpenAI 1.0+ (GPT_API_free for LLM)

**整合模式：**
- 使用 Python module imports 直接調用 Obj 1 函數
- 透過 `utils/trends_api.py` wrapper 統一介面
- Streamlit session state 管理用戶輸入和結果

**現有 Touch Points：**
- `enhanced_trends_pipeline.generate_prompt(character_info, trend_keywords)` - 主要整合點
- Google Trends API 調用（透過 pytrends）
- GPT_API_free LLM 調用（需要 API token）

---

## Acceptance Criteria

### Functional Requirements

**FR1: Streamlit 應用基礎結構**
- [ ] 建立 `obj4_web_app/` 目錄結構
  - `app.py` - Streamlit 主入口（landing page）
  - `pages/1_🎨_設計生成.py` - Page 1
  - `utils/__init__.py` - Utility 模組初始化
  - `config.py` - 應用配置（API keys, 常數）
- [ ] `app.py` 顯示歡迎訊息和導航指引
- [ ] Streamlit sidebar 包含頁面導航和設定選項

**FR2: 趨勢關鍵字輸入介面**
- [ ] Page 1 包含文字輸入框（接受逗號分隔的關鍵字）
- [ ] 預設範例關鍵字（如 "春節, 可愛, 紅色"）
- [ ] 角色資訊輸入區（角色名稱、描述）
- [ ] "分析趨勢" 按鈕觸發分析

**FR3: Google Trends 分析顯示**
- [ ] 點擊 "分析趨勢" 後顯示 loading spinner
- [ ] 顯示 Google Trends 趨勢圖表（使用 Streamlit line_chart 或 Plotly）
- [ ] 顯示提取的 Top 10 關鍵字
- [ ] 錯誤處理：API 失敗時顯示友善錯誤訊息

**FR4: Prompt 生成與顯示**
- [ ] "生成 Prompt" 按鈕觸發 LLM 調用
- [ ] 顯示生成的完整 Prompt（使用 `st.code()` 或 `st.text_area()`）
- [ ] Prompt 可複製到剪貼簿
- [ ] 錯誤處理：LLM API 失敗時顯示錯誤並提供重試選項

### Integration Requirements

**IR1: Obj 1 API 封裝**
- [ ] 建立 `utils/trends_api.py` wrapper
- [ ] 實作 `extract_google_trends(keywords: List[str])` 函數
- [ ] 實作 `generate_prompt(character_info: dict, trend_keywords: List[str])` 函數
- [ ] 所有函數包含 docstrings 和型別提示

**IR2: 現有功能保留**
- [ ] Obj 1 CLI 腳本仍可獨立運行
- [ ] `enhanced_trends_pipeline.py` 的 API 不被修改
- [ ] 現有測試腳本（如 `test_enhanced_pipeline.py`）仍能執行

**IR3: 配置管理**
- [ ] `config.py` 從環境變數讀取 API keys
- [ ] 支援 `.env` 檔案（使用 `python-dotenv`）
- [ ] 敏感資訊不寫死在程式碼中

### Quality Requirements

**QR1: 錯誤處理**
- [ ] Google Trends API 錯誤有 try-except 捕捉
- [ ] GPT_API_free 錯誤有 retry 機制（最多 3 次）
- [ ] 所有錯誤訊息使用繁體中文且友善

**QR2: 性能優化**
- [ ] 使用 `@st.cache_data` 快取 Google Trends 查詢結果（TTL=1小時）
- [ ] 避免重複 API 調用（檢查 session state）
- [ ] Streamlit app 啟動時間 < 5 秒

**QR3: 用戶體驗**
- [ ] 所有 loading 狀態有明確提示
- [ ] UI 文字使用繁體中文
- [ ] 輸入驗證（如：不允許空白關鍵字）

**QR4: 測試覆蓋**
- [ ] 為 `utils/trends_api.py` 編寫單元測試
- [ ] 測試錯誤處理路徑（mock API 失敗）
- [ ] 執行 Obj 1 regression test（確保原有功能正常）

---

## Technical Notes

### Integration Approach

**Wrapper 設計模式：**
```python
# utils/trends_api.py
from obj1_nlp_prompt.enhanced_trends_pipeline import EnhancedTrendsPipeline

class TrendsAPIWrapper:
    def __init__(self):
        self.pipeline = EnhancedTrendsPipeline()

    def extract_google_trends(self, keywords: List[str]) -> pd.DataFrame:
        """提取 Google Trends 數據並返回 DataFrame"""
        try:
            return self.pipeline.fetch_trends(keywords)
        except Exception as e:
            raise TrendsAPIError(f"Google Trends 提取失敗: {str(e)}")

    def generate_prompt(self, character_info: dict, trend_keywords: List[str]) -> str:
        """生成設計 Prompt"""
        try:
            return self.pipeline.generate_prompt(character_info, trend_keywords)
        except Exception as e:
            raise PromptGenerationError(f"Prompt 生成失敗: {str(e)}")
```

### Existing Pattern Reference

**Streamlit Session State 管理：**
- 使用 `st.session_state` 儲存用戶輸入和 API 結果
- Key naming convention: `trends_keywords`, `generated_prompt`, `trends_data`

**Streamlit Cache 使用：**
```python
@st.cache_data(ttl=3600)  # 1小時 TTL
def fetch_trends_cached(keywords):
    return trends_api.extract_google_trends(keywords)
```

### Key Constraints

- **API Quota 限制：**
  - Google Trends: 無官方限制，但建議間隔 1 秒
  - GPT_API_free: 免費 tier 有 rate limit（具體數字待確認）

- **相依性管理：**
  - 不修改 Obj 1 原有程式碼
  - 透過 wrapper 隔離直接依賴

- **錯誤處理優先級：**
  - 優先保證 Streamlit app 不 crash
  - 所有外部 API 調用都需 try-except

---

## Tasks

### Task 1: 建立 Streamlit 基礎結構 (2 hrs)
- [ ] 建立 `obj4_web_app/` 目錄結構
- [ ] 實作 `app.py`（歡迎頁面 + sidebar）
- [ ] 實作 `config.py`（環境變數讀取）
- [ ] 測試 Streamlit app 可正常啟動

### Task 2: 實作 Obj 1 API Wrapper (2 hrs)
- [ ] 建立 `utils/trends_api.py`
- [ ] 實作 `TrendsAPIWrapper` 類別
- [ ] 實作 `extract_google_trends()` 函數
- [ ] 實作 `generate_prompt()` 函數
- [ ] 編寫單元測試（`tests/test_trends_api.py`）

### Task 3: 實作 Page 1 趨勢分析介面 (2 hrs)
- [ ] 建立 `pages/1_🎨_設計生成.py`
- [ ] 實作關鍵字輸入表單
- [ ] 整合 Google Trends 查詢
- [ ] 顯示趨勢圖表（Plotly line chart）
- [ ] 顯示 Top 10 關鍵字列表

### Task 4: 實作 Prompt 生成功能 (1.5 hrs)
- [ ] 實作 "生成 Prompt" 按鈕邏輯
- [ ] 整合 LLM API 調用（透過 wrapper）
- [ ] 顯示生成結果（`st.code()` 區塊）
- [ ] 實作複製到剪貼簿功能（`st.button()` + clipboard API）

### Task 5: 錯誤處理與優化 (1.5 hrs)
- [ ] 實作所有 try-except 錯誤捕捉
- [ ] 實作 retry 機制（LLM API）
- [ ] 添加 `@st.cache_data` 快取
- [ ] 優化 loading 狀態顯示

### Task 6: 測試與文檔 (1 hr)
- [ ] 執行端到端測試（手動測試完整流程）
- [ ] 執行 Obj 1 regression test
- [ ] 更新 `obj4_web_app/README.md`
- [ ] 更新主 `README.md`（新增 Streamlit 啟動指令）

---

## Definition of Done

### Functionality
- [ ] Streamlit app 可正常啟動（`streamlit run obj4_web_app/app.py`）
- [ ] Page 1 所有功能正常運作（趨勢輸入 → 分析 → Prompt 生成）
- [ ] 測試 3 組不同關鍵字，均能成功生成 Prompt

### Integration
- [ ] Obj 1 API wrapper 測試通過
- [ ] Obj 1 原有 CLI 腳本仍可運行（regression test）
- [ ] 錯誤處理覆蓋所有整合點

### Quality
- [ ] 單元測試通過（`pytest tests/test_trends_api.py`）
- [ ] 程式碼符合 PEP 8 風格
- [ ] 所有函數有 docstrings

### Documentation
- [ ] `obj4_web_app/README.md` 包含啟動指令和使用說明
- [ ] `utils/trends_api.py` 函數有完整註解
- [ ] 主 `README.md` 更新 Objective 4 狀態

---

## Testing Scenarios

### Scenario 1: 春節主題設計
**輸入：**
- 關鍵字: "春節, 紅色, 喜慶"
- 角色名稱: "Lulu Pig"
- 角色描述: "可愛粉紅豬，大眼睛"

**預期結果：**
- Google Trends 圖表顯示過去 3 個月趨勢
- Prompt 包含春節元素（如：紅包、燈籠）
- Prompt 保持 Lulu Pig 角色特徵

### Scenario 2: 萬聖節主題設計
**輸入：**
- 關鍵字: "萬聖節, 南瓜, 搞怪"
- 角色名稱: "Lulu Pig"
- 角色描述: "可愛粉紅豬，大眼睛"

**預期結果：**
- Google Trends 圖表顯示萬聖節相關趨勢
- Prompt 包含萬聖節元素（如：南瓜、糖果）
- Prompt 保持角色一致性

### Scenario 3: 錯誤處理測試
**操作：**
1. 輸入無效關鍵字（空白或特殊符號）
2. 模擬 Google Trends API 失敗（網路中斷）
3. 模擬 LLM API 失敗（quota exceeded）

**預期結果：**
- 顯示清晰的錯誤訊息（繁體中文）
- 提供重試選項
- Streamlit app 不 crash

---

## Dev Notes

### 開發環境設定
```bash
# 安裝依賴
pip install streamlit plotly python-dotenv

# 設定 API keys
cp .env.example .env
# 編輯 .env，填入 GPT_API_free token
```

### 啟動指令
```bash
streamlit run obj4_web_app/app.py
```

### 測試指令
```bash
# 單元測試
pytest tests/test_trends_api.py -v

# Obj 1 Regression Test
python obj1_nlp_prompt/test_enhanced_pipeline.py
```

---

## Agent Model Used
*將由 Developer Agent 填寫*

---

## Dev Agent Record

### Debug Log References
*將由 Developer Agent 記錄*

### Completion Notes
*將由 Developer Agent 填寫*

### File List
*將由 Developer Agent 維護*

### Change Log
*將由 Developer Agent 記錄*

---

**Story Status:** Draft
**Next Action:** 等待 Developer 開始實作
