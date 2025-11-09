# Streamlit Cloud 部署教學文檔

**Project:** FYP-RoleMarket - AI Character IP Design & Demand Forecasting System
**Date:** 2025-11-07
**Version:** 1.0
**Author:** Developer (James) with PM Agent (John)

---

## 📋 目錄

1. [部署前準備](#部署前準備)
2. [Streamlit Cloud 部署步驟](#streamlit-cloud-部署步驟)
3. [Secrets 配置](#secrets-配置)
4. [驗證部署](#驗證部署)
5. [常見問題](#常見問題)
6. [故障排除](#故障排除)

---

## 📦 部署前準備

### 1. 確認檔案已提交

檢查所有必要檔案已 commit 到 GitHub：

```bash
# 檢查 git status
git status

# 確認這些檔案存在並已提交
git ls-files | grep -E "(app.py|config.py|requirements.txt|.streamlit/config.toml)"
```

**必須檔案清單：**
- ✅ `obj4_web_app/app.py` - 主應用程式
- ✅ `obj4_web_app/config.py` - 配置檔案（支援 Streamlit Secrets）
- ✅ `requirements.txt` - Python 依賴
- ✅ `.streamlit/config.toml` - Streamlit 配置
- ✅ `models/transformer_lulu/best_transformer_model.pth` - 預測模型

### 2. 準備 API Keys

你需要以下 API keys：

**必需（Blocking）：**
- ✅ `GPT_API_FREE_KEY` - 用於 Prompt 生成
  - 來源：https://github.com/chatanywhere/GPT_API_free
  - 當前值：已在 `.env` 檔案

**可選（Feature-Specific）：**
- `HF_TOKEN` - Hugging Face API（推薦用於圖片生成）
  - 來源：https://huggingface.co/settings/tokens
  - 建議申請：免費 + 冇地區限制

- `GOOGLE_API_KEY` - Google Gemini API（香港需 VPN）
  - 來源：https://aistudio.google.com/apikey
  - 備註：香港地區不支援

- `TTAPI_API_KEY` - Midjourney API（已有）
  - 來源：https://ttapi.io
  - 當前值：已在 `.env` 檔案

### 3. Push 到 GitHub

```bash
# 確認所有更改已提交
git add .
git commit -m "feat: 準備 Streamlit Cloud 部署"
git push origin main
```

---

## 🚀 Streamlit Cloud 部署步驟

### Step 1: 註冊/登入 Streamlit Cloud

1. 前往：https://share.streamlit.io
2. 點擊 **"Sign in"** 或 **"Sign up"**
3. 使用 **GitHub 帳號** 登入

![Streamlit Cloud Login](https://docs.streamlit.io/images/streamlit-cloud/get-started-button.png)

---

### Step 2: 新增應用程式

1. 點擊右上角 **"New app"** 按鈕

2. 填寫應用資訊：
   ```
   Repository: your-username/FYP-RoleMarket
   Branch: main
   Main file path: obj4_web_app/app.py
   App URL (optional): fyp-rolemarket-demo
   ```

3. 點擊 **"Advanced settings"** 展開進階設定：
   - **Python version**: `3.10`
   - **Requirements file**: `requirements.txt`（預設，無需更改）

![New App Settings](https://docs.streamlit.io/images/streamlit-cloud/deploy-an-app-1.png)

---

### Step 3: 配置 Secrets（重要！）

在部署前必須配置 API keys。

#### 3.1 進入 Secrets 設定

部署開始後：
1. 點擊右下角 **"Settings"** 按鈕
2. 選擇左側 **"Secrets"** 選項
3. 或直接前往：`https://share.streamlit.io/[your-app-url]/settings/secrets`

#### 3.2 複製 Secrets Template

打開專案中的 secrets template：
```bash
cat docs/streamlit-secrets-template.toml
```

#### 3.3 填入實際 API Keys

在 Streamlit Cloud Secrets 編輯器中貼上並**替換**以下內容：

```toml
# =====================================
# Required APIs (Deployment Blocking)
# =====================================

# GPT_API_free (Llama 3.1)
GPT_API_FREE_KEY = "your-gpt-api-key-here"
GPT_API_FREE_BASE_URL = "https://api.chatanywhere.org/v1"
GPT_API_FREE_MODEL = "gpt-3.5-turbo"

# =====================================
# Optional APIs (建議配置)
# =====================================

# Hugging Face API (推薦用於圖片生成)
HF_TOKEN = "your-hf-token-here"

# TTAPI Midjourney API (Backup)
TTAPI_API_KEY = "your-ttapi-key-here"

# Google Gemini API (Optional - 香港需 VPN)
# GOOGLE_API_KEY = "your-google-api-key-here"

# =====================================
# Configuration Settings
# =====================================

TRENDS_REGION = "HK"
TRENDS_LANGUAGE = "zh-TW"
CLIP_THRESHOLD_CORE = 0.75
CLIP_THRESHOLD_STYLE = 0.60
PROJECT_NAME = "FYP-RoleMarket"
DEBUG = false
```

**⚠️ 重要提示：**
- 必須填入真實的 `HF_TOKEN`（如想使用圖片生成）
- 可以暫時註釋掉 `GOOGLE_API_KEY`（香港不支援）
- 確保 TOML 格式正確（字串用引號）

#### 3.4 儲存 Secrets

點擊 **"Save"** 按鈕儲存配置。

**Streamlit 會自動重啟應用以套用新的 secrets。**

---

### Step 4: 部署應用

1. 點擊 **"Deploy!"** 按鈕開始部署

2. 等待部署完成（約 **5-10 分鐘**）
   - Streamlit 會自動：
     - Clone GitHub repository
     - 安裝 `requirements.txt` 依賴
     - 載入模型檔案
     - 啟動應用

3. 監控部署日誌：
   - 點擊右下角 **"Manage app"** → **"Logs"**
   - 檢查是否有錯誤訊息

![Deployment Logs](https://docs.streamlit.io/images/streamlit-cloud/app-menu.png)

---

### Step 5: 驗證部署成功

部署完成後，你會看到應用 URL：
```
https://your-username-fyp-rolemarket-demo.streamlit.app
```

**驗證清單：**
- [ ] 應用頁面成功載入
- [ ] 首頁顯示系統資訊
- [ ] 左側邊欄顯示導航選單
- [ ] "🎨 設計生成" 頁面可訪問
- [ ] "📊 銷量預測" 頁面可訪問

---

## 🔐 Secrets 配置

### Secrets 優先級

Streamlit Cloud 使用以下優先級讀取配置：

1. **Streamlit Secrets** (`st.secrets`) - 生產環境
2. **Environment Variables** (`.env`) - 本地開發

我們的 `config.py` 已實現自動偵測：

```python
def get_secret(key: str, default=None):
    # Try Streamlit secrets first (production)
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]

    # Fallback to environment variable (local dev)
    return os.getenv(key, default)
```

### 測試 Secrets 配置

部署後，在應用中測試：

1. 前往 "🎨 設計生成" 頁面
2. 嘗試生成 Prompt
3. 檢查是否有 API key 錯誤

**正常情況：**
- ✅ Prompt 生成成功
- ✅ 沒有 "API key not found" 錯誤

**異常情況：**
- ❌ "GPT_API_TOKEN not found"
  - 解決：檢查 Secrets 中的 `GPT_API_FREE_KEY`
- ❌ "HF_TOKEN not found"
  - 解決：添加 `HF_TOKEN` 到 Secrets（如需圖片生成）

---

## ✅ 驗證部署

### 功能測試清單

部署成功後，執行以下測試：

#### 1. 基礎功能測試

**Test 1: 首頁載入**
- [ ] 訪問應用 URL
- [ ] 首頁顯示 "歡迎使用 ToyzeroPlus AI 設計系統"
- [ ] 系統狀態顯示所有 Objectives 完成

**Test 2: 導航測試**
- [ ] 點擊 "🎨 設計生成" 進入頁面
- [ ] 點擊 "📊 銷量預測" 進入頁面
- [ ] 頁面切換正常，無錯誤

#### 2. Obj 1 - Prompt 生成測試

1. 進入 "🎨 設計生成" 頁面
2. 填入角色資訊：
   ```
   角色名稱: Lulu Pig
   角色描述: 可愛粉紅豬，大眼睛，圓滾滾身材
   ```
3. 切換到 "✍️ 手動輸入" 標籤
4. 輸入關鍵字：`春節, 紅色, 喜慶, 燈籠`
5. 點擊 "生成 Prompt"

**預期結果：**
- ✅ 顯示 "✅ Prompt 生成成功！"
- ✅ Prompt 包含角色名稱和關鍵字
- ✅ 可以下載 `.txt` 檔案

#### 3. Google Trends 測試（可選）

1. 在 "🔍 自動提取" 標籤
2. 選擇主題："🧧 春節"
3. 點擊 "提取關鍵字"

**可能結果：**
- ✅ 成功提取關鍵字並顯示
- ⚠️ "未找到相關趨勢數據"（Rate limiting）
  - 正常現象，使用手動輸入即可

#### 4. 圖片生成測試（需 HF_TOKEN）

**測試條件：** 必須配置 `HF_TOKEN` 在 Secrets

1. 生成 Prompt（步驟 2）
2. 選擇參考圖片
3. 點擊 "生成圖片"

**預期結果：**
- ✅ 圖片生成成功（需等待 10-15 秒）
- ✅ 顯示 CLIP 相似度分數
- ⚠️ 如未配置 HF_TOKEN：顯示 API key 錯誤

#### 5. 銷量預測測試

**前提：** 必須先生成圖片

1. 進入 "📊 銷量預測" 頁面
2. 選擇已生成的圖片
3. 選擇季節
4. 點擊 "預測銷量"

**預期結果：**
- ✅ 顯示預測銷量範圍
- ✅ 圖表正常顯示

---

## ❓ 常見問題

### Q1: 部署失敗，顯示 "ModuleNotFoundError"

**原因：** `requirements.txt` 遺漏依賴套件

**解決方法：**
1. 檢查本地 `requirements.txt` 是否包含所有依賴
2. 確認版本相容性（Python 3.10）
3. 重新部署

**檢查命令：**
```bash
pip freeze | grep -E "(streamlit|torch|transformers|pytrends)"
```

---

### Q2: "GPT_API_TOKEN not found" 錯誤

**原因：** Secrets 未正確配置

**解決方法：**
1. 前往 Streamlit Cloud → Settings → Secrets
2. 確認 `GPT_API_FREE_KEY` 已填入
3. 檢查 TOML 格式（字串必須用引號）
4. 儲存並等待應用重啟

**正確格式：**
```toml
GPT_API_FREE_KEY = "sk-xxxxx"  # ✅ 有引號
GPT_API_FREE_KEY = sk-xxxxx    # ❌ 無引號（錯誤）
```

---

### Q3: Google Trends 提取失敗

**原因：** pytrends 遇到 rate limiting（429 error）

**已實施解決方案：**
- ✅ Retry logic with exponential backoff（3 次重試）
- ✅ 2 秒延遲 between requests

**Workaround：**
- 使用 "✍️ 手動輸入" 標籤頁
- 等待 1-2 分鐘後重試

**Rate Limiting 資訊：**
- **Library:** `pytrends 4.9.2` (unofficial API)
- **Rate Limit:** ~1400 requests 後觸發
- **建議延遲:** 60 秒 between requests after limit
- **來源:** https://github.com/GeneralMills/pytrends

---

### Q4: 圖片生成失敗

**可能原因：**

1. **HF_TOKEN 未配置**
   - 解決：添加 `HF_TOKEN` 到 Secrets
   - 獲取：https://huggingface.co/settings/tokens

2. **GOOGLE_API_KEY 地區限制**（如使用 Gemini）
   - 解決：切換到 Hugging Face FLUX
   - 或：使用 VPN 連接非 HK 地區

3. **API Rate Limit**
   - 解決：等待幾分鐘後重試

---

### Q5: 應用載入緩慢

**原因：** Cold start - 首次載入需要載入模型

**預期載入時間：**
- 首次訪問：10-15 秒（載入 Transformer 模型）
- 後續訪問：2-3 秒

**優化方法：**
- 模型已使用 `@st.cache_resource` 快取
- 無需額外優化

---

### Q6: 模型檔案過大，部署失敗

**檢查模型大小：**
```bash
ls -lh models/transformer_lulu/best_transformer_model.pth
# 預期: ~1.5 MB
```

**Streamlit Cloud 限制：**
- 免費 tier: 1 GB RAM
- 模型大小: 無限制（但建議 < 100 MB）

**我們的模型：**
- ✅ 1.48 MB（遠低於限制）
- ✅ 無需優化

---

## 🛠️ 故障排除

### 檢查部署日誌

1. 前往 Streamlit Cloud Dashboard
2. 點擊應用名稱
3. 點擊右下角 "Manage app"
4. 選擇 "Logs" 查看詳細日誌

**常見錯誤日誌：**

```
# Error 1: Missing secrets
ValueError: GPT_API_TOKEN or GPT_API_FREE_KEY not found
→ 解決：配置 Secrets

# Error 2: Module not found
ModuleNotFoundError: No module named 'pytrends'
→ 解決：檢查 requirements.txt

# Error 3: Google Trends rate limit
ERROR: The request failed: Google returned a response with code 429
→ 正常情況，使用 retry logic 或手動輸入
```

---

### 重新部署

如需重新部署（例如更新程式碼）：

1. **本地更新並 push：**
   ```bash
   git add .
   git commit -m "fix: 修復錯誤"
   git push origin main
   ```

2. **Streamlit Cloud 自動重新部署：**
   - Streamlit 會偵測 GitHub 更新
   - 自動觸發重新部署
   - 無需手動操作

3. **手動重啟應用：**
   - Dashboard → Manage app → Reboot app

---

### 清除快取

如果應用行為異常：

1. 在應用右上角點擊 "⋮" 選單
2. 選擇 "Clear cache"
3. 選擇 "Rerun"

---

## 📊 監控與維護

### 應用健康檢查

定期檢查以下指標：

**每日檢查：**
- [ ] 應用是否正常運行
- [ ] 是否有 error logs
- [ ] API keys 是否過期

**每週檢查：**
- [ ] 依賴套件是否有更新
- [ ] 模型檔案是否完整
- [ ] 用戶反饋收集

### 效能監控

**Streamlit Cloud 提供：**
- CPU 使用率
- 記憶體使用率
- 請求次數統計

**訪問方式：**
Dashboard → App analytics

---

## 🎓 參考資源

### 官方文檔

- **Streamlit Cloud:** https://docs.streamlit.io/streamlit-community-cloud
- **Secrets Management:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- **Deploy an app:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app

### 專案文檔

- **Tech-Spec:** `docs/tech-specs/production-deployment-tech-spec.md`
- **API Alternatives:** `docs/api-alternatives.md`
- **Testing Report:** `docs/testing/manual-testing-report.md`
- **User Stories:** `docs/stories/DEPLOY-*.md`

### 社群支援

- **Streamlit Forum:** https://discuss.streamlit.io
- **GitHub Issues:** https://github.com/your-username/FYP-RoleMarket/issues

---

## ✅ 部署完成檢查清單

完成部署後，確認以下項目：

### 配置檢查
- [ ] GitHub repository 已連接
- [ ] 應用路徑正確：`obj4_web_app/app.py`
- [ ] Python version: 3.10
- [ ] Secrets 已配置（至少 `GPT_API_FREE_KEY`）
- [ ] `.streamlit/config.toml` 已載入

### 功能檢查
- [ ] 首頁正常載入
- [ ] Prompt 生成功能正常
- [ ] 手動關鍵字輸入正常
- [ ] 參考圖片選擇器正常
- [ ] 模型載入成功（無錯誤）

### 文檔更新
- [ ] README.md 更新部署 URL
- [ ] 測試報告更新部署結果
- [ ] 已記錄 known issues

### 下一步
- [ ] 分享應用 URL 給 stakeholders
- [ ] 收集用戶反饋
- [ ] 計劃下一版本改進

---

## 🚀 下一步：擴展功能

部署成功後，可考慮以下擴展：

### Phase 2: 圖片生成整合

**目標：** 整合 Hugging Face FLUX.1-dev

**步驟：**
1. 實施 `docs/api-alternatives.md` 中的 HF integration
2. 創建 `HuggingFaceImageGenerator` class
3. 更新 `design_generator.py`
4. 測試端到端流程

**預計時間：** 1-2 天

---

### Phase 3: HF Spaces 模型部署

**目標：** 獨立部署銷量預測模型

**步驟：**
1. 上傳模型到 Hugging Face Hub
2. 創建 HF Space demo
3. 整合到主應用

**參考：** `docs/stories/DEPLOY-004-hf-spaces-model-deployment.md`

---

### Phase 4: 效能優化

**優化項目：**
- [ ] 快取 Google Trends 結果
- [ ] 優化 CLIP 模型載入
- [ ] 壓縮生成的圖片
- [ ] 添加 loading indicators

---

**文檔版本：** 1.0
**最後更新：** 2025-11-07
**狀態：** 準備就緒 ✅

**下一步：立即部署！** 🚀
