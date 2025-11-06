# Streamlit Cloud 部署指南

**專案：** FYP-RoleMarket - AI 角色設計與需求預測系統
**最後更新：** 2025-11-06
**目標平台：** Streamlit Community Cloud (免費版)

---

## 📋 前置準備

### 1. 檢查 Requirements

#### 必須檔案
- [x] `requirements.txt` - Python 依賴清單
- [x] `.streamlit/config.toml` - Streamlit 配置
- [ ] `README.md` - 項目說明（已有）
- [ ] `.gitignore` - Git 忽略檔案（已有）

#### 環境變數需求
- `GPT_API_TOKEN` 或 `GPT_API_FREE_KEY` - LLM API（必須）
- `GOOGLE_API_KEY` - Google Gemini（可選，影響圖片生成）

---

## 🔧 步驟 1: 準備 Requirements.txt

### 當前依賴檢查

運行以下命令生成 requirements.txt：

```bash
pip freeze > requirements_freeze.txt
```

### 精簡版 Requirements.txt

**建議結構：**

```txt
# Core Dependencies
streamlit==1.29.0
pandas==2.3.3
numpy==1.26.0
plotly==5.18.0

# Obj 1: NLP & Trends
pytrends==4.9.2
openai==1.3.0
python-dotenv==1.0.0

# Obj 2: Image Generation & CLIP
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
Pillow==10.1.0
requests==2.31.0

# Obj 3: Forecasting
scikit-learn==1.3.2

# Utilities
tqdm==4.66.1
```

### 潛在問題

1. **PyTorch 過大**
   - Streamlit 免費版有 1GB 空間限制
   - PyTorch (~2GB) 可能超出限制
   - **解決方案：** 使用 CPU-only 版本

```txt
# 改用 CPU-only PyTorch（較小）
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.1.0+cpu
torchvision==0.16.0+cpu
```

2. **CLIP Model 下載**
   - CLIP model (~1.7GB) 首次載入需下載
   - 可能導致冷啟動慢
   - **解決方案：** 使用 `@st.cache_resource` 已處理

---

## 🔒 步驟 2: 設置環境變數

### 在 Streamlit Cloud 設置

1. 登入 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 選擇你的 app
3. 點擊 **Settings** → **Secrets**
4. 加入以下內容：

```toml
# .streamlit/secrets.toml (本地測試用)
GPT_API_TOKEN = "sk-xxxxxxxxxxxxx"
GOOGLE_API_KEY = "AIzaSyxxxxxxxxxxxxx"
```

### 在程式碼中讀取

**已處理：** `obj4_web_app/config.py`

```python
# obj4_web_app/config.py
import os
import streamlit as st

# Priority: Streamlit secrets > .env > environment variables
if hasattr(st, 'secrets'):
    GPT_API_TOKEN = st.secrets.get("GPT_API_TOKEN") or os.getenv("GPT_API_TOKEN")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
else:
    GPT_API_TOKEN = os.getenv("GPT_API_TOKEN") or os.getenv("GPT_API_FREE_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

---

## 📁 步驟 3: 檢查檔案結構

### 必須包含的檔案

```
FYP-RoleMarket/
├── obj4_web_app/
│   ├── Home.py                          # Streamlit 入口
│   ├── pages/
│   │   ├── 1_🎨_設計生成.py
│   │   └── 2_📊_銷量預測.py
│   ├── utils/
│   │   ├── trends_api.py
│   │   ├── trends_extractor_wrapper.py
│   │   ├── design_generator.py
│   │   └── forecast_predictor.py
│   └── config.py
├── obj1_nlp_prompt/                     # 依賴模組
├── obj2_midjourney_api/
├── obj3_lstm_forecast/
├── data/
│   ├── reference_images/                # 必須包含
│   │   ├── lulu_pig_ref_1.png
│   │   └── lulu_pig_ref_2.jpg
│   └── character_descriptions/          # 必須包含
│       └── lulu_pig.txt
├── models/
│   └── transformer_lulu/
│       └── best_transformer_model.pth   # ⚠️ 模型檔案（~50MB）
├── requirements.txt                      # ✅ 必須
├── .streamlit/
│   └── config.toml                       # ✅ 必須
└── README.md
```

### ⚠️ 大檔案處理

**問題：** 模型檔案（`best_transformer_model.pth` ~50MB）超過 Git 限制

**解決方案 1: Git LFS**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/transformer_lulu/best_transformer_model.pth
git commit -m "Add model weights via Git LFS"
```

**解決方案 2: 外部託管（推薦）**
- 上傳至 Hugging Face Model Hub
- 程式碼中動態下載：

```python
# obj4_web_app/utils/forecast_predictor.py
from huggingface_hub import hf_hub_download

def _load_model(self):
    # Download from Hugging Face
    model_path = hf_hub_download(
        repo_id="your-username/fyp-rolemarket",
        filename="best_transformer_model.pth"
    )
    # Load model...
```

---

## 🚀 步驟 4: 部署至 Streamlit Cloud

### 4.1 連接 GitHub Repository

1. 確保程式碼已推送至 GitHub：
```bash
git add .
git commit -m "chore: 準備 Streamlit Cloud 部署"
git push origin main
```

2. 登入 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 點擊 **New app**
4. 選擇 Repository: `your-username/FYP-RoleMarket`
5. Branch: `main`
6. Main file path: `obj4_web_app/Home.py`

### 4.2 設置環境變數

在 **Advanced settings** → **Secrets** 加入：

```toml
GPT_API_TOKEN = "sk-xxxxxxxxxxxxx"
GOOGLE_API_KEY = "AIzaSyxxxxxxxxxxxxx"
```

### 4.3 部署

點擊 **Deploy!** 開始部署。

**預計時間：** 5-10 分鐘（首次部署較慢，需下載 PyTorch + CLIP）

---

## ✅ 步驟 5: 驗證部署

### 測試 Checklist

- [ ] **首頁顯示正常**
  - [ ] 標題和說明正確
  - [ ] 側邊欄導航正常

- [ ] **Page 1: 設計生成**
  - [ ] Obj 1 Prompt 生成正常
  - [ ] Google Trends 自動提取正常（或顯示錯誤訊息）
  - [ ] Obj 2 圖片生成正常（需 GOOGLE_API_KEY）

- [ ] **Page 2: 銷量預測**
  - [ ] 前置檢查正常（未生成圖片時顯示警告）
  - [ ] 預測功能正常
  - [ ] Plotly 圖表顯示正常

- [ ] **錯誤處理**
  - [ ] 缺少 API key 時顯示清晰錯誤訊息
  - [ ] 模型載入失敗時提示用戶

---

## 🐛 常見問題與解決

### 問題 1: ModuleNotFoundError

**錯誤：**
```
ModuleNotFoundError: No module named 'obj1_nlp_prompt'
```

**原因：** Streamlit Cloud 無法找到依賴模組

**解決：**
確保 `obj4_web_app/Home.py` 正確設置 `sys.path`：

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
```

---

### 問題 2: 模型檔案過大

**錯誤：**
```
Error: Repository exceeds 1GB limit
```

**解決：**
1. 使用 Git LFS
2. 或改用 Hugging Face 託管模型（推薦）

---

### 問題 3: CLIP 模型下載慢

**現象：** 首次載入需 5-10 分鐘

**原因：** CLIP model (~1.7GB) 需從 Hugging Face 下載

**解決：**
- **已處理：** 使用 `@st.cache_resource` 確保只下載一次
- **優化：** 加入 Loading spinner 提示用戶

---

### 問題 4: API Key 不生效

**錯誤：**
```
ValueError: GPT_API_TOKEN not found
```

**檢查清單：**
1. ✅ Secrets 是否正確設置？（Settings → Secrets）
2. ✅ Key 名稱是否匹配？（`GPT_API_TOKEN` vs `GPT_API_FREE_KEY`）
3. ✅ 程式碼是否正確讀取？（`st.secrets.get("GPT_API_TOKEN")`）

---

## 📊 性能優化

### 1. 冷啟動優化

**問題：** 首次訪問需載入 PyTorch + CLIP（5-10 分鐘）

**優化方案：**
- 使用 `@st.cache_resource` 緩存模型（已實作）
- 考慮使用 lighter 模型（如 DistilBERT）

### 2. Session State 管理

**已優化：**
- ✅ 使用 `st.session_state` 避免重複計算
- ✅ Lazy loading 模型（首次使用時才載入）

### 3. 資源限制

**Streamlit 免費版限制：**
- CPU: 1 core
- RAM: 1GB
- Storage: 1GB
- 無 GPU 支援

**建議：**
- 如需 GPU 加速，考慮升級至 Streamlit Teams ($200/月)
- 或自行部署至 AWS/GCP/Azure

---

## 📝 部署清單總結

### Pre-Deployment Checklist

- [ ] `requirements.txt` 已準備（CPU-only PyTorch）
- [ ] 模型檔案已上傳（Git LFS 或 Hugging Face）
- [ ] Reference images 已包含在 repo
- [ ] `.streamlit/config.toml` 已設置
- [ ] 環境變數已準備（GPT_API_TOKEN, GOOGLE_API_KEY）
- [ ] 所有程式碼已推送至 GitHub
- [ ] README.md 已更新部署說明

### Deployment Steps

1. [ ] 登入 Streamlit Community Cloud
2. [ ] 連接 GitHub Repository
3. [ ] 設置 Main file path: `obj4_web_app/Home.py`
4. [ ] 設置 Secrets（API keys）
5. [ ] 點擊 Deploy
6. [ ] 驗證所有功能正常

### Post-Deployment

- [ ] 測試所有 3 個 pages
- [ ] 檢查錯誤日誌（Settings → Logs）
- [ ] 更新 README 加入部署 URL
- [ ] 分享給用戶測試

---

## 🔗 相關資源

- [Streamlit Community Cloud 文檔](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Secrets 管理](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Git LFS 安裝](https://git-lfs.github.com/)
- [Hugging Face Model Hub](https://huggingface.co/models)

---

**準備日期：** 2025-11-06
**預計部署日期：** TBD（修復已知問題後）
