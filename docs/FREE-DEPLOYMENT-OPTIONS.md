# 🆓 完全免費部署平台對比（FYP 演講適用）

## 📊 快速對比

| 平台 | 難度 | 部署時間 | 限制 | URL 格式 | 推薦度 |
|------|------|---------|------|---------|--------|
| **Hugging Face Spaces** | ⭐ | 10 分鐘 | 無信用卡 | `hf.co/spaces/你/app` | ⭐⭐⭐⭐⭐ 最推薦 |
| **Streamlit Cloud** | ⭐ | 5 分鐘 | 需 GitHub | `你-app.streamlit.app` | ⭐⭐⭐⭐⭐ 最快 |
| **Railway** | ⭐⭐ | 15 分鐘 | $5/月額度 | `你-app.railway.app` | ⭐⭐⭐⭐ |
| **Render** | ⭐⭐ | 20 分鐘 | 需信用卡驗證 | `你-app.onrender.com` | ⭐⭐⭐ |
| **PythonAnywhere** | ⭐⭐ | 25 分鐘 | 限制多 | `你.pythonanywhere.com` | ⭐⭐ |
| **Google Cloud Run** | ⭐⭐⭐ | 30 分鐘 | $300 試用額度 | Custom domain | ⭐⭐ 複雜 |

---

## 🥇 方案 1：Hugging Face Spaces（最推薦）

### 為什麼最推薦？
- ✅ **完全免費**（永久，無信用卡）
- ✅ **專業外觀**（AI/ML 社群認可）
- ✅ **你已有完整配置**（`hf-spaces-deploy/` 目錄）
- ✅ **簡單 URL**（適合寫在簡報上）
- ✅ **可分享**（任何人可訪問）

### 部署步驟（10 分鐘）

#### Step 1: 安裝 CLI
```bash
pip install huggingface_hub
```

#### Step 2: 登入
```bash
huggingface-cli login
# 輸入 Token（從 https://huggingface.co/settings/tokens 獲取）
```

#### Step 3: 創建 Space
```bash
huggingface-cli repo create rolemarket-fyp-demo --type space --space_sdk streamlit
```

#### Step 4: 推送代碼
```bash
git clone https://huggingface.co/spaces/你的用戶名/rolemarket-fyp-demo
cd rolemarket-fyp-demo

# 複製部署文件
cp -r ../FYP-RoleMarket/hf-spaces-deploy/* .

# 推送
git add .
git commit -m "feat: FYP 演講 Demo"
git push
```

#### Step 5: 訪問（5-10 分鐘後）
```
https://huggingface.co/spaces/你的用戶名/rolemarket-fyp-demo
```

### 添加 API Keys（重要）
在 Space 設定頁面 > Settings > Secrets 添加：
```
GEMINI_OPENAI_API_KEY = "你的 API Key"
```

---

## 🥈 方案 2：Streamlit Cloud（最快）

### 為什麼選這個？
- ✅ **5 分鐘部署**（最快方案）
- ✅ **完全免費**（無限制）
- ✅ **簡單 URL**（`你-app.streamlit.app`）
- ⚠️ **需要 GitHub**（代碼必須在 GitHub）

### 部署步驟（5 分鐘）

#### Step 1: 推送到 GitHub（如尚未推送）
```bash
# 如果還沒有 GitHub remote
git remote add origin https://github.com/你的用戶名/FYP-RoleMarket.git
git push -u origin main
```

#### Step 2: 訪問 Streamlit Cloud
1. 前往 https://share.streamlit.io/
2. 使用 GitHub 登入
3. 點擊 **"New app"**

#### Step 3: 配置
- **Repository**: `你的用戶名/FYP-RoleMarket`
- **Branch**: `main`
- **Main file path**: `obj4_web_app/app.py`
- 點擊 **"Deploy!"**

#### Step 4: 添加 Secrets
在 App settings > Secrets 添加：
```toml
GEMINI_OPENAI_API_KEY = "你的 API Key"
```

#### Step 5: 訪問（2 分鐘後）
```
https://你的用戶名-fyp-rolemarket.streamlit.app
```

---

## 🥉 方案 3：Railway（$5 免費額度）

### 特點
- ✅ **$5/月 免費額度**（夠用 1-2 個月）
- ✅ **支援 Docker**（你已有 Dockerfile）
- ✅ **自動 HTTPS**
- ⚠️ **需註冊**（但不需信用卡驗證）

### 部署步驟（15 分鐘）

#### Step 1: 安裝 Railway CLI
```bash
npm install -g @railway/cli
```

#### Step 2: 登入並初始化
```bash
railway login
railway init
```

#### Step 3: 部署
```bash
railway up
```

#### Step 4: 添加環境變數
```bash
railway variables set GEMINI_OPENAI_API_KEY="你的 Key"
```

#### Step 5: 獲取 URL
```bash
railway domain
# 輸出: https://你的app.railway.app
```

---

## 🏆 方案 4：Render（需信用卡驗證，但免費）

### 特點
- ✅ **完全免費**（Free Tier）
- ✅ **支援 Docker**
- ⚠️ **需信用卡驗證**（但不會扣款）
- ⚠️ **15 分鐘無活動會休眠**（冷啟動需 30-60 秒）

### 部署步驟（20 分鐘）

#### Step 1: 註冊 Render
訪問 https://render.com/

#### Step 2: 連接 GitHub
點擊 "New +" > "Web Service" > 連接 GitHub repo

#### Step 3: 配置
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run obj4_web_app/app.py --server.port=$PORT`
- **Plan**: Free

#### Step 4: 添加環境變數
在 Environment > Add Environment Variable:
```
GEMINI_OPENAI_API_KEY = 你的 Key
```

#### Step 5: 部署
點擊 "Create Web Service"，等待 10-15 分鐘

---

## 💡 FYP 演講最佳方案組合

### 推薦配置：**雙重保險**

#### 主方案：Streamlit Cloud（線上）
- 5 分鐘部署
- 演講時展示線上 URL
- 專業形象

#### 備用方案：本地運行
```bash
streamlit run obj4_web_app/app.py
```
- 網絡故障時備用
- 確保演講順利

---

## 🚀 一鍵部署腳本

你已有 `deploy_fyp.sh` 腳本：

```bash
./deploy_fyp.sh
# 選擇：
# 1 = Hugging Face Spaces（推薦）
# 2 = Streamlit Cloud（最快）
# 3 = 本地運行（備用）
# 4 = Docker（進階）
```

---

## 📞 緊急部署方案（演講前 1 小時）

### 如果所有線上方案都失敗：

#### 方案 A：本地運行 + 分享畫面
```bash
streamlit run obj4_web_app/app.py
# 在簡報中說："這是本地運行版本，可通過 [URL] 在線訪問"
```

#### 方案 B：錄製演示視頻
```bash
# 使用 QuickTime 或 OBS 錄製
# 展示完整操作流程（3-5 分鐘）
# 演講時播放視頻
```

#### 方案 C：使用截圖
準備關鍵步驟截圖：
1. 輸入頁面
2. Trends 圖表
3. 生成的圖片
4. 預測結果

---

## 🎯 我的建議（根據你的情況）

### 最佳方案：**Streamlit Cloud**
**原因**：
1. ✅ 最快（5 分鐘）
2. ✅ 最穩定（99.9% uptime）
3. ✅ URL 專業（`你-fyp-rolemarket.streamlit.app`）
4. ✅ 無需信用卡

**步驟**：
```bash
# 1. 確保代碼在 GitHub（應該已推送）
git push origin main

# 2. 訪問 https://share.streamlit.io/
# 3. GitHub 登入 > New app > 填寫：
#    - Repo: 你的用戶名/FYP-RoleMarket
#    - Branch: main
#    - Main file: obj4_web_app/app.py
# 4. Deploy!
```

**演講前 1 天** 部署完成，**演講當天** 同時啟動本地版本作為備用。

---

需要我幫你執行部署嗎？或者有其他問題？
