# 🎓 FYP 演講部署完整指南

## 📋 快速選擇

| 方案 | 難度 | 費用 | 適合場景 | 部署時間 |
|------|------|------|----------|----------|
| **Hugging Face Spaces** | ⭐ 簡單 | 免費 | FYP 演講展示 | 10 分鐘 |
| **Streamlit Cloud** | ⭐ 簡單 | 免費 | 快速原型 | 5 分鐘 |
| **Docker + Railway** | ⭐⭐ 中等 | 免費/$5 | 生產環境 | 15 分鐘 |
| **本地運行** | ⭐ 簡單 | 免費 | 演講備用 | 2 分鐘 |

---

## 🚀 方案 1：Hugging Face Spaces（推薦）

### 為什麼選擇 HF Spaces？
- ✅ **完全免費**（無需信用卡）
- ✅ **可分享 URL**：`https://huggingface.co/spaces/你的用戶名/rolemarket-demo`
- ✅ **專業外觀**：適合 FYP 報告引用
- ✅ **已有完整配置**：你的 `hf-spaces-deploy/` 目錄已準備好

### 部署步驟（10 分鐘完成）

#### 1. 創建 Hugging Face 帳號
```bash
# 訪問 https://huggingface.co/join
# 註冊帳號（免費）
```

#### 2. 安裝 Hugging Face CLI
```bash
pip install huggingface_hub
huggingface-cli login
# 輸入你的 Access Token（從 https://huggingface.co/settings/tokens 獲取）
```

#### 3. 創建 Space
```bash
# 方法 A：使用 CLI
huggingface-cli repo create rolemarket-demo --type space --space_sdk streamlit

# 方法 B：網頁創建
# 1. 訪問 https://huggingface.co/new-space
# 2. Space name: rolemarket-demo
# 3. SDK: Streamlit
# 4. 點擊 "Create Space"
```

#### 4. Clone 並推送代碼
```bash
# Clone 你的 Space
git clone https://huggingface.co/spaces/你的用戶名/rolemarket-demo
cd rolemarket-demo

# 複製部署文件
cp -r ../FYP-RoleMarket/hf-spaces-deploy/* .

# 推送到 HF
git add .
git commit -m "feat: 初始化 RoleMarket Demo for FYP"
git push
```

#### 5. 等待部署（5-10 分鐘）
- HF 會自動構建並部署
- 訪問：`https://huggingface.co/spaces/你的用戶名/rolemarket-demo`

### 配置 Secrets（如需 API）
在 Space 設定頁面添加：
- `GEMINI_OPENAI_API_KEY`: Gemini API Key
- `GOOGLE_APPLICATION_CREDENTIALS`: Firebase 憑證（如果用 Firebase）

---

## 🌟 方案 2：Streamlit Cloud（最快）

### 部署步驟（5 分鐘）

#### 1. 推送代碼到 GitHub
```bash
# 如果尚未推送
git push origin main
```

#### 2. 訪問 Streamlit Cloud
- 前往 https://share.streamlit.io/
- 使用 GitHub 帳號登入
- 點擊 "New app"

#### 3. 配置部署
- Repository: `你的用戶名/FYP-RoleMarket`
- Branch: `main`
- Main file path: `obj4_web_app/app.py`
- 點擊 "Deploy!"

#### 4. 添加 Secrets
在 App settings > Secrets 添加：
```toml
GEMINI_OPENAI_API_KEY = "sk-xxx..."
```

#### 5. 訪問 App
- URL: `https://你的用戶名-fyp-rolemarket.streamlit.app`

---

## 🐳 方案 3：Docker + Railway（進階）

### 為什麼選擇這個方案？
- ✅ 完全控制環境
- ✅ 適合生產部署
- ✅ 你已有 `Dockerfile`

### 部署步驟（15 分鐘）

#### 1. 測試 Docker 本地運行
```bash
cd /Volumes/Work/greewich/FYP/FYP-RoleMarket

# 構建鏡像
docker build -t fyp-rolemarket .

# 運行容器
docker run -p 8501:8501 \
  -e GEMINI_OPENAI_API_KEY="sk-xxx..." \
  fyp-rolemarket

# 訪問 http://localhost:8501
```

#### 2. 部署到 Railway（免費額度 $5/月）
```bash
# 安裝 Railway CLI
npm install -g @railway/cli

# 登入
railway login

# 初始化項目
railway init

# 部署
railway up

# 添加 Secrets
railway variables set GEMINI_OPENAI_API_KEY="sk-xxx..."

# 獲取 URL
railway domain
```

---

## 💻 方案 4：本地運行（演講備用）

### 適用場景
- 網絡不穩定時的備用方案
- 演講現場展示

### 運行步驟（2 分鐘）

#### 1. 啟動應用
```bash
cd /Volumes/Work/greewich/FYP/FYP-RoleMarket

# 啟動 Streamlit
streamlit run obj4_web_app/app.py
```

#### 2. 訪問
- 自動打開瀏覽器：`http://localhost:8501`
- 或訪問網絡 URL（顯示在終端）

#### 3. 演講提示
準備演講稿：
> "這是本地運行的版本，也可通過 https://huggingface.co/spaces/xxx/rolemarket-demo 在線訪問"

---

## 📊 FYP 報告引用範例

### 系統架構圖
```
用戶輸入關鍵詞
    ↓
[Google Trends API] → 趨勢分析
    ↓
[LLM API] → Prompt 生成
    ↓
[Gemini 2.5 Flash] → 圖片生成
    ↓
[CLIP 模型] → 特徵提取
    ↓
[Transformer 模型] → 銷量預測
    ↓
[Streamlit Web UI] → 用戶展示
```

### 部署證明（寫入 FYP 報告）

```markdown
## 5.2 系統部署

本系統已完整部署至雲端平台，提供公開可訪問的 Web 介面：

- **Demo URL**: https://huggingface.co/spaces/你的用戶名/rolemarket-demo
- **源碼**: https://github.com/你的用戶名/FYP-RoleMarket
- **技術棧**:
  - Frontend: Streamlit 1.31
  - Backend: Python 3.14
  - ML Framework: PyTorch 2.1
  - 部署平台: Hugging Face Spaces
  - 容器化: Docker

部署架構圖：
[參考上方架構圖]

所有資源均為開源，任何人可復現實驗結果。
```

### 截圖建議
1. **首頁截圖**：展示系統介面
2. **輸入頁面**：展示關鍵詞輸入
3. **趨勢分析**：Google Trends 圖表
4. **生成結果**：AI 生成的圖片
5. **預測結果**：銷量預測圖表
6. **部署證明**：HF Spaces URL 截圖

---

## 🚨 演講前檢查清單

### 部署檢查
- [ ] 應用可正常訪問（測試 URL）
- [ ] 所有功能正常運作
  - [ ] Google Trends 查詢
  - [ ] Prompt 生成
  - [ ] 圖片生成（如有 API Key）
  - [ ] 銷量預測
- [ ] 界面美觀無錯誤
- [ ] 加載速度可接受（<10 秒）

### 備用方案
- [ ] 本地運行測試成功
- [ ] 準備離線數據（如網絡故障）
- [ ] 截圖準備（如 demo 失敗）
- [ ] 視頻錄製（展示完整流程）

### 演講準備
- [ ] 準備 3-5 個測試案例
  - 案例 1: 聖誕節 (Christmas, cute, red)
  - 案例 2: 新年 (Chinese New Year, lucky, gold)
  - 案例 3: 夏日 (Summer, beach, blue)
- [ ] URL 寫在簡報上（方便觀眾訪問）
- [ ] 準備演講稿：
  ```
  "我們的系統已部署在 Hugging Face Spaces，
   這是一個專業的 AI/ML 應用託管平台。
   現在讓我展示系統如何運作..."
  ```

---

## ⚡ 快速部署命令（複製貼上即可）

### 選擇 1：Hugging Face Spaces
```bash
# 一鍵部署
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create rolemarket-demo --type space --space_sdk streamlit
git clone https://huggingface.co/spaces/$(huggingface-cli whoami | grep 'username:' | awk '{print $2}')/rolemarket-demo
cd rolemarket-demo
cp -r ../FYP-RoleMarket/hf-spaces-deploy/* .
git add .
git commit -m "feat: 初始化 RoleMarket Demo"
git push
echo "✅ 部署完成！訪問：https://huggingface.co/spaces/$(huggingface-cli whoami | grep 'username:' | awk '{print $2}')/rolemarket-demo"
```

### 選擇 2：本地運行（最簡單）
```bash
cd /Volumes/Work/greewich/FYP/FYP-RoleMarket
streamlit run obj4_web_app/app.py
```

---

## 🎯 演講演示建議

### 時間分配（5-10 分鐘 Demo）
1. **開場（30 秒）**
   - "讓我展示系統的實際運作"
   - 打開部署 URL

2. **輸入展示（1 分鐘）**
   - 輸入關鍵詞："Christmas, cute, red"
   - 說明："系統會自動分析 Google Trends"

3. **趨勢分析（1 分鐘）**
   - 展示趨勢圖表
   - 說明："這是近期搜索熱度"

4. **Prompt 生成（1 分鐘）**
   - 展示生成的 prompt
   - 說明："LLM 自動生成最佳 prompt"

5. **圖片生成（2 分鐘）**
   - 展示生成的角色圖片
   - 說明："使用 Gemini 2.5 Flash 生成"

6. **銷量預測（2 分鐘）**
   - 展示預測結果
   - 說明："Transformer 模型預測未來銷量"

7. **總結（30 秒）**
   - "完整流程約 X 秒完成"
   - "系統可公開訪問"

### 演講話術範例
```
"各位評審，現在讓我展示系統的實際運作。

[打開 URL]
這是我們部署在 Hugging Face Spaces 的 Web 應用。

[輸入關鍵詞]
我輸入 'Christmas, cute, red' 作為市場趨勢關鍵詞。

[點擊分析]
系統正在調用 Google Trends API 分析搜索熱度...

[展示圖表]
可以看到 'Christmas' 在 12 月達到高峰，這符合市場規律。

[生成 Prompt]
基於趨勢數據，LLM 自動生成了最佳的圖片生成 prompt。

[生成圖片]
使用 Gemini 2.5 Flash API 生成角色設計...

[展示結果]
這是生成的角色圖片，符合我們輸入的主題。

[預測銷量]
最後，Transformer 模型基於 CLIP 特徵預測銷量為 XXX 件。

整個流程完全自動化，大約 X 秒完成。
系統已開源並可公開訪問，任何人都可以測試。"
```

---

## 📞 需要幫助？

### 部署問題
- HF Spaces Logs 查看錯誤
- 檢查 `requirements.txt` 版本兼容性
- 確保 API Keys 已配置

### 演講支援
- 準備離線版本
- 錄製演示視頻
- 準備多個測試案例

---

**祝你 FYP 演講順利！** 🎓✨
