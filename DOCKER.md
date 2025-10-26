# FYP-RoleMarket Docker Setup Guide

## 快速開始

### 1. 設置環境變數

```bash
# 複製 .env 範本並填入你嘅 API keys
cp .env.template .env
nano .env  # 或用任何編輯器
```

確保設置：
- `TTAPI_API_KEY` - TTAPI Midjourney API key
- `GPT_API_FREE_KEY` - GPT_API_free access token

### 2. 啟動容器

**啟動 Streamlit Web App：**
```bash
docker-compose up -d
```

Web interface 會喺 http://localhost:8501 啟動

**（可選）啟動 Jupyter Notebook 用於開發：**
```bash
docker-compose --profile dev up -d
```

Jupyter 會喺 http://localhost:8888 啟動

### 3. 查看日誌

```bash
# Streamlit logs
docker-compose logs -f fyp-rolemarket

# Jupyter logs
docker-compose logs -f jupyter
```

### 4. 停止容器

```bash
docker-compose down
```

## Docker 指令參考

### 建構映像

```bash
# 重新建構映像（如果更新咗 requirements.txt）
docker-compose build --no-cache
```

### 進入容器執行指令

```bash
# 進入運行中嘅容器
docker exec -it fyp-rolemarket bash

# 執行 Python 腳本
docker exec -it fyp-rolemarket python obj1_nlp_prompt/script.py
```

### 資料持久化

所有重要資料都掛載到本機：
- `./data` → `/app/data` （生成嘅圖像、cache、trends）
- `./config` → `/app/config` （API 配置）
- 源碼目錄 → 對應 `/app` 目錄（支援熱重載）

## 架構說明

### 服務

1. **fyp-rolemarket（主服務）**
   - Port: 8501 (Streamlit)
   - 自動重啟: 除非手動停止
   - Health check: 每 30 秒檢查一次

2. **jupyter（開發服務，可選）**
   - Port: 8888 (Jupyter Notebook)
   - 只在 `--profile dev` 時啟動
   - 用於測試同原型開發

### 映像規格

- Base: `python:3.14-slim`
- Python 依賴: 根據 `requirements.txt`
- 系統依賴: build-essential, git, wget, curl

## 故障排除

### 容器啟動失敗

```bash
# 檢查日誌
docker-compose logs fyp-rolemarket

# 重建映像
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### API Key 錯誤

確保 `.env` 檔案包含正確嘅 API keys：
```bash
cat .env  # 檢查內容
```

### 端口被占用

如果 8501 或 8888 被占用，修改 `docker-compose.yml`：
```yaml
ports:
  - "9501:8501"  # 改用其他端口
```

## 性能優化

### GPU 支援（如果有 NVIDIA GPU）

修改 `docker-compose.yml` 添加：
```yaml
services:
  fyp-rolemarket:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

需要安裝 `nvidia-container-toolkit`。

### 減少映像大小

已優化：
- 使用 `python:3.14-slim`
- `.dockerignore` 排除唔必要檔案
- `--no-cache-dir` 喺 pip install

## 環境變數

完整列表（見 `.env.template`）：
- `TTAPI_API_KEY` - TTAPI Midjourney API key
- `GPT_API_FREE_KEY` - GPT API access key
- `TRENDS_REGION` - Google Trends 地區（預設 HK）
- `TRENDS_LANGUAGE` - 語言（預設 zh-TW）
- `CLIP_THRESHOLD_CORE` - CLIP 核心相似度閾值（預設 0.75）
- `CLIP_THRESHOLD_STYLE` - CLIP 風格相似度閾值（預設 0.60）

---

**Docker 版本要求：**
- Docker Engine: >= 20.10
- Docker Compose: >= 2.0
