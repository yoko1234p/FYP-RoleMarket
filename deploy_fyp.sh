#!/bin/bash
# FYP 演講快速部署腳本
# 支援 3 種部署方案：HF Spaces, Streamlit Cloud, Local

set -e  # Exit on error

echo "======================================"
echo "🎓 FYP RoleMarket 演講部署工具"
echo "======================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 選擇部署方案
echo "請選擇部署方案："
echo "1) 🚀 Hugging Face Spaces（推薦，免費，10 分鐘）"
echo "2) ⚡ Streamlit Cloud（最快，5 分鐘）"
echo "3) 💻 本地運行（演講備用，2 分鐘）"
echo "4) 🐳 Docker 部署（進階，15 分鐘）"
echo ""
read -p "輸入選項 (1-4): " choice

case $choice in
  1)
    echo ""
    echo "${GREEN}=== 方案 1: Hugging Face Spaces ===${NC}"
    echo ""

    # 檢查 huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        echo "${YELLOW}安裝 Hugging Face CLI...${NC}"
        pip install huggingface_hub
    fi

    # 登入
    echo "${YELLOW}請先登入 Hugging Face（需要 Access Token）${NC}"
    echo "獲取 Token: https://huggingface.co/settings/tokens"
    huggingface-cli login

    # 獲取用戶名
    USERNAME=$(huggingface-cli whoami | grep 'username:' | awk '{print $2}')
    echo ""
    echo "${GREEN}✅ 登入成功！用戶名: $USERNAME${NC}"

    # 創建 Space
    echo ""
    echo "${YELLOW}創建 Hugging Face Space...${NC}"
    huggingface-cli repo create rolemarket-demo --type space --space_sdk streamlit || echo "Space 可能已存在"

    # Clone Space
    echo ""
    echo "${YELLOW}Clone Space Repository...${NC}"
    cd ..
    git clone https://huggingface.co/spaces/$USERNAME/rolemarket-demo || cd rolemarket-demo
    cd rolemarket-demo

    # 複製文件
    echo ""
    echo "${YELLOW}複製部署文件...${NC}"
    cp -r ../FYP-RoleMarket/hf-spaces-deploy/* .

    # Commit & Push
    echo ""
    echo "${YELLOW}推送到 HF Spaces...${NC}"
    git add .
    git commit -m "feat: FYP 演講 Demo 部署" || echo "無新更改"
    git push

    echo ""
    echo "${GREEN}✅ 部署完成！${NC}"
    echo ""
    echo "🌐 訪問 URL:"
    echo "   https://huggingface.co/spaces/$USERNAME/rolemarket-demo"
    echo ""
    echo "⏰ 等待 5-10 分鐘讓 HF 構建應用"
    echo "📊 查看構建進度: https://huggingface.co/spaces/$USERNAME/rolemarket-demo/logs"
    ;;

  2)
    echo ""
    echo "${GREEN}=== 方案 2: Streamlit Cloud ===${NC}"
    echo ""

    # 檢查 git remote
    echo "${YELLOW}檢查 GitHub remote...${NC}"
    if ! git remote | grep -q origin; then
        echo "${RED}❌ 未找到 GitHub remote！${NC}"
        echo "請先推送代碼至 GitHub："
        echo "  git remote add origin https://github.com/你的用戶名/FYP-RoleMarket.git"
        echo "  git push -u origin main"
        exit 1
    fi

    echo "${GREEN}✅ GitHub remote 已配置${NC}"
    echo ""
    echo "📋 手動步驟（5 分鐘）："
    echo ""
    echo "1. 訪問: https://share.streamlit.io/"
    echo "2. 點擊 'New app'"
    echo "3. 配置："
    echo "   - Repository: $(git remote get-url origin | sed 's/\.git$//')"
    echo "   - Branch: main"
    echo "   - Main file path: obj4_web_app/app.py"
    echo "4. 點擊 'Deploy!'"
    echo ""
    echo "✅ 部署完成後，URL 格式："
    echo "   https://你的用戶名-fyp-rolemarket.streamlit.app"
    ;;

  3)
    echo ""
    echo "${GREEN}=== 方案 3: 本地運行 ===${NC}"
    echo ""

    # 檢查 venv
    if [ ! -d ".venv" ]; then
        echo "${YELLOW}創建虛擬環境...${NC}"
        python3 -m venv .venv
    fi

    # 啟動 venv
    echo "${YELLOW}啟動虛擬環境...${NC}"
    source .venv/bin/activate

    # 安裝依賴
    echo "${YELLOW}檢查依賴...${NC}"
    pip install -q streamlit

    # 啟動 Streamlit
    echo ""
    echo "${GREEN}✅ 啟動 Streamlit 應用...${NC}"
    echo ""
    echo "🌐 應用將在瀏覽器自動打開"
    echo "📍 URL: http://localhost:8501"
    echo ""
    echo "⚠️  演講提示："
    echo "   - 確保筆記本電腦網絡穩定"
    echo "   - 準備離線數據（如 API 失效）"
    echo "   - 錄製演示視頻作為備用"
    echo ""
    streamlit run obj4_web_app/app.py
    ;;

  4)
    echo ""
    echo "${GREEN}=== 方案 4: Docker 部署 ===${NC}"
    echo ""

    # 檢查 Docker
    if ! command -v docker &> /dev/null; then
        echo "${RED}❌ Docker 未安裝！${NC}"
        echo "請先安裝 Docker: https://www.docker.com/products/docker-desktop"
        exit 1
    fi

    echo "${YELLOW}構建 Docker 鏡像...${NC}"
    docker build -t fyp-rolemarket .

    echo ""
    echo "${GREEN}✅ 鏡像構建完成${NC}"
    echo ""
    echo "🚀 啟動容器："
    echo ""
    echo "docker run -p 8501:8501 \\"
    echo "  -e GEMINI_OPENAI_API_KEY='your-key-here' \\"
    echo "  fyp-rolemarket"
    echo ""
    echo "🌐 訪問: http://localhost:8501"
    ;;

  *)
    echo "${RED}無效選項！${NC}"
    exit 1
    ;;
esac

echo ""
echo "${GREEN}======================================"
echo "🎉 部署流程完成！"
echo "======================================${NC}"
echo ""
echo "📚 詳細指南: docs/FYP-DEPLOYMENT-GUIDE.md"
echo "📄 FYP Report: docs/final-year-project-report.md (10,298 字)"
echo ""
echo "祝你 FYP 演講順利！🎓✨"
