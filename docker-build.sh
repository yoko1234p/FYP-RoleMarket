#!/bin/bash
# Docker build and test script

set -e

echo "🐳 Building Docker image..."
docker-compose build

echo "✅ Docker image built successfully"
echo ""
echo "📝 Quick Start:"
echo "  1. Setup API keys: cp .env.template .env && nano .env"
echo "  2. Start services: docker-compose up -d"
echo "  3. Access Streamlit: http://localhost:8501"
echo "  4. (Optional) Start Jupyter: docker-compose --profile dev up -d"
echo "  5. Access Jupyter: http://localhost:8888"
echo ""
echo "🔍 View logs: docker-compose logs -f fyp-rolemarket"
echo "🛑 Stop services: docker-compose down"
