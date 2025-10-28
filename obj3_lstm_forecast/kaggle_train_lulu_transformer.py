"""
Kaggle Notebook 版本 - Lulu 罐頭豬 Hybrid Transformer 訓練腳本

專為 Lulu 罐頭豬 Production 數據設計：
- 1075 筆記錄 (2017-2024, 8年)
- 9 種產品類型
- 36 個特徵欄位
- 自動檢測 GPU (CUDA/MPS)

使用方式:
1. 上傳 lulu_production_sales/ 資料夾到 Kaggle Dataset
2. 在 Kaggle Notebook 中運行此腳本
3. 啟用 GPU accelerator (T4)
4. 下載訓練好的模型權重

Author: Product Manager (John)
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import math
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Kaggle 環境配置
# ============================================================
KAGGLE_MODE = Path('/kaggle/input').exists()

if KAGGLE_MODE:
    INPUT_DIR = Path('/kaggle/input/lulu-rolemarket-sales-data')  # Kaggle Dataset 路徑
    OUTPUT_DIR = Path('/kaggle/working')
    logger.info("Running in KAGGLE environment")
else:
    INPUT_DIR = Path('data/lulu_production_sales')
    OUTPUT_DIR = Path('models/transformer_lulu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Running in LOCAL environment with Lulu Production data")

# 自動檢測設備
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    logger.info("Apple Silicon MPS available")
else:
    DEVICE = torch.device('cpu')
    logger.info("Using CPU")


# ============================================================
# Hyperparameters
# ============================================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping patience
SEQ_LENGTH = 4  # Google Trends 歷史長度
TS_INPUT_DIM = 1  # 每個時間步的特徵數（trend_score）
STATIC_INPUT_DIM = 772  # 4 (其他 trend 特徵) + 768 (CLIP embeddings)
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    """位置編碼（用於 Transformer）"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


# ============================================================
# Hybrid Transformer Model
# ============================================================
class HybridTransformer(nn.Module):
    """
    混合架構 Transformer:
    - 時序特徵 (Google Trends 歷史) → Transformer Encoder
    - 靜態特徵 (CLIP embeddings, trend 統計) → Fully Connected
    - 融合層 → 最終預測
    """
    def __init__(
        self,
        ts_input_dim=1,
        static_input_dim=772,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        seq_length=4
    ):
        super().__init__()

        # === 1. 時序分支 (Transformer Encoder) ===
        self.ts_embedding = nn.Linear(ts_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # CLS token (用於序列聚合)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # === 2. 靜態特徵分支 (Fully Connected) ===
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # === 3. 融合層 ===
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, ts_input, static_input):
        """
        Args:
            ts_input: (batch_size, seq_length, ts_input_dim)
            static_input: (batch_size, static_input_dim)

        Returns:
            output: (batch_size, 1) - 預測的銷量
        """
        # 1. 時序分支
        batch_size = ts_input.size(0)
        ts_emb = self.ts_embedding(ts_input)  # (batch_size, seq_length, d_model)
        ts_emb = ts_emb.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        ts_emb = self.pos_encoder(ts_emb)

        # 加入 CLS token
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # (1, batch_size, d_model)
        ts_emb = torch.cat([cls_tokens, ts_emb], dim=0)  # (seq_length + 1, batch_size, d_model)

        # Transformer Encoder
        ts_output = self.transformer_encoder(ts_emb)  # (seq_length + 1, batch_size, d_model)
        ts_features = ts_output[0]  # 取 CLS token 的輸出 (batch_size, d_model)

        # 2. 靜態特徵分支
        static_features = self.static_fc(static_input)  # (batch_size, 64)

        # 3. 融合
        combined = torch.cat([ts_features, static_features], dim=1)  # (batch_size, d_model + 64)
        output = self.fusion_fc(combined)  # (batch_size, 1)

        return output


# ============================================================
# Data Loading
# ============================================================
def load_and_preprocess_data():
    """讀取並預處理 Lulu Production 數據"""
    logger.info("Loading data...")

    # 1. 讀取 CSV
    csv_path = INPUT_DIR / "historical_data.csv"
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    logger.info(f"  Loaded {len(df)} records from CSV")

    # 2. 讀取 CLIP embeddings
    clip_path = INPUT_DIR / "clip_embeddings.npy"
    clip_embeddings = np.load(clip_path)
    logger.info(f"  Loaded CLIP embeddings: {clip_embeddings.shape}")

    # 3. 讀取 Trends history
    trends_path = INPUT_DIR / "trends_history.json"
    with open(trends_path, 'r', encoding='utf-8') as f:
        trends_history = json.load(f)
    logger.info(f"  Loaded trends data for {len(trends_history)} designs")

    # 4. 數據預處理
    logger.info("Preprocessing data...")

    # 構建時序特徵 (Google Trends 歷史)
    ts_data = []
    for design_id in df['design_id']:
        trends = trends_history[design_id]
        ts_data.append(trends)
    ts_data = np.array(ts_data, dtype=np.float32)  # (N, 4)
    ts_data = ts_data.reshape(-1, 4, 1)  # (N, seq_length, 1)
    logger.info(f"  Time-series shape: {ts_data.shape}")

    # 構建靜態特徵
    # trend_momentum, trend_volatility, trend_score_current, trend_score_q4
    trend_features = df[[
        'trend_momentum', 'trend_volatility', 'trend_score_current', 'trend_score_q4'
    ]].values.astype(np.float32)

    static_features = np.concatenate([trend_features, clip_embeddings], axis=1)  # (N, 4 + 768)
    logger.info(f"  Static features shape: {static_features.shape}")

    # 目標變數
    target = df['sales_quantity'].values.astype(np.float32)
    logger.info(f"  Target shape: {target.shape}")

    return ts_data, static_features, target


# ============================================================
# Training Function
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
    """訓練模型"""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    logger.info("Starting training...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for ts_batch, static_batch, target_batch in train_loader:
            ts_batch = ts_batch.to(DEVICE)
            static_batch = static_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(ts_batch, static_batch)
            loss = criterion(outputs.squeeze(), target_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ts_batch, static_batch, target_batch in val_loader:
                ts_batch = ts_batch.to(DEVICE)
                static_batch = static_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)

                outputs = model(ts_batch, static_batch)
                loss = criterion(outputs.squeeze(), target_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_transformer_model.pth")
            logger.info(f"  ✓ Model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    return train_losses, val_losses


# ============================================================
# Evaluation Function
# ============================================================
def evaluate_model(model, test_loader, scaler_target):
    """評估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ts_batch, static_batch, target_batch in test_loader:
            ts_batch = ts_batch.to(DEVICE)
            static_batch = static_batch.to(DEVICE)

            outputs = model(ts_batch, static_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(target_batch.numpy())

    # 反標準化
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets = scaler_target.inverse_transform(targets.reshape(-1, 1)).flatten()

    # 計算指標
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    logger.info("=" * 80)
    logger.info("Evaluation Results:")
    logger.info(f"  MAE:  {mae:.2f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info("=" * 80)

    return mae, rmse, r2, predictions, targets


# ============================================================
# Visualization Function
# ============================================================
def plot_training_curve(train_losses, val_losses):
    """繪製訓練曲線"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Lulu Pig - Hybrid Transformer Training Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    logger.info(f"Training curve saved to: {OUTPUT_DIR / 'training_curve.png'}")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    logger.info("=" * 80)
    logger.info("Lulu Pig - Kaggle Hybrid Transformer Training Pipeline")
    logger.info("=" * 80)

    # 1. Load Data
    ts_data, static_features, target = load_and_preprocess_data()

    # 2. Split Data
    indices = np.arange(len(target))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    ts_train, ts_test = ts_data[train_idx], ts_data[test_idx]
    static_train, static_test = static_features[train_idx], static_features[test_idx]
    target_train, target_test = target[train_idx], target[test_idx]

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # 3. Standardization
    scaler_static = StandardScaler()
    scaler_target = StandardScaler()

    static_train = scaler_static.fit_transform(static_train)
    static_test = scaler_static.transform(static_test)

    target_train = scaler_target.fit_transform(target_train.reshape(-1, 1)).flatten()
    target_test = scaler_target.transform(target_test.reshape(-1, 1)).flatten()

    # 4. Create DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(ts_train),
        torch.FloatTensor(static_train),
        torch.FloatTensor(target_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(ts_test),
        torch.FloatTensor(static_test),
        torch.FloatTensor(target_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Split validation from train
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model
    model = HybridTransformer(
        ts_input_dim=TS_INPUT_DIM,
        static_input_dim=STATIC_INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        seq_length=SEQ_LENGTH
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")

    # 6. Train Model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS, PATIENCE
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # 7. Load Best Model & Evaluate
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_transformer_model.pth"))
    mae, rmse, r2, predictions, targets = evaluate_model(model, test_loader, scaler_target)

    # 8. Save Results
    results = {
        "model_name": "HybridTransformer",
        "dataset": "Lulu Pig Production Sales (1075 records)",
        "total_params": total_params,
        "training_time": training_time,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }

    with open(OUTPUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {OUTPUT_DIR / 'training_results.json'}")

    # 9. Plot Training Curve
    plot_training_curve(train_losses, val_losses)

    logger.info("=" * 80)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
