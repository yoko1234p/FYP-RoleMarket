# Lulu 罐頭豬 Production 銷售數據集

## 數據集概述

本數據集包含 ToyzeroPlus 旗艦 IP「**Lulu 罐頭豬**」的 Production 級銷售數據，專為需求預測機器學習模型訓練設計。

**數據規模**：
- 📊 **1,075 筆記錄** (2017-2024，8年)
- 🎨 **9 種產品類型**
- 📈 **36 個特徵欄位**
- 🧠 **CLIP Embeddings** (768維視覺特徵)

---

## 產品類型 (9種)

| 產品類型 | 英文名稱 | 平均銷量 | 說明 |
|---------|---------|---------|------|
| 🎬 **2D視頻** | 2D Animation | 2,635 | 2D動畫內容 |
| 🎥 **3D視頻** | 3D Animation | 3,238 | 3D動畫內容 |
| 📚 **漫畫** | Comic | 1,937 | 漫畫出版物 |
| 🖼️ **視覺圖** | Single Visual | 2,338 | 單張視覺設計 |
| 🤝 **聯乘** | Collaboration | 3,686 | 品牌聯乘合作 |
| 🌍 **LuLu World** | LuLu World | 2,781 | LuLu 世界觀內容 |
| 📢 **公關** | PR/Seeding | 1,634 | 公關宣傳活動 |
| 💬 **表情包/貼圖** | Sticker | 4,216 | 通訊軟體貼圖 |
| 🎪 **活動** | Campaign | 3,118 | 行銷活動 |

---

## 檔案結構

```
data/lulu_production_sales/
├── historical_data.csv          # 主數據檔 (1075 rows × 33 columns)
├── clip_embeddings.npy          # CLIP 視覺特徵 (1075, 768)
├── trends_history.json          # Google Trends 歷史數據
├── data_summary.txt             # 數據摘要報告
└── README.md                    # 本文件
```

---

## 特徵說明 (36 欄位)

### 1. 識別欄位 (5)
- `design_id`: 設計唯一 ID
- `ip_id`: IP 角色 ID (固定為 `lulu_pig`)
- `ip_name`: IP 角色名稱 (固定為 `Lulu罐頭豬`)
- `product_type`: 產品類型代碼 (如 `2d_animation`)
- `product_type_name`: 產品類型中文名稱 (如 `2D視頻`)

### 2. 時間特徵 (6)
- `year`: 年份 (2017-2024)
- `quarter`: 季度 (1-4)
- `season`: 季節 (`Spring`, `Summer`, `Fall`, `Winter`)
- `month`: 月份 (1-12)
- `week_of_year`: 年中第幾週 (1-52)
- `is_holiday_season`: 是否節日季節 (0/1)

### 3. 設計特徵 (4)
- `theme`: 主題名稱 (如 `春節`, `聖誕節`)
- `product_age`: 產品年齡 (年)
- `production_cost`: 製作成本等級 (`low`, `medium`, `high`, `very_high`)
- `popularity_trend`: 人氣趨勢 (`growing`, `stable`, `declining`)

### 4. Google Trends 特徵 (7)
- `trend_score_current`: 當前趨勢分數 (40-95)
- `trend_score_q1`: Q1 趨勢分數
- `trend_score_q2`: Q2 趨勢分數
- `trend_score_q3`: Q3 趨勢分數
- `trend_score_q4`: Q4 趨勢分數
- `trend_momentum`: 趨勢動能 (當前 - Q1)
- `trend_volatility`: 趨勢波動性 (標準差)

### 5. 社群媒體特徵 (4)
- `social_reach`: 社群觸及人數
- `social_engagement`: 社群互動次數
- `sentiment_score`: 情感分數 (0.65-0.90)
- `viral_coefficient`: 病毒式傳播係數 (互動率)

### 6. 競爭特徵 (2)
- `competition_level`: 競爭程度 (`low`, `medium`, `high`, `very_high`)
- `theme_saturation`: 主題飽和度 (0.3-0.8)

### 7. 定價特徵 (2)
- `retail_price`: 零售價格 (HKD)
- `price_multiplier`: 定價倍數 (0.9-1.35)

### 8. 目標變數 (3)
- `sales_quantity`: 銷售數量 ⭐ **預測目標**
- `revenue`: 營收 (HKD)
- `sellthrough_rate`: 售罄率 (0.75-0.95)

### 9. 外部檔案
- **CLIP Embeddings** (`clip_embeddings.npy`): 768維視覺特徵向量
- **Trends History** (`trends_history.json`): 完整的 Google Trends 時序數據

---

## 使用範例

### Python 讀取數據

```python
import pandas as pd
import numpy as np
import json

# 1. 讀取 CSV
df = pd.read_csv('historical_data.csv')

# 2. 讀取 CLIP Embeddings
clip_embeddings = np.load('clip_embeddings.npy')
print(f"CLIP shape: {clip_embeddings.shape}")  # (1075, 768)

# 3. 讀取 Trends History
with open('trends_history.json', 'r', encoding='utf-8') as f:
    trends_history = json.load(f)

# 4. 查看數據
print(df.head())
print(f"Total records: {len(df)}")
print(f"Features: {df.columns.tolist()}")
```

### Kaggle Notebook

```python
# Kaggle 環境中的數據路徑
INPUT_DIR = Path('/kaggle/input/lulu-rolemarket-sales-data')

df = pd.read_csv(INPUT_DIR / 'historical_data.csv')
clip_embeddings = np.load(INPUT_DIR / 'clip_embeddings.npy')

# 開始訓練...
```

---

## 數據統計

### 銷量分布
- **最小值**: 1,396
- **最大值**: 4,763
- **平均值**: 2,847
- **中位數**: 2,818
- **標準差**: 794

### 營收統計
- **總營收**: $997,981,591 HKD
- **平均營收**: $928,355 HKD/設計

### 季節分布
- **Spring**: 270 筆 (平均 2,833)
- **Summer**: 274 筆 (平均 2,869)
- **Fall**: 267 筆 (平均 2,814)
- **Winter**: 264 筆 (平均 2,871)

---

## 適用場景

✅ **時間序列預測** (LSTM, Transformer)
✅ **需求預測** (銷量預測)
✅ **多模態學習** (文本 + 視覺特徵)
✅ **市場趨勢分析**
✅ **產品類型比較**

---

## 模型訓練建議

### 推薦模型架構
1. **Hybrid Transformer**: 結合時序特徵 (Transformer) 和靜態特徵 (FC)
2. **LSTM**: 適合處理 Google Trends 時序數據
3. **XGBoost**: 基線模型

### 特徵工程建議
1. 使用 `trend_score_q1` ~ `trend_score_q4` 作為時序輸入
2. 結合 CLIP embeddings (768維) 作為視覺特徵
3. One-hot encode: `product_type`, `season`, `theme`
4. StandardScaler normalize: 數值特徵

### 訓練/測試分割
- **Train**: 80% (860 筆)
- **Test**: 20% (215 筆)
- **時間順序分割** (推薦): 2017-2022 訓練，2023-2024 測試

---

## 授權與引用

**數據來源**: ToyzeroPlus FYP Project
**角色 IP**: Lulu 罐頭豬
**生成日期**: 2025-10-28
**用途**: 學術研究與機器學習訓練

如使用本數據集，請引用：
```
@dataset{lulu_rolemarket_2024,
  title={Lulu Pig Production Sales Dataset},
  author={ToyzeroPlus FYP Team},
  year={2024},
  publisher={Kaggle}
}
```

---

## 聯絡資訊

- **專案**: FYP-RoleMarket
- **GitHub**: [Your Repo URL]
- **Kaggle**: [Your Kaggle Profile]

---

**最後更新**: 2025-10-28
