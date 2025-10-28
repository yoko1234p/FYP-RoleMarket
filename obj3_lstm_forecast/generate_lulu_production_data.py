"""
Production 級數據生成器 - Lulu 罐頭豬專屬版本

專為 FYP 項目設計，聚焦於 Lulu 罐頭豬 IP 的 9 種產品類型：
- 2D Animation (2D視頻)
- 3D Animation (3D視頻)
- Comic (漫畫)
- Single Visual (視覺圖)
- Collaboration (聯乘)
- LuLu World
- PR/Seeding (公關)
- Sticker (表情包/貼圖)
- Campaign (活動)

目標數據量：800-1200 筆 (8年 × 9產品類型 × 12-18設計/產品/年)

Author: Product Manager (John)
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Lulu 罐頭豬產品類型定義
# ============================================================
LULU_PRODUCT_TYPES = {
    '2d_animation': {
        'name': '2D Animation',
        'name_zh': '2D視頻',
        'avg_sales': 1800,  # 視頻內容高互動
        'popularity_trend': 'growing',
        'production_cost': 'high',
        'seasonality_impact': 1.2,  # 高季節影響
    },
    '3d_animation': {
        'name': '3D Animation',
        'name_zh': '3D視頻',
        'avg_sales': 2200,  # 3D 內容更吸引
        'popularity_trend': 'growing',
        'production_cost': 'very_high',
        'seasonality_impact': 1.3,
    },
    'comic': {
        'name': 'Comic',
        'name_zh': '漫畫',
        'avg_sales': 1400,
        'popularity_trend': 'stable',
        'production_cost': 'medium',
        'seasonality_impact': 0.9,  # 較低季節影響
    },
    'single_visual': {
        'name': 'Single Visual',
        'name_zh': '視覺圖',
        'avg_sales': 1600,
        'popularity_trend': 'growing',
        'production_cost': 'low',
        'seasonality_impact': 1.1,
    },
    'collaboration': {
        'name': 'Collaboration',
        'name_zh': '聯乘',
        'avg_sales': 2500,  # 聯乘最高銷量
        'popularity_trend': 'growing',
        'production_cost': 'very_high',
        'seasonality_impact': 1.4,
    },
    'lulu_world': {
        'name': 'LuLu World',
        'name_zh': 'LuLu World',
        'avg_sales': 1900,
        'popularity_trend': 'growing',
        'production_cost': 'high',
        'seasonality_impact': 1.2,
    },
    'pr_seeding': {
        'name': 'PR/Seeding',
        'name_zh': '公關',
        'avg_sales': 1200,  # 公關導向，銷量較低
        'popularity_trend': 'stable',
        'production_cost': 'medium',
        'seasonality_impact': 0.8,
    },
    'sticker': {
        'name': 'Sticker',
        'name_zh': '表情包/貼圖',
        'avg_sales': 3000,  # 表情包高頻使用
        'popularity_trend': 'growing',
        'production_cost': 'low',
        'seasonality_impact': 0.7,  # 最低季節影響
    },
    'campaign': {
        'name': 'Campaign',
        'name_zh': '活動',
        'avg_sales': 2100,
        'popularity_trend': 'growing',
        'production_cost': 'high',
        'seasonality_impact': 1.5,  # 最高季節影響
    },
}

# ============================================================
# 季節主題庫 (針對香港/亞洲市場)
# ============================================================
SEASONAL_THEMES = {
    'Spring': [
        {'name': '春節', 'trend_boost': 50, 'price_multiplier': 1.3, 'competition': 'very_high'},
        {'name': '情人節', 'trend_boost': 40, 'price_multiplier': 1.2, 'competition': 'high'},
        {'name': '櫻花季', 'trend_boost': 35, 'price_multiplier': 1.0, 'competition': 'medium'},
        {'name': '復活節', 'trend_boost': 25, 'price_multiplier': 0.95, 'competition': 'low'},
    ],
    'Summer': [
        {'name': '兒童節', 'trend_boost': 35, 'price_multiplier': 1.1, 'competition': 'medium'},
        {'name': '端午節', 'trend_boost': 30, 'price_multiplier': 1.0, 'competition': 'medium'},
        {'name': '暑假', 'trend_boost': 45, 'price_multiplier': 1.2, 'competition': 'high'},
        {'name': '海洋主題', 'trend_boost': 28, 'price_multiplier': 0.95, 'competition': 'medium'},
    ],
    'Fall': [
        {'name': '開學季', 'trend_boost': 38, 'price_multiplier': 1.1, 'competition': 'high'},
        {'name': '中秋節', 'trend_boost': 42, 'price_multiplier': 1.25, 'competition': 'high'},
        {'name': '萬聖節', 'trend_boost': 35, 'price_multiplier': 1.0, 'competition': 'medium'},
        {'name': '感恩節', 'trend_boost': 20, 'price_multiplier': 0.9, 'competition': 'low'},
    ],
    'Winter': [
        {'name': '聖誕節', 'trend_boost': 55, 'price_multiplier': 1.35, 'competition': 'very_high'},
        {'name': '冬季溫暖', 'trend_boost': 30, 'price_multiplier': 1.0, 'competition': 'medium'},
        {'name': '跨年', 'trend_boost': 40, 'price_multiplier': 1.15, 'competition': 'high'},
        {'name': '冰雪主題', 'trend_boost': 32, 'price_multiplier': 1.05, 'competition': 'medium'},
    ],
}

SEASON_MULTIPLIERS = {
    'Spring': 1.1,
    'Summer': 1.05,
    'Fall': 0.95,
    'Winter': 1.15,
}

COMPETITION_PENALTY = {
    'very_high': -0.15,
    'high': -0.10,
    'medium': -0.05,
    'low': 0.0,
}

# ============================================================
# 數據生成類
# ============================================================
class LuluProductionDataGenerator:
    def __init__(
        self,
        start_year: int = 2017,
        end_year: int = 2024,
        output_dir: str = "data/lulu_production_sales",
        seed: int = 42
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(seed)
        self.generated_data = []

        logger.info(f"Initialized Lulu Production Data Generator")
        logger.info(f"  Time Range: {start_year}-{end_year} ({end_year - start_year + 1} years)")
        logger.info(f"  Product Types: {len(LULU_PRODUCT_TYPES)}")
        logger.info(f"  Target: 800-1200 records")

    def generate_all_data(self):
        """生成所有產品類型的數據"""
        logger.info("Starting data generation...")

        for year in range(self.start_year, self.end_year + 1):
            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                quarter = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}[season]

                # 每個產品類型在每個季節生成多個設計
                for product_id, product_info in LULU_PRODUCT_TYPES.items():
                    # 每個產品類型每季生成 3-5 個設計
                    num_designs = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])

                    # 隨機選擇該季節的主題
                    themes = SEASONAL_THEMES[season]
                    selected_themes = np.random.choice(
                        range(len(themes)),
                        size=min(num_designs, len(themes)),
                        replace=False
                    )

                    for theme_idx in selected_themes:
                        theme = themes[theme_idx]

                        # 生成單筆記錄
                        record = self._generate_single_record(
                            year=year,
                            quarter=quarter,
                            season=season,
                            product_id=product_id,
                            product_info=product_info,
                            theme=theme
                        )
                        self.generated_data.append(record)

        logger.info(f"Generated {len(self.generated_data)} records")
        return self.generated_data

    def _generate_single_record(
        self,
        year: int,
        quarter: int,
        season: str,
        product_id: str,
        product_info: dict,
        theme: dict
    ) -> dict:
        """生成單筆銷售記錄"""

        # 生成設計 ID
        design_id = f"lulu_{product_id}_{year}_{season.lower()}_{theme['name']}"
        design_id = design_id.replace('/', '_').replace(' ', '_')

        # 1. Google Trends (40-95)
        base_trend = 70
        trend_boost = theme['trend_boost']
        trend_score = base_trend + trend_boost + np.random.normal(0, 5)
        trend_score = np.clip(trend_score, 40, 95)

        # 生成 Google Trends 歷史（過去 4 季）
        trends_history = [
            max(40, trend_score + np.random.normal(0, 8)) for _ in range(4)
        ]

        # 2. 社群媒體數據
        social_base = product_info['avg_sales'] * 2.5  # 觸及 = 銷量 × 2.5
        social_reach = int(social_base * (1 + np.random.uniform(-0.2, 0.3)))
        social_engagement = int(social_reach * np.random.uniform(0.08, 0.15))
        sentiment_score = np.random.uniform(0.65, 0.90)

        # 3. 模擬銷量
        sales_quantity = self._simulate_sales_quantity(
            product_info=product_info,
            theme=theme,
            year=year,
            season=season,
            trend_score=trend_score,
            social_engagement=social_engagement
        )

        # 4. 定價策略
        base_price = 299  # Lulu 產品基礎價格
        retail_price = base_price * theme['price_multiplier']

        # 5. 營收計算
        revenue = sales_quantity * retail_price

        # 6. CLIP Embedding (模擬，768維)
        clip_embedding = np.random.randn(768).astype(np.float32)

        # 7. 時間特徵
        is_holiday_season = 1 if season in ['Winter', 'Spring'] else 0
        month = quarter * 3 - 1  # 每季第二個月
        week_of_year = (quarter - 1) * 13 + 7

        # 8. 產品年齡
        product_age = year - self.start_year + 1

        # 組裝完整記錄
        record = {
            # 識別欄位
            'design_id': design_id,
            'ip_id': 'lulu_pig',
            'ip_name': 'Lulu罐頭豬',
            'product_type': product_id,
            'product_type_name': product_info['name_zh'],

            # 時間特徵
            'year': year,
            'quarter': quarter,
            'season': season,
            'month': month,
            'week_of_year': week_of_year,
            'is_holiday_season': is_holiday_season,

            # 設計特徵
            'theme': theme['name'],
            'product_age': product_age,
            'production_cost': product_info['production_cost'],
            'popularity_trend': product_info['popularity_trend'],

            # Google Trends 特徵
            'trend_score_current': round(trend_score, 2),
            'trend_score_q1': round(trends_history[0], 2),
            'trend_score_q2': round(trends_history[1], 2),
            'trend_score_q3': round(trends_history[2], 2),
            'trend_score_q4': round(trends_history[3], 2),
            'trend_momentum': round(trend_score - trends_history[0], 2),
            'trend_volatility': round(np.std(trends_history), 2),

            # 社群媒體特徵
            'social_reach': social_reach,
            'social_engagement': social_engagement,
            'sentiment_score': round(sentiment_score, 4),
            'viral_coefficient': round(social_engagement / social_reach, 4),

            # 競爭特徵
            'competition_level': theme['competition'],
            'theme_saturation': np.random.uniform(0.3, 0.8),

            # 定價特徵
            'retail_price': round(retail_price, 2),
            'price_multiplier': theme['price_multiplier'],

            # 目標變數
            'sales_quantity': int(sales_quantity),
            'revenue': round(revenue, 2),
            'sellthrough_rate': round(np.random.uniform(0.75, 0.95), 4),

            # CLIP Embedding (單獨保存)
            '_clip_embedding': clip_embedding,
            '_trends_history': trends_history,
        }

        return record

    def _simulate_sales_quantity(
        self,
        product_info: dict,
        theme: dict,
        year: int,
        season: str,
        trend_score: float,
        social_engagement: int
    ) -> float:
        """複雜的銷量模擬算法（10 因素）"""

        base_sales = product_info['avg_sales']

        # 1. Google Trends 影響 (30%)
        trend_factor = (trend_score / 100) * 0.30

        # 2. 主題熱度影響 (20%)
        theme_boost = theme['trend_boost']
        theme_factor = (theme_boost / 100) * 0.20

        # 3. 季節因素 (15%)
        season_multiplier = SEASON_MULTIPLIERS[season]
        season_factor = (season_multiplier - 1.0) * 0.15

        # 4. 產品類型季節性 (10%)
        product_seasonality = product_info['seasonality_impact']
        product_season_factor = (product_seasonality - 1.0) * 0.10

        # 5. 社群互動影響 (10%)
        social_factor = (social_engagement / (base_sales * 2.5 * 0.12)) * 0.10

        # 6. 年份成長趨勢 (5%)
        year_growth = (year - self.start_year) * 0.03  # 每年 3% 成長
        year_factor = year_growth * 0.05

        # 7. 產品趨勢 (5%)
        popularity_map = {'growing': 0.05, 'stable': 0, 'declining': -0.05}
        popularity_factor = popularity_map[product_info['popularity_trend']]

        # 8. 競爭懲罰 (-5% to 0%)
        competition_factor = COMPETITION_PENALTY[theme['competition']]

        # 9. 定價影響 (-5% to +5%)
        price_factor = (1.0 - theme['price_multiplier']) * 0.1

        # 10. 隨機噪音 (±15%)
        noise = np.random.uniform(-0.15, 0.15)

        # 總合因素
        total_factor = 1.0 + (
            trend_factor +
            theme_factor +
            season_factor +
            product_season_factor +
            social_factor +
            year_factor +
            popularity_factor +
            competition_factor +
            price_factor +
            noise
        )

        sales = base_sales * total_factor

        # 限制在合理範圍
        production_cap = base_sales * 1.8  # 最高 180% 基準銷量
        production_floor = base_sales * 0.5  # 最低 50% 基準銷量

        return np.clip(sales, production_floor, production_cap)

    def save_to_files(self):
        """保存數據到檔案"""
        if not self.generated_data:
            raise ValueError("No data generated yet. Call generate_all_data() first.")

        logger.info("Saving data to files...")

        # 1. 保存 CSV（不含 CLIP embeddings）
        df_data = []
        clip_embeddings = []
        trends_history = {}

        for record in self.generated_data:
            # 提取並移除 CLIP embedding
            clip_emb = record.pop('_clip_embedding')
            clip_embeddings.append(clip_emb)

            # 提取並移除 trends history
            trends_hist = record.pop('_trends_history')
            trends_history[record['design_id']] = trends_hist

            df_data.append(record)

        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "historical_data.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"  Saved CSV: {csv_path} ({len(df)} rows)")

        # 2. 保存 CLIP embeddings (numpy)
        clip_array = np.array(clip_embeddings, dtype=np.float32)
        npy_path = self.output_dir / "clip_embeddings.npy"
        np.save(npy_path, clip_array)
        logger.info(f"  Saved CLIP embeddings: {npy_path} {clip_array.shape}")

        # 3. 保存 Trends history (JSON)
        json_path = self.output_dir / "trends_history.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(trends_history, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved Trends history: {json_path}")

        # 4. 生成數據摘要
        self._generate_summary(df)

        logger.info(f"All files saved to: {self.output_dir}")
        return {
            'csv_path': csv_path,
            'npy_path': npy_path,
            'json_path': json_path,
            'num_records': len(df)
        }

    def _generate_summary(self, df: pd.DataFrame):
        """生成數據摘要報告"""
        summary_path = self.output_dir / "data_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Lulu 罐頭豬 Production 數據集摘要\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 數據規模
            f.write("【數據規模】\n")
            f.write(f"  總記錄數: {len(df)}\n")
            f.write(f"  時間範圍: {self.start_year}-{self.end_year} ({self.end_year - self.start_year + 1}年)\n")
            f.write(f"  產品類型數: {df['product_type'].nunique()}\n")
            f.write(f"  設計數量: {df['design_id'].nunique()}\n\n")

            # 銷量統計
            f.write("【銷量統計】\n")
            f.write(f"  最小值: {df['sales_quantity'].min()}\n")
            f.write(f"  最大值: {df['sales_quantity'].max()}\n")
            f.write(f"  平均值: {df['sales_quantity'].mean():.2f}\n")
            f.write(f"  中位數: {df['sales_quantity'].median():.2f}\n")
            f.write(f"  標準差: {df['sales_quantity'].std():.2f}\n\n")

            # 營收統計
            f.write("【營收統計】\n")
            f.write(f"  總營收: ${df['revenue'].sum():,.2f}\n")
            f.write(f"  平均營收: ${df['revenue'].mean():,.2f}\n\n")

            # 各產品類型表現
            f.write("【各產品類型表現】\n")
            for product_type in sorted(df['product_type'].unique()):
                product_data = df[df['product_type'] == product_type]
                product_name = product_data['product_type_name'].iloc[0]
                f.write(f"  {product_name} ({product_type}):\n")
                f.write(f"    設計數: {len(product_data)}\n")
                f.write(f"    平均銷量: {product_data['sales_quantity'].mean():.2f}\n")
                f.write(f"    總銷量: {product_data['sales_quantity'].sum()}\n")
                f.write(f"    總營收: ${product_data['revenue'].sum():,.2f}\n")

            # 季節分布
            f.write("\n【季節分布】\n")
            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                season_data = df[df['season'] == season]
                f.write(f"  {season}: {len(season_data)} 筆, 平均 {season_data['sales_quantity'].mean():.2f}\n")

            # Top 主題
            f.write("\n【Top 10 熱門主題】\n")
            top_themes = df.groupby('theme')['sales_quantity'].agg(['count', 'mean']).sort_values('mean', ascending=False).head(10)
            for theme, row in top_themes.iterrows():
                f.write(f"  {theme}: {row['count']} 次, 平均銷量 {row['mean']:.2f}\n")

        logger.info(f"  Saved summary: {summary_path}")


# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Lulu 罐頭豬 Production 數據生成器")
    logger.info("=" * 80)

    # 創建生成器
    generator = LuluProductionDataGenerator(
        start_year=2017,
        end_year=2024,
        output_dir="data/lulu_production_sales",
        seed=42
    )

    # 生成數據
    data = generator.generate_all_data()

    # 保存檔案
    result = generator.save_to_files()

    logger.info("=" * 80)
    logger.info("✅ 數據生成完成！")
    logger.info(f"   記錄數: {result['num_records']}")
    logger.info(f"   CSV: {result['csv_path']}")
    logger.info(f"   CLIP: {result['npy_path']}")
    logger.info(f"   Trends: {result['json_path']}")
    logger.info("=" * 80)
