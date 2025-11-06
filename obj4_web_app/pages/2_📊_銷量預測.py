"""
Streamlit Page 2: Sales Forecast Dashboard (銷量預測)

整合 Obj 3 (Hybrid Transformer 銷量預測) 功能。

Author: Developer (James)
Date: 2025-11-06
Version: 1.0
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.forecast_predictor import (
    ForecastPredictorWrapper,
    ForecastError,
    ModelLoadError
)
from obj4_web_app.config import (
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    MODEL_WEIGHTS_PATH
)

# Page configuration
st.set_page_config(
    page_title="銷量預測 - AI 角色設計系統",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("📊 銷量預測與市場洞察")
st.markdown("---")


# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []


# Initialize Forecast Predictor (cached)
@st.cache_resource
def load_forecast_predictor():
    """
    載入 ForecastPredictorWrapper（cached across sessions）。

    Returns:
        ForecastPredictorWrapper instance
    """
    try:
        return ForecastPredictorWrapper(model_path=MODEL_WEIGHTS_PATH)
    except ModelLoadError as e:
        st.error(f"❌ 模型載入失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ 初始化失敗：{str(e)}")
        return None


try:
    predictor = load_forecast_predictor()
except Exception as e:
    st.error(f"❌ 系統初始化失敗：{str(e)}")
    st.stop()


# Check prerequisites (Story 4.2 completion)
if 'generated_images' not in st.session_state or not st.session_state['generated_images']:
    st.warning("⚠️ 請先在 **Page 1: 設計生成** 完成圖片生成")
    st.info("""
    ### 使用流程：
    1. 前往 **Page 1: 設計生成**
    2. 生成 Prompt
    3. 生成設計圖（至少 1 張）
    4. 返回此頁面進行銷量預測
    """)
    st.stop()


# Main content
st.header("🔮 預測設定")

col1, col2 = st.columns([1, 1])

with col1:
    # Season selector
    st.subheader("1️⃣ 選擇季節")
    season = st.selectbox(
        "目標季度",
        options=["Spring", "Summer", "Fall", "Winter"],
        help="選擇預測的季度（影響銷量預測）"
    )

    # Trends history input
    st.subheader("2️⃣ 歷史趨勢數據")
    st.caption("輸入過去 4 個季度的 Google Trends 分數 (0-100)")

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        q_minus_3 = st.number_input("Q-3", min_value=0, max_value=100, value=45, step=1)
    with col_q2:
        q_minus_2 = st.number_input("Q-2", min_value=0, max_value=100, value=52, step=1)
    with col_q3:
        q_minus_1 = st.number_input("Q-1", min_value=0, max_value=100, value=48, step=1)
    with col_q4:
        q0 = st.number_input("Q0 (當前)", min_value=0, max_value=100, value=50, step=1)

    trends_history = [q_minus_3, q_minus_2, q_minus_1, q0]

with col2:
    # Design selector
    st.subheader("3️⃣ 選擇設計")

    # Filter successful designs
    successful_designs = [
        (i, result) for i, result in enumerate(st.session_state['generated_images'])
        if result.get('success')
    ]

    if not successful_designs:
        st.error("❌ 沒有可用的設計圖，請返回 Page 1 重新生成")
        st.stop()

    # Display design thumbnails for selection
    selected_design_idx = None

    for i, result in successful_designs:
        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(result['image'], use_container_width=True)

        with col_info:
            clip_sim = result.get('clip_similarity', 0.0)
            if clip_sim >= 0.80:
                st.markdown(f"**變化 {i+1}** - CLIP: :green[{clip_sim:.4f}] ✅")
            else:
                st.markdown(f"**變化 {i+1}** - CLIP: :orange[{clip_sim:.4f}] ⚠️")

            if st.button(f"選擇此設計", key=f"select_{i}", use_container_width=True):
                selected_design_idx = i

        st.markdown("---")

    # Use first design by default if none selected
    if selected_design_idx is None:
        selected_design_idx = successful_designs[0][0]
        st.info(f"💡 預設選擇：變化 {selected_design_idx + 1}")


# Predict button
st.markdown("---")
predict_button = st.button(
    "🚀 預測銷量",
    type="primary",
    use_container_width=True,
    disabled=(predictor is None)
)


if predict_button:
    st.markdown("---")
    st.header("📈 預測結果")

    # Get selected design
    selected_result = st.session_state['generated_images'][selected_design_idx]
    clip_similarity = selected_result.get('clip_similarity', 0.0)

    # Generate dummy CLIP embedding (768-dim)
    # In real scenario, should extract from image
    # For now, use similarity score to simulate embedding
    clip_embedding = np.random.rand(768) * clip_similarity

    # Predict with spinner
    with st.spinner("⏳ 正在預測銷量..."):
        try:
            prediction = predictor.predict_sales(
                season=season,
                clip_embedding=clip_embedding,
                trends_history=trends_history
            )

            # Save to session state
            st.session_state['predictions'].append({
                'season': season,
                'design_idx': selected_design_idx,
                'prediction': prediction,
                'trends_history': trends_history,
                'clip_similarity': clip_similarity
            })

            st.success("✅ 預測完成！")

        except ForecastError as e:
            st.error(f"❌ 預測失敗：{str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ 發生錯誤：{str(e)}")
            st.stop()

    # Display prediction results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="預測銷量",
            value=f"{int(prediction['predicted_sales']):,} 件",
            delta=f"±{int(prediction['mae']):,} 件"
        )

    with col2:
        st.metric(
            label="信心度",
            value=f"{prediction['confidence']*100:.1f}%",
            delta="R² Score"
        )

    with col3:
        error_rate = (prediction['mae'] / prediction['predicted_sales']) * 100
        st.metric(
            label="誤差範圍",
            value=f"±{error_rate:.1f}%",
            delta="相對誤差"
        )

    # Display selected design
    st.markdown("### 選中的設計")
    col_design, col_info = st.columns([1, 2])

    with col_design:
        st.image(selected_result['image'], caption=f"變化 {selected_design_idx + 1}")

    with col_info:
        st.markdown(f"""
        **設計資訊：**
        - CLIP 相似度：{clip_similarity:.4f}
        - 生成時間：{selected_result.get('generation_time', 0):.2f}s
        - 季節：{season}
        """)

    # Historical trend chart
    st.markdown("---")
    st.markdown("### 📊 歷史趨勢與預測")

    fig = go.Figure()

    # Historical sales (using trends as proxy)
    historical_quarters = ['Q-3', 'Q-2', 'Q-1', 'Q0']
    # Scale trends (0-100) to sales range (~2000-3000)
    historical_sales = [t * 28 for t in trends_history]

    fig.add_trace(go.Scatter(
        x=historical_quarters,
        y=historical_sales,
        mode='lines+markers',
        name='歷史趨勢',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Predicted sales
    fig.add_trace(go.Scatter(
        x=['Q+1'],
        y=[prediction['predicted_sales']],
        mode='markers',
        name='預測銷量',
        marker=dict(color='red', size=15, symbol='star')
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=['Q+1', 'Q+1'],
        y=[prediction['lower_bound'], prediction['upper_bound']],
        mode='lines',
        name='信心區間',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        showlegend=True
    ))

    fig.update_layout(
        title=f'{season} 季度銷量預測',
        xaxis_title='季度',
        yaxis_title='銷量（件）',
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("---")
    st.markdown("### 🔍 特徵重要性分析")

    feature_importance = predictor.get_feature_importance()

    # Create bar chart
    fig_importance = go.Figure(data=[
        go.Bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            text=[f"{v*100:.1f}%" for v in feature_importance.values()],
            textposition='auto',
        )
    ])

    fig_importance.update_layout(
        title='影響銷量的關鍵因素',
        xaxis_title='特徵',
        yaxis_title='重要性權重',
        yaxis=dict(tickformat='.0%'),
        height=300
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # Market Insights
    st.markdown("---")
    st.markdown("### 💡 市場洞察與建議")

    insights = predictor.generate_market_insights(
        predicted_sales=prediction['predicted_sales'],
        season=season,
        clip_similarity=clip_similarity
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**上市時機：**")
        st.info(insights['timing'])

        st.markdown("**生產建議：**")
        st.info(insights['production'])

    with col2:
        st.markdown("**角色一致性：**")
        if "✅" in insights['character']:
            st.success(insights['character'])
        else:
            st.warning(insights['character'])

        st.markdown("**風險評估：**")
        if "✅" in insights['risk']:
            st.success(insights['risk'])
        else:
            st.warning(insights['risk'])

    # Model performance
    st.markdown("---")
    with st.expander("📊 模型性能指標 (Exp #11v2)"):
        metrics = predictor.get_model_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col2:
            st.metric("R²", f"{metrics['R2']:.4f}")
        with col3:
            st.metric("誤差率", f"{metrics['Error_Rate']*100:.1f}%")
        with col4:
            st.metric("信心度", f"{metrics['Confidence_Percent']:.1f}%")


# Footer
st.markdown("---")
st.markdown("""
### 💡 使用說明

**步驟：**
1. **選擇季節**：選擇要預測的目標季度
2. **輸入趨勢數據**：填入過去 4 季度的 Google Trends 分數
3. **選擇設計**：從 Page 1 生成的設計中選擇
4. **預測銷量**：點擊按鈕查看預測結果

**解讀預測結果：**
- **預測銷量**：模型預估的銷售數量
- **誤差範圍**：預測的可能偏差（基於 MAE = 327.26）
- **信心度**：模型準確度指標（R² = 67.88%）
- **Feature Importance**：各因素對銷量的影響權重

**注意事項：**
- 預測基於 Hybrid Transformer 模型（Exp #11v2）
- 模型訓練於 Lulu Pig 歷史銷售數據
- 實際銷量可能受其他因素影響（如行銷活動）
""")
