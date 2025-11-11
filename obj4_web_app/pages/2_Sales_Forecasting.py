"""
Streamlit Page 2: Sales Forecast Dashboard

Integrates Obj 3 (Hybrid Transformer Sales Forecasting) functionality.

Author: Developer (James)
Date: 2025-11-09
Version: 1.1
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
from obj4_web_app.utils.firebase_manager import FirebaseManager, FirebaseError
from obj4_web_app.config import (
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    MODEL_WEIGHTS_PATH
)

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting - AI Character Design System",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("📊 Sales Forecasting & Market Insights")
st.markdown("---")


# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []


# Initialize Forecast Predictor (cached)
@st.cache_resource
def load_forecast_predictor():
    """Load ForecastPredictorWrapper (cached across sessions)."""
    try:
        return ForecastPredictorWrapper(model_path=MODEL_WEIGHTS_PATH)
    except ModelLoadError as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return None


# Initialize Firebase Manager (cached)
@st.cache_resource
def load_firebase_manager():
    """Load FirebaseManager (cached across sessions)."""
    try:
        return FirebaseManager()
    except FirebaseError as e:
        st.warning(f"⚠️ Firebase initialization failed: {str(e)}")
        st.info("Record saving will not be available. Please check Firebase configuration in .env file.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Unexpected error initializing Firebase: {str(e)}")
        return None


try:
    predictor = load_forecast_predictor()
    firebase_manager = load_firebase_manager()
except Exception as e:
    st.error(f"❌ System initialization failed: {str(e)}")
    st.stop()


# Check prerequisites (Story 4.2 completion)
if 'generated_images' not in st.session_state or not st.session_state['generated_images']:
    st.warning("⚠️ Please complete image generation on **Page 1: Design Generation** first")
    st.info("""
    ### Workflow:
    1. Go to **Page 1: Design Generation**
    2. Generate Prompt
    3. Generate design images (at least 1)
    4. Return to this page for sales forecasting
    """)
    st.stop()


# Main content
st.header("🔮 Forecast Settings")

col1, col2 = st.columns([1, 1])

with col1:
    # Season selector
    st.subheader("1️⃣ Select Season")
    season = st.selectbox(
        "Target Quarter",
        options=["Spring", "Summer", "Fall", "Winter"],
        help="Choose the forecast season (affects sales prediction)"
    )

    # Trends history input
    st.subheader("2️⃣ Historical Trend Data")
    st.caption("Enter Google Trends scores from the past 4 quarters (0-100)")

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        q_minus_3 = st.number_input("Q-3", min_value=0, max_value=100, value=45, step=1)
    with col_q2:
        q_minus_2 = st.number_input("Q-2", min_value=0, max_value=100, value=52, step=1)
    with col_q3:
        q_minus_1 = st.number_input("Q-1", min_value=0, max_value=100, value=48, step=1)
    with col_q4:
        q0 = st.number_input("Q0 (Current)", min_value=0, max_value=100, value=50, step=1)

    trends_history = [q_minus_3, q_minus_2, q_minus_1, q0]

with col2:
    # Design selector
    st.subheader("3️⃣ Select Design")

    # Filter successful designs
    successful_designs = [
        (i, result) for i, result in enumerate(st.session_state['generated_images'])
        if result.get('success')
    ]

    if not successful_designs:
        st.error("❌ No available designs, please return to Page 1 to regenerate")
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
                st.markdown(f"**Variation {i+1}** - CLIP: :green[{clip_sim:.4f}] ✅")
            else:
                st.markdown(f"**Variation {i+1}** - CLIP: :orange[{clip_sim:.4f}] ⚠️")

            if st.button(f"Select This Design", key=f"select_{i}", use_container_width=True):
                selected_design_idx = i

        st.markdown("---")

    # Use first design by default if none selected
    if selected_design_idx is None:
        selected_design_idx = successful_designs[0][0]
        st.info(f"💡 Default selection: Variation {selected_design_idx + 1}")


# Predict button
st.markdown("---")
predict_button = st.button(
    "🚀 Predict Sales",
    type="primary",
    use_container_width=True,
    disabled=(predictor is None)
)


if predict_button:
    st.markdown("---")
    st.header("📈 Prediction Results")

    # Get selected design
    selected_result = st.session_state['generated_images'][selected_design_idx]
    clip_similarity = selected_result.get('clip_similarity', 0.0)
    clip_embedding = selected_result.get('clip_embedding')

    # Validate CLIP embedding
    if clip_embedding is None:
        st.error("❌ Selected design does not have CLIP embedding. Please regenerate the design.")
        st.stop()

    # Ensure embedding is the correct shape
    if clip_embedding.shape != (768,):
        st.error(f"❌ Invalid CLIP embedding shape: {clip_embedding.shape}. Expected (768,)")
        st.stop()

    st.info(f"📊 Using real CLIP embedding from selected design (similarity: {clip_similarity:.4f})")

    # Predict with spinner
    with st.spinner("⏳ Predicting sales..."):
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

            # Save to Firebase (if available)
            if firebase_manager:
                try:
                    # Get design_id from Firebase (if available)
                    design_id = selected_result.get('firebase_doc_id', 'unknown')

                    # Create prediction record
                    prediction_doc_id = firebase_manager.create_prediction_record(
                        design_id=design_id,
                        season=season,
                        predicted_sales=prediction['predicted_sales'],
                        lower_bound=prediction['lower_bound'],
                        upper_bound=prediction['upper_bound'],
                        confidence=prediction['confidence'],
                        mae=prediction['mae'],
                        trends_history=trends_history,
                        clip_similarity=clip_similarity,
                        metadata={
                            'design_idx': selected_design_idx,
                            'model_version': 'Experiment #11v2'
                        }
                    )

                    # Add Firebase doc ID to last prediction
                    st.session_state['predictions'][-1]['firebase_doc_id'] = prediction_doc_id

                    st.success("✅ Prediction saved to Firebase!")

                except FirebaseError as e:
                    st.warning(f"⚠️ Failed to save prediction to Firebase: {str(e)}")
                except Exception as e:
                    st.warning(f"⚠️ Unexpected error saving to Firebase: {str(e)}")

            st.success("✅ Prediction complete!")

        except ForecastError as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            st.stop()

    # Display prediction results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Sales",
            value=f"{int(prediction['predicted_sales']):,} units",
            delta=f"±{int(prediction['mae']):,} units"
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{prediction['confidence']*100:.1f}%",
            delta="R² Score"
        )

    with col3:
        error_rate = (prediction['mae'] / prediction['predicted_sales']) * 100
        st.metric(
            label="Error Range",
            value=f"±{error_rate:.1f}%",
            delta="Relative Error"
        )

    # Display selected design
    st.markdown("### Selected Design")
    col_design, col_info = st.columns([1, 2])

    with col_design:
        st.image(selected_result['image'], caption=f"Variation {selected_design_idx + 1}")

    with col_info:
        st.markdown(f"""
        **Design Information:**
        - CLIP Similarity: {clip_similarity:.4f}
        - Generation Time: {selected_result.get('generation_time', 0):.2f}s
        - Season: {season}
        """)

    # Historical trend chart
    st.markdown("---")
    st.markdown("### 📊 Historical Trends & Forecast")

    fig = go.Figure()

    # Historical sales (using trends as proxy)
    historical_quarters = ['Q-3', 'Q-2', 'Q-1', 'Q0']
    # Scale trends (0-100) to sales range (~2000-3000)
    historical_sales = [t * 28 for t in trends_history]

    fig.add_trace(go.Scatter(
        x=historical_quarters,
        y=historical_sales,
        mode='lines+markers',
        name='Historical Trends',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Predicted sales
    fig.add_trace(go.Scatter(
        x=['Q+1'],
        y=[prediction['predicted_sales']],
        mode='markers',
        name='Predicted Sales',
        marker=dict(color='red', size=15, symbol='star')
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=['Q+1', 'Q+1'],
        y=[prediction['lower_bound'], prediction['upper_bound']],
        mode='lines',
        name='Confidence Interval',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        showlegend=True
    ))

    fig.update_layout(
        title=f'{season} Quarter Sales Forecast',
        xaxis_title='Quarter',
        yaxis_title='Sales (units)',
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("---")
    st.markdown("### 🔍 Feature Importance Analysis")

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
        title='Key Factors Affecting Sales',
        xaxis_title='Feature',
        yaxis_title='Importance Weight',
        yaxis=dict(tickformat='.0%'),
        height=300
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # Market Insights
    st.markdown("---")
    st.markdown("### 💡 Market Insights & Recommendations")

    insights = predictor.generate_market_insights(
        predicted_sales=prediction['predicted_sales'],
        season=season,
        clip_similarity=clip_similarity
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Launch Timing:**")
        st.info(insights['timing'])

        st.markdown("**Production Recommendation:**")
        st.info(insights['production'])

    with col2:
        st.markdown("**Character Consistency:**")
        if "✅" in insights['character']:
            st.success(insights['character'])
        else:
            st.warning(insights['character'])

        st.markdown("**Risk Assessment:**")
        if "✅" in insights['risk']:
            st.success(insights['risk'])
        else:
            st.warning(insights['risk'])

    # Model performance
    st.markdown("---")
    with st.expander("📊 Model Performance Metrics (Exp #11v2)"):
        metrics = predictor.get_model_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col2:
            st.metric("R²", f"{metrics['R2']:.4f}")
        with col3:
            st.metric("Error Rate", f"{metrics['Error_Rate']*100:.1f}%")
        with col4:
            st.metric("Confidence", f"{metrics['Confidence_Percent']:.1f}%")


# Footer
st.markdown("---")
st.markdown("""
### 💡 Usage Instructions

**Steps:**
1. **Select Season**: Choose the target quarter for prediction
2. **Input Trend Data**: Enter Google Trends scores from the past 4 quarters
3. **Select Design**: Choose from designs generated on Page 1
4. **Predict Sales**: Click button to view forecast results

**Interpreting Results:**
- **Predicted Sales**: Model's estimated sales quantity
- **Error Range**: Possible deviation of prediction (based on MAE = 327.26)
- **Confidence**: Model accuracy indicator (R² = 67.88%)
- **Feature Importance**: Impact weight of each factor on sales

**Notes:**
- Prediction based on Hybrid Transformer model (Exp #11v2)
- Model trained on Lulu Pig historical sales data
- Actual sales may be affected by other factors (e.g., marketing campaigns)
""")
