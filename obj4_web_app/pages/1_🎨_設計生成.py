"""
Streamlit Page 1: Design Generation (設計生成)

整合 Obj 1 (Trend Analysis + Prompt Generation) 功能。

Author: Developer (James)
Date: 2025-11-06
Version: 1.0
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.trends_api import TrendsAPIWrapper, PromptGenerationError
from obj4_web_app.utils.design_generator import DesignGeneratorWrapper, DesignGenerationError
from obj4_web_app.utils.trends_extractor_wrapper import TrendsExtractorWrapper, TrendsExtractionError
from obj4_web_app.config import (
    DEFAULT_REGION,
    DEFAULT_LANG,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    CLIP_SIMILARITY_THRESHOLD,
    REFERENCE_IMAGES_DIR
)
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="設計生成 - AI 角色設計系統",
    page_icon="🎨",
    layout="wide"
)

# Page title
st.title("🎨 設計生成 - 趨勢分析與 Prompt 生成")
st.markdown("---")


# Initialize session state
if 'generated_prompt' not in st.session_state:
    st.session_state['generated_prompt'] = None

if 'last_keywords' not in st.session_state:
    st.session_state['last_keywords'] = ""

if 'last_character_name' not in st.session_state:
    st.session_state['last_character_name'] = ""

if 'generated_images' not in st.session_state:
    st.session_state['generated_images'] = []

if 'clip_embeddings' not in st.session_state:
    st.session_state['clip_embeddings'] = []

if 'extracted_trends' not in st.session_state:
    st.session_state['extracted_trends'] = []

if 'selected_keywords' not in st.session_state:
    st.session_state['selected_keywords'] = []


# Initialize API wrappers (cached)
@st.cache_resource
def load_trends_api():
    """
    載入 TrendsAPIWrapper（cached across sessions）。

    Returns:
        TrendsAPIWrapper instance
    """
    return TrendsAPIWrapper(region=DEFAULT_REGION, lang=DEFAULT_LANG)


@st.cache_resource
def load_trends_extractor():
    """
    載入 TrendsExtractorWrapper（cached across sessions）。

    Returns:
        TrendsExtractorWrapper instance
    """
    return TrendsExtractorWrapper(region=DEFAULT_REGION, lang=DEFAULT_LANG)


@st.cache_resource
def load_design_generator():
    """
    載入 DesignGeneratorWrapper（cached across sessions）。

    Returns:
        DesignGeneratorWrapper instance
    """
    try:
        return DesignGeneratorWrapper()
    except Exception as e:
        st.warning(f"⚠️ Design Generator 初始化失敗：{str(e)}")
        st.info("圖片生成功能將不可用。請檢查 GOOGLE_API_KEY 環境變數。")
        return None


try:
    api_wrapper = load_trends_api()
    trends_extractor = load_trends_extractor()
    design_generator = load_design_generator()
except Exception as e:
    st.error(f"❌ 系統初始化失敗：{str(e)}")
    st.stop()


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 輸入趨勢資訊")

    # Character information
    st.subheader("1️⃣ 角色資訊")
    character_name = st.text_input(
        "角色名稱",
        value="Lulu Pig",
        help="輸入角色名稱，例如：Lulu Pig"
    )

    character_desc = st.text_area(
        "角色描述",
        value="可愛粉紅豬，大眼睛，圓滾滾身材",
        help="簡短描述角色特徵，例如：可愛粉紅豬，大眼睛"
    )

    # Trend keywords
    st.subheader("2️⃣ 趨勢關鍵字")

    # Tabs for manual input vs auto-extraction
    tab1, tab2 = st.tabs(["🔍 自動提取 (Google Trends)", "✍️ 手動輸入"])

    with tab1:
        st.markdown("**從 Google Trends 自動提取熱門關鍵字**")

        # Theme selector
        all_themes = trends_extractor.get_all_themes()
        theme_options = [t['display'] for t in all_themes]
        theme_values = [t['value'] for t in all_themes]

        # Get current month for suggestions
        current_month = datetime.now().month
        suggested_themes = trends_extractor.get_theme_suggestions(current_month)

        # Show suggestions
        if suggested_themes:
            st.info(f"💡 本月推薦主題：{', '.join([trends_extractor.THEME_DISPLAY_NAMES[t] for t in suggested_themes])}")

        selected_theme_display = st.selectbox(
            "選擇主題",
            options=theme_options,
            help="選擇一個主題以提取相關熱門關鍵字"
        )

        # Get theme value
        selected_theme_idx = theme_options.index(selected_theme_display)
        selected_theme = theme_values[selected_theme_idx]

        col_extract, col_top_n = st.columns([3, 1])

        with col_extract:
            extract_button = st.button(
                "🔍 提取熱門關鍵字",
                use_container_width=True,
                type="secondary"
            )

        with col_top_n:
            top_n = st.number_input(
                "數量",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="提取前 N 個熱門關鍵字"
            )

        # Extract trends
        if extract_button:
            with st.spinner(f"⏳ 正在從 Google Trends 提取 {selected_theme_display} 的熱門關鍵字..."):
                try:
                    keywords = trends_extractor.get_trending_keywords(
                        theme=selected_theme,
                        timeframe='today 12-m',
                        top_n=top_n
                    )

                    if keywords:
                        st.session_state['extracted_trends'] = keywords
                        st.session_state['selected_keywords'] = []  # Reset selection
                        st.success(f"✅ 成功提取 {len(keywords)} 個關鍵字！")
                    else:
                        st.warning("⚠️ 未找到相關趨勢數據，請嘗試其他主題")

                except TrendsExtractionError as e:
                    st.error(f"❌ 提取失敗：{str(e)}")
                except Exception as e:
                    st.error(f"❌ 發生錯誤：{str(e)}")

        # Display extracted trends with checkboxes
        if st.session_state['extracted_trends']:
            st.markdown("---")
            st.markdown(f"**提取結果（過去 12 個月）：**")

            # Select all / deselect all buttons
            col_select_all, col_deselect_all = st.columns(2)
            with col_select_all:
                if st.button("✅ 全選", use_container_width=True):
                    st.session_state['selected_keywords'] = [
                        kw['keyword'] for kw in st.session_state['extracted_trends']
                    ]
                    st.rerun()

            with col_deselect_all:
                if st.button("❌ 全不選", use_container_width=True):
                    st.session_state['selected_keywords'] = []
                    st.rerun()

            # Keyword checkboxes
            for kw_data in st.session_state['extracted_trends']:
                keyword = kw_data['keyword']
                trend_score = kw_data['trend_score']
                rank = kw_data['rank']
                is_high_trend = kw_data['is_high_trend']

                # Emoji indicator
                emoji = "🔥" if is_high_trend else "📊"

                # Checkbox state
                is_selected = keyword in st.session_state['selected_keywords']

                col_checkbox, col_info = st.columns([4, 1])

                with col_checkbox:
                    if st.checkbox(
                        f"{emoji} {keyword}",
                        value=is_selected,
                        key=f"kw_{rank}_{keyword}"
                    ):
                        if keyword not in st.session_state['selected_keywords']:
                            st.session_state['selected_keywords'].append(keyword)
                    else:
                        if keyword in st.session_state['selected_keywords']:
                            st.session_state['selected_keywords'].remove(keyword)

                with col_info:
                    st.caption(f"Trend: {trend_score}")

            # Format selected keywords
            if st.session_state['selected_keywords']:
                formatted_keywords = trends_extractor.format_keywords_for_prompt(
                    st.session_state['selected_keywords']
                )
                keywords_input = formatted_keywords

                st.markdown("---")
                st.markdown(f"**已選擇 {len(st.session_state['selected_keywords'])} 個關鍵字：**")
                st.info(formatted_keywords)
            else:
                keywords_input = ""
        else:
            st.info("👆 點擊「提取熱門關鍵字」按鈕開始")
            keywords_input = ""

    with tab2:
        st.markdown("**手動輸入趨勢關鍵字**")
        keywords_input_manual = st.text_input(
            "關鍵字（逗號分隔）",
            value="春節, 紅色, 喜慶, 燈籠",
            help="輸入趨勢關鍵字，用逗號分隔，例如：春節, 紅色, 喜慶",
            key="manual_keywords"
        )

        st.info("💡 提示：也可以前往 [Google Trends](https://trends.google.com.hk/) 查看熱門關鍵字")

        # Use manual input if in manual tab
        if keywords_input_manual.strip():
            keywords_input = keywords_input_manual

    # Generate button
    generate_button = st.button(
        "🚀 生成 Prompt",
        type="primary",
        use_container_width=True
    )

with col2:
    st.header("✨ 生成結果")

    # Trend Score Visualization (if trends extracted)
    if st.session_state['extracted_trends']:
        with st.expander("📊 Trend Score 視覺化", expanded=True):
            st.markdown("**過去 12 個月搜尋熱度：**")

            # Prepare data for Plotly
            keywords = [kw['keyword'] for kw in st.session_state['extracted_trends']]
            scores = [kw['trend_score'] for kw in st.session_state['extracted_trends']]
            is_selected_list = [
                kw['keyword'] in st.session_state['selected_keywords']
                for kw in st.session_state['extracted_trends']
            ]

            # Color based on selection
            colors = ['#1f77b4' if selected else '#d3d3d3' for selected in is_selected_list]

            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=keywords,
                    y=scores,
                    marker_color=colors,
                    text=[f"{score:.1f}" for score in scores],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Trend Score: %{y:.2f}<extra></extra>'
                )
            ])

            fig.update_layout(
                title="關鍵字搜尋熱度 (Google Trends)",
                xaxis_title="關鍵字",
                yaxis_title="Trend Score",
                height=300,
                showlegend=False,
                xaxis={'tickangle': -45}
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption("🔵 藍色 = 已選擇 | ⚪ 灰色 = 未選擇")

    # Generation logic
    if generate_button:
        # Validation
        if not keywords_input.strip():
            st.error(ERROR_MESSAGES['empty_keywords'])
        elif not character_name.strip():
            st.error("❌ 請輸入角色名稱")
        elif not character_desc.strip():
            st.error("❌ 請輸入角色描述")
        else:
            # Extract keywords
            keywords_list = api_wrapper.extract_keywords_simple(keywords_input)

            if not keywords_list:
                st.error(ERROR_MESSAGES['empty_keywords'])
            else:
                # Display keywords
                st.subheader("📋 已提取關鍵字")
                st.write(", ".join(keywords_list))

                # Generate prompt with progress bar
                with st.spinner("⏳ 正在生成 Prompt..."):
                    try:
                        generated_prompt = api_wrapper.generate_prompt(
                            character_name=character_name,
                            character_desc=character_desc,
                            trend_keywords=keywords_list,
                            max_retries=3
                        )

                        # Save to session state
                        st.session_state['generated_prompt'] = generated_prompt
                        st.session_state['last_keywords'] = keywords_input
                        st.session_state['last_character_name'] = character_name

                        # Success message
                        st.success(SUCCESS_MESSAGES['prompt_generated'])

                    except PromptGenerationError as e:
                        st.error(f"❌ Prompt 生成失敗：{str(e)}")
                    except Exception as e:
                        st.error(ERROR_MESSAGES['api_error'].format(error=str(e)))

    # Display generated prompt
    if st.session_state['generated_prompt']:
        st.subheader("📝 生成的 Prompt")

        # Display in code block
        st.code(
            st.session_state['generated_prompt'],
            language="text"
        )

        # Copy button
        st.download_button(
            label="📋 複製 Prompt",
            data=st.session_state['generated_prompt'],
            file_name=f"prompt_{st.session_state['last_character_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )

        # Display metadata
        st.caption(f"角色：{st.session_state['last_character_name']} | 關鍵字：{st.session_state['last_keywords']}")
    else:
        st.info("👆 請在左側輸入資訊並點擊「生成 Prompt」按鈕")


# Image Generation Section
st.markdown("---")
st.header("🎨 圖片生成 (Obj 2)")

if st.session_state['generated_prompt'] and design_generator:
    st.markdown("### 設定")

    # Reference Image selector
    available_refs = list(REFERENCE_IMAGES_DIR.glob("lulu_pig_ref_*.png")) + \
                     list(REFERENCE_IMAGES_DIR.glob("lulu_pig_ref_*.jpg"))

    if not available_refs:
        st.warning("⚠️ 未找到 Reference Images，請檢查 data/reference_images/ 目錄")
    else:
        # Display reference images for selection
        ref_names = [ref.name for ref in available_refs]
        selected_ref_name = st.selectbox(
            "選擇 Reference Image",
            options=ref_names,
            help="選擇角色參考圖，用於保持角色一致性"
        )

        selected_ref_path = REFERENCE_IMAGES_DIR / selected_ref_name

        # Show selected reference image
        with st.expander("📷 查看 Reference Image"):
            st.image(str(selected_ref_path), caption=selected_ref_name, width=300)

        # Generation parameters
        with st.expander("⚙️ 生成參數"):
            num_images = st.slider(
                "生成數量",
                min_value=1,
                max_value=4,
                value=4,
                help="選擇要生成的設計圖數量 (1-4 張)"
            )

        # Generate Images button
        generate_images_button = st.button(
            f"🎨 生成 {num_images} 張設計圖",
            type="primary",
            use_container_width=True,
            disabled=(design_generator is None)
        )

        if generate_images_button:
            st.markdown("### 生成中...")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: float, message: str):
                """更新進度條和狀態文字"""
                progress_bar.progress(progress)
                status_text.text(message)

            # Generate designs
            try:
                results = design_generator.generate_designs(
                    prompt=st.session_state['generated_prompt'],
                    reference_image_path=str(selected_ref_path),
                    num_images=num_images,
                    progress_callback=update_progress,
                    max_retries=3
                )

                # Save to session state
                st.session_state['generated_images'] = results

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                # Success summary
                successful_count = sum(1 for r in results if r.get('success'))
                if successful_count == num_images:
                    st.success(f"✅ 成功生成 {successful_count}/{num_images} 張設計圖！")
                elif successful_count > 0:
                    st.warning(f"⚠️ 生成完成：{successful_count}/{num_images} 張成功")
                else:
                    st.error(f"❌ 全部生成失敗，請稍後重試")

            except DesignGenerationError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ 圖片生成失敗：{str(e)}")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ 發生錯誤：{str(e)}")

        # Display generated images
        if st.session_state['generated_images']:
            st.markdown("### 生成結果")

            results = st.session_state['generated_images']
            successful_results = [r for r in results if r.get('success')]

            if successful_results:
                # Calculate average similarity
                avg_similarity = design_generator.get_average_similarity(results)
                st.metric(
                    label="平均 CLIP 相似度",
                    value=f"{avg_similarity:.4f}",
                    delta="✅ 達標" if avg_similarity >= CLIP_SIMILARITY_THRESHOLD else "⚠️ 低於門檻"
                )

                # Display images in 2x2 grid
                cols = st.columns(2)
                for i, result in enumerate(results):
                    col = cols[i % 2]

                    with col:
                        if result.get('success'):
                            # Display image
                            st.image(
                                result['image'],
                                caption=f"變化 {i+1}",
                                use_container_width=True
                            )

                            # CLIP similarity
                            similarity = result.get('clip_similarity', 0.0)
                            if similarity >= CLIP_SIMILARITY_THRESHOLD:
                                st.markdown(f"**CLIP 相似度:** :green[{similarity:.4f}] ✅")
                            else:
                                st.markdown(f"**CLIP 相似度:** :orange[{similarity:.4f}] ⚠️")

                            # Generation time
                            gen_time = result.get('generation_time', 0.0)
                            st.caption(f"生成時間：{gen_time:.2f}s")

                            # Download button
                            img_bytes = design_generator.image_to_bytes(result['image'])
                            st.download_button(
                                label="📥 下載",
                                data=img_bytes,
                                file_name=f"design_{i+1}.png",
                                mime="image/png",
                                key=f"download_{i}"
                            )

                        else:
                            # Display error
                            st.error(f"變化 {i+1} 生成失敗")
                            st.caption(f"錯誤：{result.get('error', '未知錯誤')}")

                        st.markdown("---")

            else:
                st.warning("⚠️ 所有圖片生成失敗，請檢查 API 配置或稍後重試")

elif not st.session_state['generated_prompt']:
    st.info("👆 請先在上方生成 Prompt")
elif not design_generator:
    st.warning("⚠️ Design Generator 未初始化，圖片生成功能不可用")


# Footer
st.markdown("---")
st.markdown("""
### 💡 使用說明

**步驟 1: 生成 Prompt**
1. **輸入角色資訊**：填寫角色名稱和描述
2. **輸入趨勢關鍵字**：填寫與市場趨勢相關的關鍵字（逗號分隔）
3. **生成 Prompt**：點擊按鈕生成 AI 設計 Prompt
4. **複製結果**：使用「複製 Prompt」按鈕保存結果

**步驟 2: 生成設計圖 (可選)**
1. **選擇 Reference Image**：選擇角色參考圖（用於保持一致性）
2. **設定生成數量**：選擇要生成的圖片數量 (1-4 張)
3. **生成設計圖**：點擊「生成設計圖」按鈕
4. **查看結果**：檢查 CLIP 相似度分數（≥ 0.80 為達標）
5. **下載圖片**：使用「下載」按鈕保存圖片

**注意事項：**
- 關鍵字建議 3-10 個為佳
- 描述盡量簡短明確
- 系統會自動重試失敗的請求（最多 3 次）
- 圖片生成需要 GOOGLE_API_KEY（每張約 11 秒）
- CLIP 相似度 ≥ 0.80 表示角色一致性良好
""")
