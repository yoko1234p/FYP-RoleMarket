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
from obj4_web_app.config import (
    DEFAULT_REGION,
    DEFAULT_LANG,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES
)

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


# Initialize API wrapper (cached)
@st.cache_resource
def load_trends_api():
    """
    載入 TrendsAPIWrapper（cached across sessions）。

    Returns:
        TrendsAPIWrapper instance
    """
    return TrendsAPIWrapper(region=DEFAULT_REGION, lang=DEFAULT_LANG)


try:
    api_wrapper = load_trends_api()
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
    keywords_input = st.text_input(
        "關鍵字（逗號分隔）",
        value="春節, 紅色, 喜慶, 燈籠",
        help="輸入趨勢關鍵字，用逗號分隔，例如：春節, 紅色, 喜慶"
    )

    st.info("💡 提示：輸入與市場趨勢相關的關鍵字，系統會自動生成設計 Prompt")

    # Generate button
    generate_button = st.button(
        "🚀 生成 Prompt",
        type="primary",
        use_container_width=True
    )

with col2:
    st.header("✨ 生成結果")

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


# Footer
st.markdown("---")
st.markdown("""
### 💡 使用說明
1. **輸入角色資訊**：填寫角色名稱和描述
2. **輸入趨勢關鍵字**：填寫與市場趨勢相關的關鍵字（逗號分隔）
3. **生成 Prompt**：點擊按鈕生成 AI 設計 Prompt
4. **複製結果**：使用「複製 Prompt」按鈕保存結果

**注意事項：**
- 關鍵字建議 3-10 個為佳
- 描述盡量簡短明確
- 系統會自動重試失敗的請求（最多 3 次）
""")
