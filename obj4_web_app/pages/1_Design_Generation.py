# obj4_web_app/pages/1_Design_Generation.py
"""
Streamlit Page 1: Design Generation Wizard

Version 7.0 - Stepper UI Redesign

Features:
- 5-step wizard with horizontal stepper
- Cumulative display (completed steps collapsed)
- Clean minimal design
- All existing functionality preserved

Author: Developer (James)
Date: 2026-01-26
Version: 7.0
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import components
from obj4_web_app.components import (
    inject_custom_css,
    render_stepper,
    render_step_card,
    init_step_state,
    get_current_step,
    get_completed_steps,
    get_step_data,
    get_step_summary,
    go_to_step
)
from obj4_web_app.components.steps import (
    render_step1_content,
    render_step2_content,
    render_step3_content,
    render_step4_content,
    render_step5_content
)

# Import utilities
from obj4_web_app.utils.enhanced_trends_wrapper import EnhancedTrendsWrapper
from obj4_web_app.utils.design_generator import DesignGeneratorWrapper
from obj4_web_app.utils.firebase_manager import FirebaseManager, FirebaseError
from obj4_web_app.config import (
    DEFAULT_REGION,
    DEFAULT_LANG,
    CLIP_SIMILARITY_THRESHOLD,
    REFERENCE_IMAGES_DIR
)

# Import trendspyg config
try:
    from trendspyg.config import COUNTRIES, US_STATES, CATEGORIES
except ImportError:
    COUNTRIES = {'HK': 'Hong Kong', 'US': 'United States', 'JP': 'Japan', 'TW': 'Taiwan'}
    US_STATES = {}
    CATEGORIES = {'all'}

# Detect deployment environment
STREAMLIT_CLOUD_HOME = '/home/appuser'
IS_STREAMLIT_CLOUD = bool(os.getenv('STREAMLIT_RUNTIME_ENV')) or \
                     bool(os.getenv('STREAMLIT_SHARING_MODE')) or \
                     os.getenv('HOME') == STREAMLIT_CLOUD_HOME

# Popular regions
POPULAR_REGIONS = {
    'HK': 'Hong Kong', 'TW': 'Taiwan', 'JP': 'Japan',
    'KR': 'South Korea', 'SG': 'Singapore', 'US': 'United States',
    'GB': 'United Kingdom', 'CA': 'Canada', 'AU': 'Australia',
}

# Step definitions
STEPS = [
    (1, "Character Info", "Character Info"),
    (2, "Trend Keywords", "Trend Keywords"),
    (3, "Generate Prompt", "Generate Prompt"),
    (4, "Generate Images", "Generate Images"),
    (5, "Complete", "Complete")
]


# Page configuration
st.set_page_config(
    page_title="Design Generation - AI Character Design System",
    page_icon="🎨",
    layout="wide"
)

# Inject custom CSS
inject_custom_css()

# Initialize step state
init_step_state()

# Page title
st.title("🎨 Design Generation")
st.caption("Trend-Driven Character Design Wizard")

# Render stepper
current_step = get_current_step()
completed_steps = get_completed_steps()
render_stepper(current_step, completed_steps)

st.markdown("---")


# Sidebar: Settings (preserved from original)
with st.sidebar:
    st.header("⚙️ Settings")

    # Region Selection
    st.subheader("🌍 Region")
    region_options = POPULAR_REGIONS
    selected_region = st.selectbox(
        "Choose Region",
        options=list(region_options.keys()),
        format_func=lambda x: region_options[x],
        index=0
    )

    st.markdown("---")

    # Backend Selection
    st.subheader("📊 Trends Backend")
    backend_options = ["trendspyg", "trendspy", "pytrends"]
    if not IS_STREAMLIT_CLOUD:
        backend_options.insert(1, "trendspyg_csv")

    backend_choice = st.selectbox("Backend", options=backend_options, index=0)

    st.markdown("---")

    # Image API Selection
    st.subheader("🎨 Image API")
    use_openai_api = st.radio(
        "API Type",
        options=[True, False],
        format_func=lambda x: "OpenAI-Compatible" if x else "Official Google",
        index=0
    )


# Initialize API wrappers (cached)
@st.cache_resource
def load_enhanced_trends(backend: str, region: str) -> EnhancedTrendsWrapper | None:
    """Load enhanced trends wrapper with specified backend and region."""
    try:
        return EnhancedTrendsWrapper(region=region, lang=DEFAULT_LANG, backend=backend)
    except Exception as e:
        st.warning(f"⚠️ Enhanced Trends unavailable: {e}")
        return None


@st.cache_resource
def load_design_generator(use_openai: bool) -> DesignGeneratorWrapper | None:
    """Load design generator wrapper."""
    try:
        return DesignGeneratorWrapper(use_openai_api=use_openai)
    except Exception as e:
        st.warning(f"⚠️ Design Generator unavailable: {e}")
        return None


@st.cache_resource
def load_firebase_manager() -> FirebaseManager | None:
    """Load Firebase manager for data persistence."""
    try:
        return FirebaseManager()
    except Exception as e:
        st.warning(f"⚠️ Firebase unavailable: {e}")
        return None


# Load resources
try:
    enhanced_trends = load_enhanced_trends(backend_choice, selected_region)
    design_generator = load_design_generator(use_openai_api)
    firebase_manager = load_firebase_manager()
except Exception as e:
    st.error(f"❌ Initialization failed: {e}")
    st.stop()


# Render step cards
for step_num, title_zh, title_en in STEPS:
    # Determine card state
    if step_num in completed_steps:
        state = 'completed'
    elif step_num == current_step:
        state = 'active'
    else:
        state = 'pending'

    # Get summary for completed steps
    summary = get_step_summary(step_num) if state == 'completed' else None

    # Render card and check if edit button clicked
    should_render = render_step_card(
        step_num=step_num,
        title=title_zh,
        state=state,
        summary=summary,
        on_edit=f"edit_{step_num}" if state == 'completed' else None
    )

    # Handle edit button click
    if state == 'completed' and should_render:
        go_to_step(step_num)
        st.rerun()

    # Render step content if active
    if state == 'active':
        with st.container():
            if step_num == 1:
                render_step1_content()

            elif step_num == 2:
                render_step2_content(
                    enhanced_trends=enhanced_trends,
                    backend_choice=backend_choice,
                    selected_region=selected_region,
                    region_label=region_options[selected_region]
                )

            elif step_num == 3:
                step1_data = get_step_data(1)
                render_step3_content(
                    enhanced_trends=enhanced_trends,
                    character_name=step1_data.get('character_name', ''),
                    character_desc=step1_data.get('character_desc', '')
                )

            elif step_num == 4:
                render_step4_content(
                    design_generator=design_generator,
                    reference_images_dir=REFERENCE_IMAGES_DIR,
                    clip_threshold=CLIP_SIMILARITY_THRESHOLD
                )

            elif step_num == 5:
                render_step5_content(design_generator=design_generator)


# Footer
st.markdown("---")
st.caption("FYP Project - ToyzeroPlus AI Pipeline | Design Generation v7.0")
