"""
Streamlit Page 1: Design Generation

Version 6.0 - trendspyg CSV Integration

Key Features:
- Real-time trending searches (trendspyg RSS backend - ultra-fast)
- Comprehensive trends data (trendspyg CSV backend - advanced filtering)
- No rate limiting issues
- Theme only used for LLM prompt generation
- Support for 4 backends: trendspyg, trendspyg_csv, trendspy, pytrends

Author: Developer (James)
Date: 2025-11-10
Version: 6.0
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from obj4_web_app.utils.enhanced_trends_wrapper import EnhancedTrendsWrapper, EnhancedTrendsError
from obj4_web_app.utils.design_generator import DesignGeneratorWrapper, DesignGenerationError
from obj4_web_app.config import (
    DEFAULT_REGION,
    DEFAULT_LANG,
    CLIP_SIMILARITY_THRESHOLD,
    REFERENCE_IMAGES_DIR
)
import plotly.graph_objects as go

# Import trendspyg config for region and category options
try:
    from trendspyg.config import COUNTRIES, US_STATES, CATEGORIES
except ImportError:
    # Fallback if trendspyg not available
    COUNTRIES = {'HK': 'Hong Kong', 'US': 'United States', 'JP': 'Japan', 'TW': 'Taiwan'}
    US_STATES = {}
    CATEGORIES = {'all', 'sports', 'entertainment', 'technology'}  # Minimal fallback

# Detect deployment environment
IS_STREAMLIT_CLOUD = bool(os.getenv('STREAMLIT_RUNTIME_ENV'))
# Alternative detection methods
if not IS_STREAMLIT_CLOUD:
    IS_STREAMLIT_CLOUD = bool(os.getenv('STREAMLIT_SHARING_MODE'))
if not IS_STREAMLIT_CLOUD:
    # Check if running in cloud environment (Streamlit Cloud has HOME=/home/appuser)
    IS_STREAMLIT_CLOUD = os.getenv('HOME') == '/home/appuser'

# Popular regions for quick selection (Asia + Major countries)
POPULAR_REGIONS = {
    'HK': 'Hong Kong 🇭🇰',
    'TW': 'Taiwan 🇹🇼',
    'JP': 'Japan 🇯🇵',
    'KR': 'South Korea 🇰🇷',
    'SG': 'Singapore 🇸🇬',
    'US': 'United States 🇺🇸',
    'GB': 'United Kingdom 🇬🇧',
    'CA': 'Canada 🇨🇦',
    'AU': 'Australia 🇦🇺',
}

# Page configuration
st.set_page_config(
    page_title="Design Generation - AI Character Design System",
    page_icon="🎨",
    layout="wide"
)

# Page title
st.title("🎨 Design Generation - Trend-Driven Prompt Generation")
st.markdown("**Powered by Enhanced Trends Pipeline (v6.0) - trendspyg RSS + CSV Integration**")
st.markdown("---")


# Initialize session state
if 'generated_prompt' not in st.session_state:
    st.session_state['generated_prompt'] = None

if 'last_keywords' not in st.session_state:
    st.session_state['last_keywords'] = ""

if 'last_character_name' not in st.session_state:
    st.session_state['last_character_name'] = ""

if 'last_theme' not in st.session_state:
    st.session_state['last_theme'] = ""

if 'generated_images' not in st.session_state:
    st.session_state['generated_images'] = []

if 'extracted_trends' not in st.session_state:
    st.session_state['extracted_trends'] = []

if 'selected_keywords' not in st.session_state:
    st.session_state['selected_keywords'] = []

if 'additional_keywords' not in st.session_state:
    st.session_state['additional_keywords'] = ""

if 'final_keywords' not in st.session_state:
    st.session_state['final_keywords'] = []


# Initialize API wrappers (cached)
@st.cache_resource
def load_enhanced_trends(
    backend='trendspyg',
    region='HK',
    proxy=None,
    request_delay=3.0,
    # RSS options
    include_images=True,
    include_articles=True,
    max_articles_per_trend=5,
    # CSV options
    category='all',
    hours=24,
    active_only=False,
    sort_by='relevance',
    headless=True
):
    """
    Load EnhancedTrendsWrapper (cached across sessions).

    Args:
        backend: Trends backend ('trendspyg', 'trendspyg_csv', 'trendspy', or 'pytrends')
        region: Google Trends region code (e.g., 'HK', 'US', 'JP')
        proxy: Optional proxy URL (only for trendspy)
        request_delay: Delay between requests in seconds (only for trendspy)

        # RSS Mode Options (backend='trendspyg'):
        include_images: Include images (trendspyg only)
        include_articles: Include news articles (trendspyg only)
        max_articles_per_trend: Max news articles per trend (trendspyg only)

        # CSV Mode Options (backend='trendspyg_csv'):
        category: Filter by category (default: 'all')
        hours: Time period in hours (4, 24, 48, 168)
        active_only: Only return rising/active trends
        sort_by: Sort order ('relevance', 'title', 'volume', 'recency')
        headless: Run browser in headless mode
    """
    return EnhancedTrendsWrapper(
        region=region,
        lang=DEFAULT_LANG,
        backend=backend,
        proxy=proxy if proxy else None,
        request_delay=request_delay,
        # RSS options
        include_images=include_images,
        include_articles=include_articles,
        max_articles_per_trend=max_articles_per_trend,
        # CSV options
        category=category,
        hours=hours,
        active_only=active_only,
        sort_by=sort_by,
        headless=headless
    )


@st.cache_resource
def load_design_generator(use_openai_api=True):
    """
    Load DesignGeneratorWrapper (cached across sessions).

    Args:
        use_openai_api: Use OpenAI-compatible API (default: True)
    """
    try:
        return DesignGeneratorWrapper(use_openai_api=use_openai_api)
    except Exception as e:
        st.warning(f"⚠️ Design Generator initialization failed: {str(e)}")
        if use_openai_api:
            st.info("Image generation will not be available. Please check GEMINI_OPENAI_API_KEY environment variable.")
        else:
            st.info("Image generation will not be available. Please check GEMINI_API_KEY environment variable.")
        return None


# Sidebar: Backend settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Region Selection
    st.subheader("🌍 Region")

    region_mode = st.radio(
        "Select Region Mode",
        options=["Popular Regions", "All Countries", "US States"],
        index=0,
        help="Choose region for Google Trends data"
    )

    if region_mode == "Popular Regions":
        region_options = POPULAR_REGIONS
        default_index = list(POPULAR_REGIONS.keys()).index('HK') if 'HK' in POPULAR_REGIONS else 0
    elif region_mode == "All Countries":
        region_options = {code: f"{name} ({code})" for code, name in COUNTRIES.items()}
        default_index = list(COUNTRIES.keys()).index('HK') if 'HK' in COUNTRIES else 0
    else:  # US States
        region_options = {code: f"{name} ({code})" for code, name in US_STATES.items()}
        default_index = 0

    selected_region = st.selectbox(
        "Choose Region",
        options=list(region_options.keys()),
        format_func=lambda x: region_options[x],
        index=default_index,
        help=f"Selected: {region_options[list(region_options.keys())[default_index]]}"
    )

    st.caption(f"📍 Current: **{region_options[selected_region]}**")

    st.markdown("---")

    st.subheader("Google Trends Backend")

    # Backend options based on environment
    if IS_STREAMLIT_CLOUD:
        # Streamlit Cloud: Disable CSV mode (requires Chrome browser)
        backend_options = ["trendspyg", "trendspy", "pytrends"]
        st.caption("☁️ **Streamlit Cloud Mode**: CSV mode disabled (requires browser automation)")
    else:
        # Local development: All backends available
        backend_options = ["trendspyg", "trendspyg_csv", "trendspy", "pytrends"]

    backend_choice = st.selectbox(
        "Select Backend",
        options=backend_options,
        index=0,
        help="**trendspyg** (RSS): Ultra-fast (0.2-0.5s), 10-20 trends, no filtering\n\n**trendspyg_csv** (CSV): Slower (~10s), 480 trends, category/time filtering (Local only)\n\n**trendspy**: Better rate limiting than pytrends\n\n**pytrends**: Original backend (archived, may have issues)"
    )

    proxy_url = None
    request_delay = 3.0  # Default delay

    # trendspyg RSS specific config
    include_images = True
    include_articles = True
    max_articles_per_trend = 5

    # trendspyg CSV specific config
    category = 'all'
    hours = 24
    active_only = False
    sort_by = 'relevance'
    headless = True

    if backend_choice == "trendspyg":
        st.caption("⚡ trendspyg (RSS): Ultra-fast, no rate limiting")
        st.success(f"✅ Real-time trending searches from RSS feed\n\n📍 Region: **{region_options[selected_region]}**")

        # trendspyg RSS configuration options
        with st.expander("⚙️ Advanced Settings (Optional)"):
            st.caption("Configure trendspyg RSS feed options")

            include_images = st.checkbox(
                "Include Images",
                value=True,
                help="Include trend thumbnail images and news article images"
            )

            include_articles = st.checkbox(
                "Include News Articles",
                value=True,
                help="Include related news articles for each trend"
            )

            if include_articles:
                max_articles_per_trend = st.slider(
                    "Max Articles per Trend",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum number of news articles to include for each trend"
                )

            # Show current config
            st.caption(f"**Current Config:** Images={include_images}, Articles={include_articles}" +
                      (f" (max {max_articles_per_trend})" if include_articles else ""))

        if not include_images and not include_articles:
            st.info("💡 Minimal mode: Only trend keywords and traffic data")
        elif include_images and include_articles:
            st.info(f"💡 Full mode: Keywords + Images + Articles (max {max_articles_per_trend} per trend)")

    elif backend_choice == "trendspyg_csv":
        st.caption("📊 trendspyg (CSV): Comprehensive data with filtering")
        st.success(f"✅ ~480 trends with category/time filtering\n\n📍 Region: **{region_options[selected_region]}**")
        st.warning("⏱️ Slower (~10 seconds) - requires Chrome browser")

        # trendspyg CSV configuration options
        with st.expander("⚙️ CSV Mode Settings", expanded=True):
            st.caption("Configure comprehensive trends extraction")

            # Category filter
            category_options = sorted(list(CATEGORIES))
            category = st.selectbox(
                "Category Filter",
                options=category_options,
                index=category_options.index('all') if 'all' in category_options else 0,
                help="Filter trends by specific category (20 categories available)"
            )

            # Time period
            time_options = {
                4: 'Past 4 hours',
                24: 'Past 24 hours',
                48: 'Past 48 hours',
                168: 'Past 7 days'
            }
            hours_display = st.select_slider(
                "Time Period",
                options=list(time_options.keys()),
                value=24,
                format_func=lambda x: time_options[x],
                help="Time period for trends (4h, 24h, 48h, 7d)"
            )
            hours = hours_display

            # Active trends only
            active_only = st.checkbox(
                "Active Trends Only",
                value=False,
                help="Only show rising/active trends (filters out declining trends)"
            )

            # Sort order
            sort_options = ['relevance', 'title', 'volume', 'recency']
            sort_by = st.selectbox(
                "Sort By",
                options=sort_options,
                index=0,
                help="Sort order for trends results"
            )

            # Show current config
            st.caption(f"**Current Config:** Category={category}, Time={time_options[hours]}, Active={active_only}, Sort={sort_by}")

        st.info(f"💡 CSV mode: ~480 trends filtered by {category} category from {time_options[hours].lower()}")

    elif backend_choice == "trendspy":
        st.caption("🚀 TrendsPy: Better rate limiting than pytrends")

        # Request delay slider
        request_delay = st.slider(
            "Request Delay (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Delay between API requests. Increase to 5.0+ if encountering rate limiting (429 errors)"
        )

        if request_delay >= 5.0:
            st.success(f"✅ Using slower delay ({request_delay}s) - better for avoiding rate limits")
        elif request_delay < 3.0:
            st.warning(f"⚠️ Low delay ({request_delay}s) - may trigger rate limiting")

        with st.expander("🌐 Proxy Settings (Optional)"):
            st.caption("Add proxy to avoid rate limiting")
            enable_proxy = st.checkbox("Enable Proxy")
            if enable_proxy:
                proxy_url = st.text_input(
                    "Proxy URL",
                    placeholder="http://10.10.1.10:3128",
                    help="Enter proxy URL (e.g., http://10.10.1.10:3128)"
                )
    else:
        st.caption("⚠️ PyTrends: Original backend (archived 2025-04-17)")
        st.warning("May encounter rate limiting issues")

    st.markdown("---")

    st.subheader("🎨 Image Generation API")

    image_api_choice = st.radio(
        "Select Image Generation API",
        options=["OpenAI-Compatible API", "Official Google API"],
        index=0,
        help="**OpenAI-Compatible API**: Uses GEMINI_OPENAI_API_KEY (更快速，支援圖生圖)\n\n**Official Google API**: Uses GEMINI_API_KEY (官方 SDK)"
    )

    use_openai_api = (image_api_choice == "OpenAI-Compatible API")

    if use_openai_api:
        st.caption("✅ Using OpenAI-Compatible API (GEMINI_OPENAI_API_KEY)")
    else:
        st.caption("✅ Using Official Google API (GEMINI_API_KEY)")

    st.markdown("---")


try:
    enhanced_trends = load_enhanced_trends(
        backend=backend_choice,
        region=selected_region,
        proxy=proxy_url,
        request_delay=request_delay,
        # RSS options
        include_images=include_images,
        include_articles=include_articles,
        max_articles_per_trend=max_articles_per_trend,
        # CSV options
        category=category,
        hours=hours,
        active_only=active_only,
        sort_by=sort_by,
        headless=headless
    )
    design_generator = load_design_generator(use_openai_api=use_openai_api)
except Exception as e:
    st.error(f"❌ System initialization failed: {str(e)}")
    st.stop()


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Input Information")

    # Character information
    st.subheader("1️⃣ Character Information")
    character_name = st.text_input(
        "Character Name",
        value="Lulu Pig",
        help="Enter character name, e.g., Lulu Pig"
    )

    character_desc = st.text_area(
        "Character Description",
        value="Cute pink pig with big eyes and chubby body",
        help="Briefly describe character features"
    )

    # Trend keywords extraction
    st.markdown("---")
    st.subheader("2️⃣ Extract Trend Keywords")

    st.info("💡 **New Approach:** Keywords are extracted based on time period only, using social media trends. No theme seed keywords required!")

    # Method selector: Auto Extract or Manual Input
    extraction_method = st.radio(
        "Extraction Method",
        options=["🔍 Auto Extract (Google Trends)", "✍️ Manual Input"],
        horizontal=True,
        help="Auto Extract: Extract trending keywords from Google Trends, then generate prompt\nManual Input: Enter complete image generation prompt directly (skip to Image Generation)"
    )

    if extraction_method == "✍️ Manual Input":
        # Manual input section
        st.markdown("**Manual Prompt Input**")
        st.caption("Enter complete image generation prompt directly. Skip keyword extraction and go straight to Image Generation.")

        manual_prompt_input = st.text_area(
            "Image Generation Prompt",
            placeholder="e.g., Lulu Pig wearing a cozy Christmas sweater, sitting by a fireplace with hot cocoa, warm lighting, festive decorations, cute and heartwarming scene",
            help="Enter the complete prompt for image generation",
            height=150
        )

        if st.button("✅ Use Manual Prompt", use_container_width=True, type="primary"):
            if manual_prompt_input.strip():
                # Directly set as generated prompt (skip keyword extraction and prompt generation)
                st.session_state['generated_prompt'] = manual_prompt_input.strip()
                st.session_state['last_keywords'] = "Manual Input"
                st.session_state['last_character_name'] = character_name
                st.session_state['last_theme'] = "Manual Prompt"

                # Clear keyword-related states (not needed for manual prompt)
                st.session_state['extracted_trends'] = []
                st.session_state['selected_keywords'] = []
                st.session_state['additional_keywords'] = ""
                st.session_state['final_keywords'] = []

                st.success(f"✅ Manual prompt loaded! You can now proceed to Image Generation (Step 4).")
                st.rerun()
            else:
                st.error("❌ Please enter a prompt")

    else:
        # Auto extraction section
        # Timeframe selector (only for pytrends/trendspy backends)
        # trendspyg/trendspyg_csv have their own time handling
        if backend_choice not in ['trendspyg', 'trendspyg_csv']:
            st.markdown("**Select Time Period**")

            timeframe_options = {
                "Past 3 months": "today 3-m",
                "Past 6 months": "today 6-m",
                "Past 12 months": "today 12-m",
                "Past 5 years": "today 5-y",
                "Custom Date Range": "custom"
            }

            selected_timeframe_label = st.selectbox(
                "Time Period",
                options=list(timeframe_options.keys()),
                index=2,  # Default to "Past 12 months"
                help="Select the time period for trend analysis"
            )

            # Custom date range if selected
            if selected_timeframe_label == "Custom Date Range":
                col_start, col_end = st.columns(2)
                with col_start:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=365),
                        max_value=datetime.now()
                    )
                with col_end:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now(),
                        max_value=datetime.now()
                    )

                # Format for Google Trends
                selected_timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
                st.caption(f"Selected: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            else:
                selected_timeframe = timeframe_options[selected_timeframe_label]

            # Save to session state for display
            st.session_state['last_timeframe_label'] = selected_timeframe_label

        else:
            # trendspyg/trendspyg_csv: use default timeframe (ignored by backend)
            selected_timeframe = "now"  # Placeholder, ignored by trendspyg
            if backend_choice == 'trendspyg':
                selected_timeframe_label = "Real-time (RSS)"
            elif backend_choice == 'trendspyg_csv':
                # Show CSV mode config
                time_label_map = {4: '4h', 24: '24h', 48: '48h', 168: '7d'}
                selected_timeframe_label = f"{category.capitalize()} - Past {time_label_map.get(hours, f'{hours}h')}"
            else:
                selected_timeframe_label = "Current"

            # Save to session state
            st.session_state['last_timeframe_label'] = selected_timeframe_label

            # Info message for trendspyg
            if backend_choice == 'trendspyg':
                st.info(f"💡 trendspyg (RSS): Extracting real-time trending searches (~10-20 trends)\n"
                       f"📍 Region: {region_options[selected_region]}")
            elif backend_choice == 'trendspyg_csv':
                st.info(f"💡 trendspyg (CSV): Extracting comprehensive trends (~480 trends)\n"
                       f"📍 Region: {region_options[selected_region]}\n"
                       f"Category: {category}, Time: {time_label_map.get(hours, f'{hours}h')}, Active: {active_only}")

        # Optional theme filter
        with st.expander("🔍 Advanced: Theme Filter (Optional)"):
            st.markdown("""
            **Optional**: Filter extracted trends by theme keywords.
            - Leave empty to get ALL trending topics in this period
            - Enter theme keywords (regex) to filter results
            - Example: `christmas|xmas` for Christmas-related trends
            """)

            theme_filter_input = st.text_input(
                "Theme Filter (regex)",
                value="",
                placeholder="e.g., christmas|xmas (leave empty for all trends)",
                help="Optional regex pattern to filter trends"
            )

        # Extract settings
        col_extract_btn, col_top_n = st.columns([3, 1])

        with col_top_n:
            top_n = st.number_input(
                "Top N",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Number of keywords to extract"
            )

        with col_extract_btn:
            extract_button = st.button(
                "🔍 Extract Keywords",
                use_container_width=True,
                type="secondary"
            )

        # Extract trends
        if extract_button:
            filter_text = theme_filter_input.strip() if theme_filter_input else None
            display_filter = f" (filtered by: {filter_text})" if filter_text else " (all trends)"

            with st.spinner(f"⏳ Extracting trends for {selected_timeframe_label}{display_filter}..."):
                try:
                    keywords = enhanced_trends.extract_trends(
                        timeframe=selected_timeframe,
                        top_n=top_n,
                        theme_filter=filter_text
                    )

                    if keywords:
                        st.session_state['extracted_trends'] = keywords
                        st.session_state['selected_keywords'] = []  # Reset selection
                        st.session_state['additional_keywords'] = ""  # Reset additional
                        st.session_state['final_keywords'] = []  # Reset final
                        st.success(f"✅ Extracted {len(keywords)} optimized keywords!")
                    else:
                        st.warning("⚠️ No trend data found, try another timeframe or remove theme filter")

                except EnhancedTrendsError as e:
                    error_msg = str(e)

                    # Check if it's rate limiting error
                    if '429' in error_msg or 'Rate Limiting' in error_msg:
                        st.error(f"🚨 **Google Trends Rate Limit Exceeded** (Backend: {backend_choice})")
                        st.warning(error_msg)

                        # Show manual input suggestion prominently
                        st.info("""
                        💡 **Quick Fix:**
                        1. Switch to **'✍️ Manual Input'** tab above
                        2. Enter keywords manually (comma-separated)
                        3. Continue with your workflow immediately!
                        """)

                        # Backend-specific suggestions
                        if backend_choice == "pytrends":
                            st.info("""
                            🚀 **Alternative Solution:**
                            - Switch to **TrendsPy** backend in sidebar (⚙️ Settings)
                            - TrendsPy has better rate limiting handling
                            - Optionally add a proxy for even better reliability
                            """)
                        else:
                            st.info("""
                            ⏰ **Or wait 2-3 minutes and try Auto Extract again**

                            🌐 **For better reliability:** Add a proxy in sidebar (⚙️ Settings → Proxy Settings)
                            """)
                    else:
                        st.error(f"❌ {error_msg}")

                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("💡 **Tip:** Try using 'Manual Input' to continue!")

with col2:
    st.header("✨ Keyword Selection")

    # Display extracted trends with checkboxes
    if st.session_state['extracted_trends']:
        # Use last timeframe label from session state (safe fallback)
        timeframe_display = st.session_state.get('last_timeframe_label', 'Unknown')
        st.markdown(f"**Extraction Results ({timeframe_display}):**")
        st.caption("Keywords sorted by combined score (trend + visual relevance)")

        # Select all / deselect all buttons
        col_select_all, col_deselect_all = st.columns(2)
        with col_select_all:
            if st.button("✅ Select All", use_container_width=True, key="select_all"):
                st.session_state['selected_keywords'] = [
                    kw['keyword'] for kw in st.session_state['extracted_trends']
                ]
                st.rerun()

        with col_deselect_all:
            if st.button("❌ Deselect All", use_container_width=True, key="deselect_all"):
                st.session_state['selected_keywords'] = []
                st.rerun()

        # Scrollable keyword list
        with st.container(height=300):
            for kw_data in st.session_state['extracted_trends']:
                keyword = kw_data['keyword']
                combined_score = kw_data['combined_score']
                visual_score = kw_data['visual_score']
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
                    st.caption(f"{combined_score:.1f}")

        # Display selected count
        if st.session_state['selected_keywords']:
            st.markdown("---")
            st.success(f"**Selected: {len(st.session_state['selected_keywords'])} keywords**")

            # Show selected keywords
            with st.expander("View Selected Keywords"):
                st.write(", ".join(st.session_state['selected_keywords']))

        # Trend visualization
        if st.session_state['selected_keywords']:
            with st.expander("📊 Score Visualization"):
                # Filter to show only selected
                keywords_to_show = [kw for kw in st.session_state['extracted_trends']
                                  if kw['keyword'] in st.session_state['selected_keywords']]

                if keywords_to_show:
                    keywords = [kw['keyword'] for kw in keywords_to_show]
                    combined_scores = [kw['combined_score'] for kw in keywords_to_show]
                    visual_scores = [kw['visual_score'] for kw in keywords_to_show]

                    fig = go.Figure()

                    # Combined score
                    fig.add_trace(go.Bar(
                        x=keywords,
                        y=combined_scores,
                        name='Combined Score',
                        marker_color='#1f77b4',
                        text=[f"{score:.1f}" for score in combined_scores],
                        textposition='auto',
                    ))

                    # Visual score
                    fig.add_trace(go.Bar(
                        x=keywords,
                        y=visual_scores,
                        name='Visual Score',
                        marker_color='#ff7f0e',
                        text=[f"{score:.1f}" for score in visual_scores],
                        textposition='auto',
                    ))

                    fig.update_layout(
                        title=f"Selected Keywords Scores",
                        xaxis_title="Keywords",
                        yaxis_title="Score",
                        height=300,
                        xaxis={'tickangle': -45},
                        barmode='group'
                    )

                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Extract keywords from Google Trends to begin")


# Keyword Adjustment Section
if st.session_state['selected_keywords']:
    st.markdown("---")
    st.subheader("3️⃣ Adjust Keywords (Optional)")

    st.markdown("**Would you like to modify or add keywords?**")

    col_yes, col_no = st.columns(2)

    with col_yes:
        show_adjustment = st.button("✏️ Yes, Adjust Keywords", use_container_width=True)

    with col_no:
        skip_adjustment = st.button("✅ No, Use Selected Keywords", use_container_width=True, type="primary")

    if show_adjustment or st.session_state.get('show_adjustment_area', False):
        st.session_state['show_adjustment_area'] = True

        st.markdown("---")

        # Section 1: Modify Selected Keywords
        st.markdown("**1️⃣ Review & Modify Selected Keywords**")
        st.caption("Remove unwanted keywords by deselecting them")

        # Initialize adjusted_selected_keywords if not exists
        if 'adjusted_selected_keywords' not in st.session_state:
            st.session_state['adjusted_selected_keywords'] = st.session_state['selected_keywords'].copy()

        # Show multiselect for modifying selected keywords
        adjusted_selected = st.multiselect(
            "Selected Keywords (you can remove by deselecting)",
            options=st.session_state['selected_keywords'],
            default=st.session_state.get('adjusted_selected_keywords', st.session_state['selected_keywords']),
            help="Deselect keywords you don't want to use"
        )

        st.session_state['adjusted_selected_keywords'] = adjusted_selected

        # Section 2: Add Additional Keywords
        st.markdown("---")
        st.markdown("**2️⃣ Add Additional Keywords**")
        st.caption("Add extra keywords not in the extracted list")

        additional_input = st.text_area(
            "Additional Keywords (comma separated)",
            value=st.session_state.get('additional_keywords', ""),
            placeholder="e.g., cozy, warm, festive",
            help="Add extra keywords that are not in the extracted list"
        )

        if st.button("💾 Save Adjustments"):
            st.session_state['additional_keywords'] = additional_input

            # Combine adjusted selected + additional
            additional_list = [kw.strip() for kw in additional_input.split(',') if kw.strip()]
            st.session_state['final_keywords'] = adjusted_selected + additional_list

            st.success(f"✅ Keywords updated! Total: {len(st.session_state['final_keywords'])} keywords")
            st.session_state['show_adjustment_area'] = False
            st.rerun()

    if skip_adjustment:
        # Use selected keywords as final
        st.session_state['final_keywords'] = st.session_state['selected_keywords'].copy()
        st.session_state['show_adjustment_area'] = False
        # Reset adjusted keywords
        if 'adjusted_selected_keywords' in st.session_state:
            del st.session_state['adjusted_selected_keywords']
        st.success(f"✅ Using {len(st.session_state['final_keywords'])} selected keywords")


# Generate Prompt Section
if st.session_state.get('final_keywords'):
    st.markdown("---")
    st.subheader("4️⃣ Generate Prompt")

    st.markdown("**Final Keywords:**")
    st.info(", ".join(st.session_state['final_keywords']))

    # Theme input (only for prompt generation)
    st.markdown("---")
    st.markdown("**Theme Context (for Prompt Generation)**")
    st.caption("⚠️ Theme is only used to provide context for LLM prompt generation, NOT for trend extraction")

    # Holiday/Theme options
    THEME_OPTIONS = {
        "🎄 Christmas": "Cozy Christmas",
        "🎃 Halloween": "Spooky Halloween",
        "🧧 Chinese New Year": "Chinese New Year / Spring Festival",
        "💝 Valentine's Day": "Valentine's Day Romance",
        "🐰 Easter": "Easter Spring",
        "🦃 Thanksgiving": "Thanksgiving Harvest",
        "🎆 New Year": "New Year Celebration",
        "🌸 Spring": "Spring Blossom",
        "☀️ Summer": "Summer Beach",
        "🍂 Autumn": "Autumn Harvest",
        "❄️ Winter": "Winter Wonderland",
        "🎂 Birthday": "Birthday Party",
        "🎓 Graduation": "Graduation Ceremony",
        "👶 Baby Shower": "Baby Shower Celebration",
        "💍 Wedding": "Wedding Celebration",
        "🎮 Gaming": "Gaming Culture",
        "⚽ Sports": "Sports & Fitness",
        "🎵 Music": "Music Festival",
        "🎬 Movies": "Movie Theme",
        "🍕 Food": "Food & Cuisine",
        "✏️ Custom": "Custom Theme"
    }

    selected_theme_key = st.selectbox(
        "Select Theme/Context",
        options=list(THEME_OPTIONS.keys()),
        index=0,  # Default to Christmas
        help="Select a theme to guide the AI prompt generation"
    )

    # If custom selected, show text input
    if selected_theme_key == "✏️ Custom":
        theme_for_prompt = st.text_input(
            "Enter Custom Theme",
            value="",
            placeholder="e.g., Cyberpunk Future, Vintage 80s, Medieval Fantasy",
            help="Enter your custom theme/context"
        )
        if not theme_for_prompt.strip():
            st.warning("⚠️ Please enter a custom theme")
    else:
        theme_for_prompt = THEME_OPTIONS[selected_theme_key]
        st.caption(f"Selected: **{theme_for_prompt}**")

    generate_button = st.button(
        "🚀 Generate AI Prompt",
        type="primary",
        use_container_width=True
    )

    if generate_button:
        # Validation
        if not character_name.strip():
            st.error("❌ Please enter character name")
        elif not character_desc.strip():
            st.error("❌ Please enter character description")
        elif selected_theme_key == "✏️ Custom" and not theme_for_prompt.strip():
            st.error("❌ Please enter custom theme or select a preset theme")
        elif not theme_for_prompt or not theme_for_prompt.strip():
            st.error("❌ Please select or enter a theme for prompt generation")
        else:
            # Generate prompt with progress bar
            with st.spinner("⏳ Generating prompt..."):
                try:
                    generated_prompt = enhanced_trends.generate_prompt_with_theme(
                        keywords=st.session_state['final_keywords'],
                        theme=theme_for_prompt,
                        character_name=character_name,
                        character_desc=character_desc
                    )

                    # Save to session state
                    st.session_state['generated_prompt'] = generated_prompt
                    st.session_state['last_keywords'] = ", ".join(st.session_state['final_keywords'])
                    st.session_state['last_character_name'] = character_name
                    st.session_state['last_theme'] = theme_for_prompt

                    # Success message
                    st.success("✅ Prompt generated successfully!")
                    st.rerun()

                except EnhancedTrendsError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error occurred: {str(e)}")

# Display generated prompt
if st.session_state['generated_prompt']:
    st.markdown("---")
    st.subheader("📝 Generated Prompt")

    # Display in read-only text area (multi-line)
    st.text_area(
        label="Generated Midjourney Prompt",
        value=st.session_state['generated_prompt'],
        height=200,
        disabled=True,
        label_visibility="collapsed"
    )

    # Copy button
    st.download_button(
        label="📋 Download Prompt",
        data=st.session_state['generated_prompt'],
        file_name=f"prompt_{st.session_state['last_character_name'].replace(' ', '_')}.txt",
        mime="text/plain"
    )

    # Display metadata
    st.caption(
        f"Character: {st.session_state['last_character_name']} | "
        f"Theme: {st.session_state['last_theme']} | "
        f"Keywords: {st.session_state['last_keywords']}"
    )


# Image Generation Section
st.markdown("---")
st.header("🎨 Image Generation (Obj 2)")

if st.session_state['generated_prompt'] and design_generator:
    st.markdown("### Settings")

    # Reference Image selector
    available_refs = list(REFERENCE_IMAGES_DIR.glob("lulu_pig_ref_*.png")) + \
                     list(REFERENCE_IMAGES_DIR.glob("lulu_pig_ref_*.jpg"))

    if not available_refs:
        st.warning("⚠️ Reference images not found, please check data/reference_images/ directory")
    else:
        # Display reference images for selection
        ref_names = [ref.name for ref in available_refs]
        selected_ref_name = st.selectbox(
            "Select Reference Image",
            options=ref_names,
            help="Choose character reference image to maintain character consistency"
        )

        selected_ref_path = REFERENCE_IMAGES_DIR / selected_ref_name

        # Show selected reference image
        with st.expander("📷 View Reference Image"):
            st.image(str(selected_ref_path), caption=selected_ref_name, width=300)

        # Generation parameters
        with st.expander("⚙️ Generation Parameters"):
            num_images = st.slider(
                "Number of Images",
                min_value=1,
                max_value=4,
                value=4,
                help="Select number of design images to generate (1-4)"
            )

        # Generate Images button
        generate_images_button = st.button(
            f"🎨 Generate {num_images} Design Images",
            type="primary",
            use_container_width=True,
            disabled=(design_generator is None)
        )

        if generate_images_button:
            st.markdown("### Generating...")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: float, message: str):
                """Update progress bar and status text"""
                progress_bar.progress(progress)
                status_text.text(message)

            # Generate designs
            try:
                results = design_generator.generate_designs(
                    prompt=st.session_state['generated_prompt'],
                    reference_image_path=str(selected_ref_path),
                    num_images=num_images,
                    progress_callback=update_progress,
                    max_retries=3,
                    use_multithreading=True  # Enable parallel generation for faster performance
                )

                # Save to session state
                st.session_state['generated_images'] = results

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                # Success summary
                successful_count = sum(1 for r in results if r.get('success'))
                if successful_count == num_images:
                    st.success(f"✅ Successfully generated {successful_count}/{num_images} design images!")
                elif successful_count > 0:
                    st.warning(f"⚠️ Generation complete: {successful_count}/{num_images} successful")
                else:
                    st.error(f"❌ All generation failed, please retry later")

            except DesignGenerationError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Image generation failed: {str(e)}")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error occurred: {str(e)}")

        # Display generated images
        if st.session_state['generated_images']:
            st.markdown("### Generation Results")

            results = st.session_state['generated_images']
            successful_results = [r for r in results if r.get('success')]

            if successful_results:
                # Calculate average similarity
                avg_similarity = design_generator.get_average_similarity(results)
                st.metric(
                    label="Average CLIP Similarity",
                    value=f"{avg_similarity:.4f}",
                    delta="✅ Pass" if avg_similarity >= CLIP_SIMILARITY_THRESHOLD else "⚠️ Below Threshold"
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
                                caption=f"Variation {i+1}",
                                use_container_width=True
                            )

                            # CLIP similarity
                            similarity = result.get('clip_similarity', 0.0)
                            if similarity >= CLIP_SIMILARITY_THRESHOLD:
                                st.markdown(f"**CLIP Similarity:** :green[{similarity:.4f}] ✅")
                            else:
                                st.markdown(f"**CLIP Similarity:** :orange[{similarity:.4f}] ⚠️")

                            # Generation time
                            gen_time = result.get('generation_time', 0.0)
                            st.caption(f"Generation time: {gen_time:.2f}s")

                            # Download button
                            img_bytes = design_generator.image_to_bytes(result['image'])
                            st.download_button(
                                label="📥 Download",
                                data=img_bytes,
                                file_name=f"design_{i+1}.png",
                                mime="image/png",
                                key=f"download_{i}"
                            )

                        else:
                            # Display error
                            st.error(f"Variation {i+1} generation failed")
                            st.caption(f"Error: {result.get('error', 'Unknown error')}")

                        st.markdown("---")

            else:
                st.warning("⚠️ All image generation failed, please check API configuration or retry later")

elif not st.session_state['generated_prompt']:
    st.info("👆 Please generate a prompt first")
elif not design_generator:
    st.warning("⚠️ Design Generator not initialized, image generation unavailable")


# Footer
st.markdown("---")
st.markdown("""
### 💡 Enhanced Workflow Guide (v3.0)

**✨ What's New:**
- **Timeframe-Driven Extraction**: Keywords are discovered based purely on time period trends
- **Social Media Trends**: Uses broad seed keywords from social media to auto-discover popular trends
- **Theme = Context Only**: Theme is now only used for LLM prompt generation, NOT trend extraction

**Step 1: Character Information**
- Enter character name and description

**Step 2: Extract Trend Keywords**
- Select time period (e.g., Past 3 months, Custom)
- Optional: Add theme filter to narrow results (e.g., "christmas")
- System extracts trending keywords from social media and entertainment categories
- Keywords are optimized with visual relevance scoring

**Step 3: Select Keywords**
- Choose relevant keywords from extracted results
- View combined scores (trend + visual relevance)
- Visualization shows both trend and visual scores

**Step 4: Adjust Keywords (Optional)**
- Add additional custom keywords if needed
- Or proceed with selected keywords

**Step 5: Generate Prompt**
- Enter theme/context for prompt generation (e.g., "Cozy Christmas")
- Review final keywords
- Generate AI design prompt with theme context

**Step 6: Generate Images (Optional)**
- Select reference image
- Generate design variations with CLIP validation

**Key Differences from v2.0:**
- ✅ No more theme-based seed keywords
- ✅ Trends discovered automatically from timeframe
- ✅ Better keyword quality with visual scoring
- ✅ Theme only affects prompt, not trend extraction
- ✅ More flexible and adaptable to any time period

**Notes:**
- CLIP similarity ≥ 0.80 indicates good character consistency
- Combined score = 70% visual relevance + 30% trend strength
- All steps are flexible - you can go back and adjust anytime
""")
