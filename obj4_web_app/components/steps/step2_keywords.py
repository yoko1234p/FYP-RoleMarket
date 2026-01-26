# obj4_web_app/components/steps/step2_keywords.py
"""
Step 2: Trend Keywords (Extract + Select + Adjust combined)
"""

import streamlit as st
from ..state_manager import get_step_data, update_step_data, go_to_next_step, go_to_previous_step


def render_step2_content(enhanced_trends, backend_choice: str, selected_region: str, region_label: str) -> None:
    """
    Render Step 2: Trend Keywords form.

    Args:
        enhanced_trends: EnhancedTrendsWrapper instance
        backend_choice: Current backend (trendspyg, trendspy, etc.)
        selected_region: Region code (e.g., 'HK')
        region_label: Region display name (e.g., 'Hong Kong')
    """
    data = get_step_data(2)

    # Tab toggle: Auto Extract / Manual Input
    method = st.radio(
        "Extraction Method",
        options=["auto", "manual"],
        format_func=lambda x: "Auto Extract (Google Trends)" if x == "auto" else "Manual Input",
        horizontal=True,
        index=0 if data.get('method', 'auto') == 'auto' else 1,
        key="step2_method"
    )

    update_step_data(2, {'method': method})

    st.markdown("---")

    if method == "auto":
        _render_auto_extract(enhanced_trends, backend_choice, selected_region, region_label, data)
    else:
        _render_manual_input(data)

    # Show selected keywords
    all_keywords = data.get('selected_keywords', []) + data.get('custom_keywords', [])

    if all_keywords:
        st.markdown("---")
        st.success(f"**Selected: {len(all_keywords)} keywords**")
        st.info(", ".join(all_keywords))

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("<- Back", key="step2_back", use_container_width=True):
            go_to_previous_step()
            st.rerun()

    with col3:
        is_valid = len(all_keywords) > 0
        if st.button("Next: 生成 Prompt ->", key="step2_next", type="primary",
                    use_container_width=True, disabled=not is_valid):
            go_to_next_step()
            st.rerun()

    if not all_keywords:
        st.warning("Please select or enter at least one keyword.")


def _render_auto_extract(enhanced_trends, backend_choice, selected_region, region_label, data):
    """Render auto-extract UI."""
    from obj4_web_app.utils.enhanced_trends_wrapper import EnhancedTrendsError

    st.caption(f"Region: **{region_label}** | Backend: **{backend_choice}**")

    col1, col2 = st.columns([3, 1])
    with col2:
        top_n = st.number_input("Top N", min_value=5, max_value=30, value=15, key="step2_top_n")

    with col1:
        if st.button("Extract Keywords", key="step2_extract", use_container_width=True):
            with st.spinner("Extracting trends..."):
                try:
                    keywords = enhanced_trends.extract_trends(timeframe="now", top_n=top_n)
                    if keywords:
                        update_step_data(2, {'extracted_trends': keywords, 'selected_keywords': []})
                        st.success(f"Extracted {len(keywords)} keywords!")
                        st.rerun()
                    else:
                        st.warning("No trends found. Try manual input.")
                except EnhancedTrendsError as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Switch to Manual Input tab to continue.")

    # Show extracted keywords for selection
    extracted = data.get('extracted_trends', [])
    if extracted:
        st.markdown("**Select Keywords:**")

        selected = data.get('selected_keywords', [])

        # Select all / Deselect all
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", key="step2_select_all", use_container_width=True):
                update_step_data(2, {'selected_keywords': [kw['keyword'] for kw in extracted]})
                st.rerun()
        with col_b:
            if st.button("Deselect All", key="step2_deselect_all", use_container_width=True):
                update_step_data(2, {'selected_keywords': []})
                st.rerun()

        # Keyword checkboxes
        new_selected = []
        for kw_data in extracted:
            keyword = kw_data['keyword']
            score = kw_data.get('combined_score', 0)
            is_checked = keyword in selected

            if st.checkbox(f"{keyword} ({score:.1f})", value=is_checked, key=f"kw_{keyword}"):
                new_selected.append(keyword)

        if new_selected != selected:
            update_step_data(2, {'selected_keywords': new_selected})

    # Add custom keywords
    st.markdown("---")
    st.markdown("**Add Custom Keywords:**")
    col1, col2 = st.columns([4, 1])
    with col1:
        custom_input = st.text_input("", placeholder="e.g., cozy, festive", key="step2_custom_input", label_visibility="collapsed")
    with col2:
        if st.button("Add", key="step2_add_custom", use_container_width=True):
            if custom_input.strip():
                current_custom = data.get('custom_keywords', [])
                new_keywords = [kw.strip() for kw in custom_input.split(',') if kw.strip()]
                current_custom.extend(new_keywords)
                update_step_data(2, {'custom_keywords': list(set(current_custom))})
                st.rerun()


def _render_manual_input(data):
    """Render manual input UI."""
    st.caption("Enter keywords manually to skip Google Trends extraction.")

    # Get current custom keywords
    current_custom = data.get('custom_keywords', [])
    default_value = ", ".join(current_custom) if current_custom else ""

    keywords_input = st.text_area(
        "Keywords (comma-separated)",
        value=default_value,
        placeholder="e.g., Christmas, cozy, warm lighting, festive",
        height=100,
        key="step2_manual_keywords"
    )

    if st.button("Use These Keywords", key="step2_use_manual", type="primary", use_container_width=True):
        if keywords_input.strip():
            keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
            update_step_data(2, {
                'extracted_trends': [],
                'selected_keywords': [],
                'custom_keywords': keywords
            })
            st.success(f"{len(keywords)} keywords loaded!")
            st.rerun()
        else:
            st.error("Please enter at least one keyword.")
