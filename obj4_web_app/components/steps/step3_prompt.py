# obj4_web_app/components/steps/step3_prompt.py
"""
Step 3: Generate Prompt
"""

import streamlit as st
from ..state_manager import get_step_data, update_step_data, go_to_next_step, go_to_previous_step


# Theme options
THEME_OPTIONS = {
    "Christmas": "Cozy Christmas",
    "Halloween": "Spooky Halloween",
    "Chinese New Year": "Chinese New Year / Spring Festival",
    "Valentine's Day": "Valentine's Day Romance",
    "Easter": "Easter Spring",
    "Spring": "Spring Blossom",
    "Summer": "Summer Beach",
    "Autumn": "Autumn Harvest",
    "Winter": "Winter Wonderland",
    "Birthday": "Birthday Party",
    "Gaming": "Gaming Culture",
    "Sports": "Sports & Fitness",
    "Custom": "Custom Theme"
}


def render_step3_content(enhanced_trends, character_name: str, character_desc: str) -> None:
    """
    Render Step 3: Generate Prompt form.

    Args:
        enhanced_trends: EnhancedTrendsWrapper instance
        character_name: Character name from Step 1
        character_desc: Character description from Step 1
    """
    data = get_step_data(3)
    step2_data = get_step_data(2)

    # Show keywords
    all_keywords = step2_data.get('selected_keywords', []) + step2_data.get('custom_keywords', [])
    st.markdown("**Keywords:**")
    st.info(", ".join(all_keywords) if all_keywords else "No keywords")

    st.markdown("---")

    # Theme selector
    st.markdown("**Theme / Context:**")
    st.caption("Theme provides context for AI prompt generation")

    theme_keys = list(THEME_OPTIONS.keys())
    current_theme = data.get('theme', '')

    # Find current theme index
    default_idx = 0
    for i, key in enumerate(theme_keys):
        if THEME_OPTIONS[key] == current_theme:
            default_idx = i
            break

    selected_theme_key = st.selectbox(
        "Select Theme",
        options=theme_keys,
        index=default_idx,
        key="step3_theme_select",
        label_visibility="collapsed"
    )

    # Custom theme input
    if selected_theme_key == "Custom":
        theme = st.text_input(
            "Enter Custom Theme",
            value=data.get('theme', '') if data.get('theme') not in THEME_OPTIONS.values() else '',
            placeholder="e.g., Cyberpunk Future, Vintage 80s",
            key="step3_custom_theme"
        )
    else:
        theme = THEME_OPTIONS[selected_theme_key]

    update_step_data(3, {'theme': theme})

    st.markdown("---")

    # Generate button
    if st.button("Generate Prompt", key="step3_generate", type="primary", use_container_width=True):
        if not theme.strip():
            st.error("Please select or enter a theme.")
        else:
            with st.spinner("Generating prompt..."):
                try:
                    prompt = enhanced_trends.generate_prompt_with_theme(
                        keywords=all_keywords,
                        theme=theme,
                        character_name=character_name,
                        character_desc=character_desc
                    )
                    update_step_data(3, {'generated_prompt': prompt})
                    st.success("Prompt generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show generated prompt
    current_prompt = data.get('generated_prompt', '')
    if current_prompt:
        st.markdown("---")
        st.markdown("**Generated Prompt:**")

        edited_prompt = st.text_area(
            "Edit prompt if needed",
            value=current_prompt,
            height=150,
            key="step3_prompt_edit",
            label_visibility="collapsed"
        )

        if edited_prompt != current_prompt:
            if st.button("Save Changes", key="step3_save_prompt"):
                update_step_data(3, {'generated_prompt': edited_prompt})
                st.success("Prompt updated!")
                st.rerun()

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("<- Back", key="step3_back", use_container_width=True):
            go_to_previous_step()
            st.rerun()

    with col3:
        is_valid = bool(data.get('generated_prompt', '').strip())
        if st.button("Next: 生成圖片 ->", key="step3_next", type="primary",
                    use_container_width=True, disabled=not is_valid):
            go_to_next_step()
            st.rerun()

    if not is_valid:
        st.warning("Please generate a prompt first.")
