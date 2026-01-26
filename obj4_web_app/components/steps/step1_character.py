# obj4_web_app/components/steps/step1_character.py
"""
Step 1: Character Information
"""

import streamlit as st
from ..state_manager import get_step_data, update_step_data, go_to_next_step


def render_step1_content() -> None:
    """Render Step 1: Character Information form."""
    data = get_step_data(1)

    # Character Name
    character_name = st.text_input(
        "Character Name",
        value=data.get('character_name', 'Lulu Pig'),
        help="Enter character name, e.g., Lulu Pig",
        key="step1_character_name"
    )

    # Character Description
    character_desc = st.text_area(
        "Character Description",
        value=data.get('character_desc', 'Cute pink pig with big eyes and chubby body'),
        help="Briefly describe character features",
        height=100,
        key="step1_character_desc"
    )

    # Save data on change
    update_step_data(1, {
        'character_name': character_name,
        'character_desc': character_desc
    })

    # Validation
    is_valid = bool(character_name.strip() and character_desc.strip())

    if not is_valid:
        st.warning("Please fill in both character name and description.")

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("Next: 趨勢關鍵字 →", key="step1_next", type="primary",
                    use_container_width=True, disabled=not is_valid):
            go_to_next_step()
            st.rerun()
