# obj4_web_app/components/steps/step5_complete.py
"""
Step 5: Complete
"""

import streamlit as st
from datetime import datetime
from ..state_manager import get_step_data, update_step_data, reset_wizard


def render_step5_content(design_generator) -> None:
    """Render Step 5: Complete summary."""
    step1 = get_step_data(1)
    step2 = get_step_data(2)
    step3 = get_step_data(3)
    step4 = get_step_data(4)

    # Mark completion time
    update_step_data(5, {'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M')})

    st.markdown("## Design Generation Complete!")

    st.markdown("---")

    # Summary card
    st.markdown("### Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Character:**")
        st.info(f"{step1.get('character_name', 'N/A')}")

        st.markdown("**Theme:**")
        st.info(f"{step3.get('theme', 'N/A')}")

    with col2:
        keywords = step2.get('selected_keywords', []) + step2.get('custom_keywords', [])
        st.markdown("**Keywords:**")
        st.info(f"{len(keywords)} keywords")

        images = step4.get('generated_images', [])
        successful = [img for img in images if img.get('success')]
        if successful:
            avg_clip = sum(img.get('clip_similarity', 0) for img in successful) / len(successful)
            st.markdown("**Images:**")
            st.info(f"{len(successful)} images (Avg CLIP: {avg_clip:.2f})")

    # Show prompt
    with st.expander("Generated Prompt"):
        st.code(step3.get('generated_prompt', 'N/A'))

    # Show images
    if successful:
        st.markdown("---")
        st.markdown("### Generated Images")

        cols = st.columns(len(successful))
        for i, img in enumerate(successful):
            with cols[i]:
                st.image(img['image'], use_container_width=True)
                st.caption(f"CLIP: {img.get('clip_similarity', 0):.4f}")

    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if design_generator and successful:
            # Download all as zip
            if st.button("Download All", key="step5_download_all", use_container_width=True):
                st.info("Use individual download buttons above for now")

    with col2:
        if st.button("Go to Sales Forecast ->", key="step5_forecast",
                    type="primary", use_container_width=True):
            # Copy images to main session state for Page 2
            st.session_state['generated_images'] = step4.get('generated_images', [])
            st.session_state['generated_prompt'] = step3.get('generated_prompt', '')
            st.switch_page("pages/2_Sales_Forecasting.py")

    with col3:
        if st.button("Start New Design", key="step5_reset", use_container_width=True):
            reset_wizard()
            st.rerun()
