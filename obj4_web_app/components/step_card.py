# obj4_web_app/components/step_card.py
"""
Step card component with three states: completed, active, pending.
"""

import streamlit as st
from typing import Optional, Literal


def render_step_card(
    step_num: int,
    title: str,
    state: Literal['completed', 'active', 'pending'],
    summary: Optional[str] = None,
    on_edit: Optional[str] = None
) -> bool:
    """
    Render a step card container with appropriate styling.

    Args:
        step_num: Step number (1-5)
        title: Step title (e.g., "角色資訊")
        state: Card state - 'completed', 'active', or 'pending'
        summary: Summary text for completed state
        on_edit: Key for edit button (completed state only)

    Returns:
        True if this card should render its content (active state)
    """
    # Icons for each state
    icons = {
        'completed': '✓',
        'active': '●',
        'pending': '○'
    }

    icon = icons.get(state, '○')

    if state == 'completed':
        # Collapsed card with summary
        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                st.html(f"""
                <div class="step-card completed">
                    <div class="step-card-header">
                        <span class="step-card-title">
                            <span class="icon" style="color: #4CAF50;">{icon}</span>
                            Step {step_num}: {title}
                        </span>
                    </div>
                    <div class="step-card-summary">{summary or ''}</div>
                </div>
                """)
            with col2:
                if on_edit:
                    if st.button("編輯", key=on_edit, use_container_width=True):
                        return True  # Signal to go back to this step
        return False

    elif state == 'active':
        # Expanded card - content rendered by caller
        # Add id for scroll target and JavaScript to auto-scroll
        st.html(f"""
        <div id="active-step-card" class="step-card active">
            <div class="step-card-header">
                <span class="step-card-title">
                    <span class="icon" style="color: #1E88E5;">{icon}</span>
                    Step {step_num}: {title}
                </span>
            </div>
        </div>
        <script>
            // Auto-scroll to active step card
            setTimeout(function() {{
                var el = document.getElementById('active-step-card');
                if (el) {{
                    el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}, 100);
        </script>
        """)
        return True  # Caller should render content

    else:  # pending
        # Locked card
        st.html(f"""
        <div class="step-card pending">
            <div class="step-card-header">
                <span class="step-card-title">
                    <span class="icon">{icon}</span>
                    Step {step_num}: {title}
                </span>
                <span style="color: #9E9E9E;">🔒</span>
            </div>
        </div>
        """)
        return False


def render_step_navigation(
    current_step: int,
    total_steps: int = 5,
    on_back: Optional[str] = None,
    on_next: Optional[str] = None,
    next_label: str = "Next",
    next_disabled: bool = False
) -> Optional[str]:
    """
    Render Back/Next navigation buttons.

    Args:
        current_step: Current step number
        total_steps: Total number of steps
        on_back: Key for back button
        on_next: Key for next button
        next_label: Label for next button
        next_disabled: Whether next button is disabled

    Returns:
        'back' or 'next' if clicked, None otherwise
    """
    col1, col2, col3 = st.columns([1, 2, 1])

    result = None

    with col1:
        if current_step > 1 and on_back:
            if st.button("← Back", key=on_back, use_container_width=True):
                result = 'back'

    with col3:
        if current_step < total_steps and on_next:
            if st.button(f"{next_label} →", key=on_next, type="primary",
                        use_container_width=True, disabled=next_disabled):
                result = 'next'

    return result
