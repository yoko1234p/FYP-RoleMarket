# obj4_web_app/components/stepper.py
"""
Horizontal stepper component for step-by-step wizard.
"""

import streamlit as st
from typing import List


STEP_LABELS = ["角色", "趨勢", "Prompt", "圖片", "完成"]


def render_stepper(current_step: int, completed_steps: List[int]) -> None:
    """
    Render horizontal stepper progress bar at top of page.

    Args:
        current_step: Current active step (1-5)
        completed_steps: List of completed step numbers
    """
    steps_html = []

    for i, label in enumerate(STEP_LABELS, 1):
        # Determine step state
        if i in completed_steps:
            state = "completed"
            circle_content = "✓"
        elif i == current_step:
            state = "active"
            circle_content = str(i)
        else:
            state = "pending"
            circle_content = str(i)

        # Line state (between steps)
        line_class = "completed" if i in completed_steps and i < len(STEP_LABELS) else ""
        line_html = f'<div class="stepper-line {line_class}"></div>' if i < len(STEP_LABELS) else ""

        step_html = f"""
        <div class="stepper-step">
            <div class="stepper-circle {state}">{circle_content}</div>
            <div class="stepper-label {state}">{label}</div>
            {line_html}
        </div>
        """
        steps_html.append(step_html)

    stepper_html = f"""
    <div class="stepper-container">
        {''.join(steps_html)}
    </div>
    """

    st.markdown(stepper_html, unsafe_allow_html=True)


def get_step_label(step_num: int) -> str:
    """Get label for a step number."""
    if 1 <= step_num <= len(STEP_LABELS):
        return STEP_LABELS[step_num - 1]
    return f"Step {step_num}"
