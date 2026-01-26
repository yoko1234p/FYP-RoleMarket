# obj4_web_app/components/state_manager.py
"""
Session state manager for step wizard.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
from datetime import datetime


def init_step_state() -> None:
    """Initialize step wizard session state if not exists."""
    if 'wizard_current_step' not in st.session_state:
        st.session_state['wizard_current_step'] = 1

    if 'wizard_completed_steps' not in st.session_state:
        st.session_state['wizard_completed_steps'] = []

    if 'wizard_step_data' not in st.session_state:
        st.session_state['wizard_step_data'] = {
            1: {'character_name': 'Lulu Pig', 'character_desc': 'Cute pink pig with big eyes and chubby body'},
            2: {'method': 'auto', 'extracted_trends': [], 'selected_keywords': [], 'custom_keywords': []},
            3: {'theme': '', 'generated_prompt': ''},
            4: {'reference_image': '', 'num_images': 2, 'variation_mode': 'single', 'generated_images': []},
            5: {'completed_at': None}
        }


def get_current_step() -> int:
    """Get current step number."""
    return st.session_state.get('wizard_current_step', 1)


def set_current_step(step: int) -> None:
    """Set current step number."""
    st.session_state['wizard_current_step'] = step


def get_completed_steps() -> List[int]:
    """Get list of completed step numbers."""
    return st.session_state.get('wizard_completed_steps', [])


def mark_step_completed(step: int) -> None:
    """Mark a step as completed."""
    completed = get_completed_steps()
    if step not in completed:
        completed.append(step)
        st.session_state['wizard_completed_steps'] = sorted(completed)


def mark_step_incomplete(step: int) -> None:
    """Mark a step as incomplete (for editing)."""
    completed = get_completed_steps()
    # Remove this step and all subsequent steps
    st.session_state['wizard_completed_steps'] = [s for s in completed if s < step]


def get_step_data(step: int) -> Dict[str, Any]:
    """Get data for a specific step."""
    return st.session_state.get('wizard_step_data', {}).get(step, {})


def update_step_data(step: int, data: Dict[str, Any]) -> None:
    """Update data for a specific step."""
    if 'wizard_step_data' not in st.session_state:
        init_step_state()
    st.session_state['wizard_step_data'][step] = {
        **st.session_state['wizard_step_data'].get(step, {}),
        **data
    }


def go_to_next_step() -> None:
    """Navigate to next step and mark current as completed."""
    current = get_current_step()
    mark_step_completed(current)
    if current < 5:
        set_current_step(current + 1)


def go_to_previous_step() -> None:
    """Navigate to previous step."""
    current = get_current_step()
    if current > 1:
        set_current_step(current - 1)


def go_to_step(step: int) -> None:
    """Navigate to a specific step (for editing)."""
    mark_step_incomplete(step)
    set_current_step(step)


def reset_wizard() -> None:
    """Reset wizard to initial state."""
    st.session_state['wizard_current_step'] = 1
    st.session_state['wizard_completed_steps'] = []
    st.session_state['wizard_step_data'] = {
        1: {'character_name': 'Lulu Pig', 'character_desc': 'Cute pink pig with big eyes and chubby body'},
        2: {'method': 'auto', 'extracted_trends': [], 'selected_keywords': [], 'custom_keywords': []},
        3: {'theme': '', 'generated_prompt': ''},
        4: {'reference_image': '', 'num_images': 2, 'variation_mode': 'single', 'generated_images': []},
        5: {'completed_at': None}
    }


def get_step_summary(step: int) -> str:
    """Generate summary text for a completed step."""
    data = get_step_data(step)

    if step == 1:
        name = data.get('character_name', '')
        desc = data.get('character_desc', '')
        if desc and len(desc) > 50:
            desc = desc[:50] + '...'
        return f"{name} - {desc}" if name else ""

    elif step == 2:
        keywords = data.get('selected_keywords', []) + data.get('custom_keywords', [])
        count = len(keywords)
        preview = ', '.join(keywords[:3])
        if count > 3:
            preview += f' (+{count - 3} more)'
        return f"{count} keywords: {preview}" if keywords else "No keywords selected"

    elif step == 3:
        theme = data.get('theme', '')
        prompt = data.get('generated_prompt', '')
        if prompt:
            return f"Theme: {theme} | Prompt generated ({len(prompt)} chars)"
        return f"Theme: {theme}" if theme else ""

    elif step == 4:
        images = data.get('generated_images', [])
        successful = [img for img in images if img.get('success')]
        if successful:
            avg_clip = sum(img.get('clip_similarity', 0) for img in successful) / len(successful)
            return f"{len(successful)} images generated (Avg CLIP: {avg_clip:.2f})"
        return "No images generated"

    elif step == 5:
        completed_at = data.get('completed_at')
        return f"Completed at {completed_at}" if completed_at else "Ready"

    return ""
