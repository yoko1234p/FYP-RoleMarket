# obj4_web_app/components/stepper.py
"""
Horizontal stepper component for step-by-step wizard.
"""

import streamlit as st
import streamlit.components.v1 as components
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

    # Render stepper HTML via st.html (for styling)
    stepper_html = f"""
    <div class="stepper-container">
        {''.join(steps_html)}
    </div>
    """
    st.html(stepper_html)

    # Inject sticky JavaScript via components.html iframe
    # This creates an iframe that can access the parent document
    sticky_script = """
    <!DOCTYPE html>
    <html>
    <head><style>body{margin:0;padding:0;height:0;overflow:hidden;}</style></head>
    <body>
    <script>
    (function() {
        var doc = window.parent.document;
        var win = window.parent;

        if (win.stepperStickyInitialized) return;
        win.stepperStickyInitialized = true;

        var isFixed = false;
        var placeholder = null;
        var originalWidth = 0;
        var originalLeft = 0;

        function initAndCheck() {
            var container = doc.querySelector('.stepper-container');
            if (!container) return;

            var stHtml = container.closest('.stHtml');
            if (!stHtml) return;

            var scrollContainer = doc.querySelector('section.stMain');
            if (!scrollContainer) return;

            var scrollTop = scrollContainer.scrollTop;
            var threshold = 120;

            if (!placeholder && stHtml.parentNode) {
                placeholder = doc.querySelector('.stepper-placeholder');
                if (!placeholder) {
                    placeholder = doc.createElement('div');
                    placeholder.className = 'stepper-placeholder';
                    placeholder.style.display = 'none';
                    stHtml.parentNode.insertBefore(placeholder, stHtml);
                }
            }

            if (!isFixed) {
                var rect = stHtml.getBoundingClientRect();
                originalWidth = rect.width;
                originalLeft = rect.left;
                if (placeholder) placeholder.style.height = rect.height + 'px';
            }

            if (scrollTop > threshold && !isFixed) {
                stHtml.style.position = 'fixed';
                stHtml.style.top = '56px';
                stHtml.style.left = originalLeft + 'px';
                stHtml.style.width = originalWidth + 'px';
                stHtml.style.zIndex = '9999';
                stHtml.style.background = 'white';
                stHtml.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
                stHtml.style.padding = '4px 0';
                if (placeholder) placeholder.style.display = 'block';
                isFixed = true;
            } else if (scrollTop <= threshold && isFixed) {
                stHtml.style.position = '';
                stHtml.style.top = '';
                stHtml.style.left = '';
                stHtml.style.width = '';
                stHtml.style.zIndex = '';
                stHtml.style.background = '';
                stHtml.style.boxShadow = '';
                stHtml.style.padding = '';
                if (placeholder) placeholder.style.display = 'none';
                isFixed = false;
            }
        }

        var scrollContainer = doc.querySelector('section.stMain');
        if (scrollContainer) {
            scrollContainer.addEventListener('scroll', initAndCheck);
        }
        setInterval(initAndCheck, 100);
    })();
    </script>
    </body>
    </html>
    """
    # Render with height=1 (minimal but allows iframe to render)
    components.html(sticky_script, height=1, scrolling=False)


def get_step_label(step_num: int) -> str:
    """Get label for a step number."""
    if 1 <= step_num <= len(STEP_LABELS):
        return STEP_LABELS[step_num - 1]
    return f"Step {step_num}"
