"""
Custom CSS styles for stepper wizard UI.
"""

import streamlit as st


def inject_custom_css() -> None:
    """Inject custom CSS for stepper and step cards."""
    st.markdown("""
    <style>
    /* CSS Variables */
    :root {
        --primary-blue: #1E88E5;
        --success-green: #4CAF50;
        --border-grey: #E0E0E0;
        --bg-light: #F8F9FA;
        --bg-white: #FFFFFF;
        --text-muted: #9E9E9E;
        --shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Stepper Container */
    .stepper-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 16px 0 20px 0;
        margin-bottom: 0;
        background: white;
    }

    .stepper-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        flex: 1;
        max-width: 150px;
    }

    .stepper-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        z-index: 1;
        transition: all 0.3s ease;
    }

    .stepper-circle.pending {
        background: white;
        border: 2px solid #E0E0E0;
        color: #9E9E9E;
    }

    .stepper-circle.active {
        background: #1E88E5;
        border: 2px solid #1E88E5;
        color: white;
        box-shadow: 0 2px 8px rgba(30, 136, 229, 0.4);
    }

    .stepper-circle.completed {
        background: #4CAF50;
        border: 2px solid #4CAF50;
        color: white;
    }

    .stepper-label {
        margin-top: 8px;
        font-size: 12px;
        text-align: center;
        color: #666;
    }

    .stepper-label.active {
        color: #1E88E5;
        font-weight: 600;
    }

    .stepper-label.completed {
        color: #4CAF50;
    }

    .stepper-line {
        position: absolute;
        top: 18px;
        left: calc(50% + 20px);
        width: calc(100% - 40px);
        height: 2px;
        background: #E0E0E0;
    }

    .stepper-line.completed {
        background: #4CAF50;
    }

    /* Step Cards */
    .step-card {
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }

    .step-card.completed {
        background: #F8F9FA;
        border-left: 4px solid #4CAF50;
    }

    .step-card.active {
        background: #FFFFFF;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 20px 24px;
    }

    .step-card.pending {
        background: #FAFAFA;
        border-left: 4px solid #E0E0E0;
        opacity: 0.6;
    }

    .step-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .step-card-title {
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .step-card-title .icon {
        font-size: 18px;
    }

    .step-card-summary {
        color: #666;
        font-size: 14px;
        margin-top: 4px;
    }

    .step-card-content {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #E0E0E0;
    }

    /* Navigation Buttons */
    .step-nav-container {
        display: flex;
        justify-content: space-between;
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid #E0E0E0;
    }

    /* Edit Button */
    .edit-btn {
        background: none;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 12px;
        color: #666;
        cursor: pointer;
        transition: all 0.2s;
    }

    .edit-btn:hover {
        border-color: #1E88E5;
        color: #1E88E5;
    }

    /* Hide Streamlit default elements for cleaner look */
    .step-card .stButton > button {
        width: auto;
    }
    </style>
    """, unsafe_allow_html=True)
