"""Sidebar controls."""

import streamlit as st


def render_model_configuration() -> tuple[float, float, bool]:
    """Render model controls and return selected values + train trigger."""
    st.sidebar.divider()
    st.sidebar.header("Model Configuration")

    val_size = st.sidebar.slider(
        "Validation Split Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        help="Percentage of data reserved for testing the model.",
    )

    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability cutoff. Users above this score are predicted as 'Click'.",
    )

    should_train = st.sidebar.button("Train Model", type="primary")
    return val_size, threshold, should_train
