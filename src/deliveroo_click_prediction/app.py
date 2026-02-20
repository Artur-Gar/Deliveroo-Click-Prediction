"""Main Streamlit application orchestration."""

import streamlit as st

from deliveroo_click_prediction.config import (
    SESSION_KEY_METRICS,
    SESSION_KEY_MODEL,
    SESSION_KEY_X_TEST,
    SESSION_KEY_X_TRAIN_FULL,
    SESSION_KEY_Y_PRED,
    SESSION_KEY_Y_PROB,
    SESSION_KEY_Y_TEST,
    SESSION_KEY_Y_TRAIN_FULL,
)
from deliveroo_click_prediction.data_loader import load_datasets_from_sidebar, render_dataset_preview
from deliveroo_click_prediction.model import train_xgb_model
from deliveroo_click_prediction.sidebar import render_model_configuration
from deliveroo_click_prediction.tabs import (
    render_batch_predictions_tab,
    render_business_insights_tab,
    render_model_metrics_tab,
    render_single_prediction_tab,
)
from deliveroo_click_prediction.ui import configure_page


def _store_training_artifacts(artifacts: dict) -> None:
    """Persist training outputs in Streamlit session state."""
    st.session_state[SESSION_KEY_MODEL] = artifacts["model"]
    st.session_state[SESSION_KEY_METRICS] = artifacts["metrics"]
    st.session_state[SESSION_KEY_X_TEST] = artifacts["X_test"]
    st.session_state[SESSION_KEY_Y_TEST] = artifacts["y_test"]
    st.session_state[SESSION_KEY_Y_PRED] = artifacts["y_pred"]
    st.session_state[SESSION_KEY_Y_PROB] = artifacts["y_prob"]


def _train_if_requested(val_size: float, should_train: bool) -> None:
    """Run model training from sidebar action."""
    if not should_train:
        return

    if (
        SESSION_KEY_X_TRAIN_FULL not in st.session_state
        or SESSION_KEY_Y_TRAIN_FULL not in st.session_state
    ):
        st.sidebar.error("Please upload and validate data first.")
        return

    x_train_full = st.session_state[SESSION_KEY_X_TRAIN_FULL]
    y_train_full = st.session_state[SESSION_KEY_Y_TRAIN_FULL]

    with st.spinner("Training XGBoost model..."):
        artifacts = train_xgb_model(x_train_full, y_train_full, val_size=val_size)

    _store_training_artifacts(artifacts)
    st.sidebar.success("Training complete.")
    st.sidebar.write(f"**AUC Score:** {artifacts['metrics']['AUC']:.3f}")


def run() -> None:
    """Run the full Streamlit app."""
    configure_page()
    df_train, df_new = load_datasets_from_sidebar()
    render_dataset_preview(df_train, df_new)

    val_size, threshold, should_train = render_model_configuration()
    _train_if_requested(val_size, should_train)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Model Metrics",
            "Business Insights",
            "Single Prediction",
            "Batch Predictions",
        ]
    )

    with tab1:
        render_model_metrics_tab()
    with tab2:
        render_business_insights_tab()
    with tab3:
        render_single_prediction_tab(threshold=threshold)
    with tab4:
        render_batch_predictions_tab(threshold=threshold)


if __name__ == "__main__":
    run()
