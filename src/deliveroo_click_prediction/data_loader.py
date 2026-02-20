"""Data loading and session-state initialization."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyreadr
import streamlit as st

from deliveroo_click_prediction.config import (
    LOCAL_RDATA_FILENAME,
    LOCAL_RDATA_PATH,
    SESSION_KEY_DF_NEW,
    SESSION_KEY_DF_TRAIN,
    SESSION_KEY_X_TRAIN_FULL,
    SESSION_KEY_Y_TRAIN_FULL,
    TARGET_COL,
)


@st.cache_data
def load_rdata_bytes(file_bytes: bytes) -> Mapping[str, Any]:
    """Load an RData payload from bytes via a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".RData", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        return pyreadr.read_r(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def load_local_rdata() -> Mapping[str, Any]:
    """Load local RData file from configured raw data location."""
    dataset_path = LOCAL_RDATA_PATH
    return pyreadr.read_r(str(dataset_path))


def load_datasets_from_sidebar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render sidebar data controls and initialize datasets in session state."""
    st.sidebar.header("Data")

    uploaded_rdata = st.sidebar.file_uploader(
        f"Upload {LOCAL_RDATA_FILENAME}",
        type=["rdata", "RData"],
    )
    use_local = st.sidebar.checkbox("Use local file in data/raw", value=True)

    result: Mapping[str, Any] | None = None

    if uploaded_rdata is not None:
        try:
            result = load_rdata_bytes(uploaded_rdata.getvalue())
            st.sidebar.success("Loaded RData from upload.")
        except Exception as exc:
            st.sidebar.error(f"Failed to read uploaded RData: {exc}")
            st.stop()
    elif use_local:
        try:
            result = load_local_rdata()
            st.sidebar.success(f"Loaded local {LOCAL_RDATA_PATH}.")
        except FileNotFoundError:
            st.sidebar.error(f"'{LOCAL_RDATA_PATH}' not found.")
            st.stop()
        except Exception as exc:
            st.sidebar.error(f"Failed to read local RData: {exc}")
            st.stop()
    else:
        st.info("Upload an .RData file or enable local loading in the sidebar.")
        st.stop()

    required_objects = {"ClickTraining", "ClickPrediction"}
    missing_objects = sorted(required_objects.difference(result.keys()))
    if missing_objects:
        st.error(
            "Expected objects 'ClickTraining' and 'ClickPrediction' in RData, "
            f"missing: {missing_objects}. Found: {list(result.keys())}"
        )
        st.stop()

    df_train = result["ClickTraining"].copy()
    df_new = result["ClickPrediction"].copy()

    if TARGET_COL not in df_train.columns:
        st.error(
            f"Target column '{TARGET_COL}' not found in training data. "
            f"Columns: {list(df_train.columns)[:20]}"
        )
        st.stop()

    x_train_full = df_train.drop(columns=[TARGET_COL])
    y_train_full = df_train[TARGET_COL].astype(int)

    st.session_state[SESSION_KEY_DF_TRAIN] = df_train
    st.session_state[SESSION_KEY_DF_NEW] = df_new
    st.session_state[SESSION_KEY_X_TRAIN_FULL] = x_train_full
    st.session_state[SESSION_KEY_Y_TRAIN_FULL] = y_train_full

    st.sidebar.write(f"Train rows: {df_train.shape[0]:,} | cols: {df_train.shape[1]:,}")
    st.sidebar.write(f"Pred rows: {df_new.shape[0]:,} | cols: {df_new.shape[1]:,}")

    return df_train, df_new


def render_dataset_preview(df_train: pd.DataFrame, df_new: pd.DataFrame) -> None:
    """Render lightweight data preview."""
    with st.expander("Preview datasets"):
        st.write("Training dataset head:")
        st.dataframe(df_train.head())
        st.write("Prediction dataset head:")
        st.dataframe(df_new.head())
