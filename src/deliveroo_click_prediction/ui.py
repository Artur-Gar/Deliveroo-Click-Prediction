"""UI primitives and global page configuration."""

import streamlit as st


def configure_page() -> None:
    """Set global Streamlit config and baseline styles."""
    st.set_page_config(page_title="Deliveroo AI Suite", page_icon=":bike:", layout="wide")

    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stMetricValue"] {font-size: 24px; font-weight: 700;}
            .stTabs [aria-selected="true"] {background-color: #00CCBC !important; color: white !important;}
            div[data-testid="stDataFrame"] {width: 100%;}
        </style>
        """,
        unsafe_allow_html=True,
    )
