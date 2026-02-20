"""Single-profile prediction tab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from deliveroo_click_prediction.config import SESSION_KEY_DF_TRAIN, SESSION_KEY_MODEL


def render_single_prediction_tab(threshold: float) -> None:
    """Render single-user prediction simulator."""
    st.header("Real-Time Prediction Simulator")
    st.info("Adjust the profile below. The model predicts click probability.")

    if SESSION_KEY_MODEL not in st.session_state or SESSION_KEY_DF_TRAIN not in st.session_state:
        st.warning("Please train the model in the sidebar first.")
        return

    model = st.session_state[SESSION_KEY_MODEL]
    df_train = st.session_state[SESSION_KEY_DF_TRAIN]

    with st.form("prediction_form"):
        st.subheader("Define User Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Location and Device**")
            region = st.selectbox("Region", options=sorted(df_train["Region"].dropna().unique()))
            carrier = st.selectbox("Mobile Carrier", options=sorted(df_train["Carrier"].dropna().unique()))

        with c2:
            st.markdown("**Context**")
            weekday = st.selectbox("Weekday", options=sorted(df_train["Weekday"].dropna().unique()))
            social = st.selectbox("Social Network", options=sorted(df_train["Social_Network"].dropna().unique()))
            daytime = st.slider("Time of Day (0=Morning, 1=Night)", 0.0, 1.0, 0.5)

        with c3:
            st.markdown("**User History**")
            rest_type = st.selectbox(
                "Restaurant Interest",
                options=sorted(df_train["Restaurant_Type"].dropna().unique()),
            )
            default_time = float(df_train["Time_On_Previous_Website"].mean())
            time_web = st.number_input("Time on Previous Website (sec)", min_value=0.0, value=default_time)

            default_orders = int(df_train["Number_of_Previous_Orders"].median())
            prev_orders = st.number_input("Previous Orders", min_value=0, step=1, value=default_orders)

        st.markdown("---")
        submitted = st.form_submit_button("Predict Probability", type="primary")

    if not submitted:
        return

    input_data = pd.DataFrame(
        {
            "Region": [region],
            "Daytime": [daytime],
            "Carrier": [carrier],
            "Time_On_Previous_Website": [time_web],
            "Weekday": [weekday],
            "Social_Network": [social],
            "Number_of_Previous_Orders": [prev_orders],
            "Restaurant_Type": [rest_type],
        }
    )
    probability = model.predict_proba(input_data)[0][1]
    is_click = probability > threshold

    st.subheader("Prediction Results")
    col_result, col_gauge = st.columns([1, 2])

    with col_result:
        if is_click:
            st.success(f"LIKELY CLICK\n\nProbability: {probability:.1%}")
            st.caption(
                f"Probability ({probability:.2f}) is above threshold ({threshold:.2f})."
            )
        else:
            st.error(f"NO CLICK\n\nProbability: {probability:.1%}")
            st.caption(
                f"Probability ({probability:.2f}) is below threshold ({threshold:.2f})."
            )

    with col_gauge:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": f"Conversion Probability (Threshold: {threshold:.0%})"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00CCBC" if is_click else "#EF553B"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "lightgray"},
                        {"range": [threshold * 100, 100], "color": "white"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, width='stretch')
