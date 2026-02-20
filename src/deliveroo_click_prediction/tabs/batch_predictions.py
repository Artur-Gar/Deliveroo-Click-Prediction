"""Batch prediction and segmentation tab."""

import plotly.express as px
import streamlit as st

from deliveroo_click_prediction.config import SESSION_KEY_DF_NEW, SESSION_KEY_MODEL


def render_batch_predictions_tab(threshold: float) -> None:
    """Render batch scoring workflow for ClickPrediction."""
    st.header("Batch Prediction and Analytics")
    st.info("Analyze the unlabeled ClickPrediction dataset and forecast behavior.")

    if SESSION_KEY_MODEL not in st.session_state:
        st.warning("Please train the model in the sidebar first.")
        return
    if SESSION_KEY_DF_NEW not in st.session_state:
        st.warning("Please load the 'ClickPrediction' data in the sidebar.")
        return

    model = st.session_state[SESSION_KEY_MODEL]
    df_new = st.session_state[SESSION_KEY_DF_NEW].copy()
    st.write(f"Loaded Prediction Dataset: {df_new.shape[0]:,} rows")

    if not st.button("Run Predictions and Analyze Batch", type="primary"):
        return

    with st.spinner("Generating predictions..."):
        probs = model.predict_proba(df_new)[:, 1]
        df_new["Click_Probability"] = probs
        df_new["Predicted_Status"] = (probs > threshold).astype(int)
        df_new["Status_Label"] = df_new["Predicted_Status"].map({1: "Likely Click", 0: "No Click"})

        st.subheader("1. Batch Summary Statistics")
        total_users = len(df_new)
        avg_confidence = probs.mean()
        expected_conversions = probs.sum()
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Batch Size", f"{total_users:,}")
        kpi2.metric("Avg. Click Probability", f"{avg_confidence:.1%}")
        kpi3.metric("Expected Conversions", f"{expected_conversions:.0f}")
        kpi4.metric("Threshold Selected", f"{threshold:.0%}")

        st.divider()
        st.subheader("2. Probability Distribution")
        fig_hist = px.histogram(
            df_new,
            x="Click_Probability",
            nbins=50,
            color="Status_Label",
            color_discrete_map={"Likely Click": "#00CCBC", "No Click": "#EF553B"},
            range_x=[0, 1],
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
        st.plotly_chart(fig_hist, width='stretch')

        st.divider()
        st.subheader("3. Segmentation by Quality (Avg. Probability)")
        st.caption("Which groups have the highest propensity to click?")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Region Quality**")
            reg_qual = df_new.groupby("Region")["Click_Probability"].mean().reset_index()
            fig_reg = px.bar(
                reg_qual,
                x="Click_Probability",
                y="Region",
                orientation="h",
                color="Click_Probability",
                color_continuous_scale="Teal",
                labels={"Click_Probability": "Avg Probability"},
                range_x=[0, 1],
            )
            fig_reg.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_reg, width='stretch')

        with c2:
            st.markdown("**Social Network Quality**")
            soc_qual = df_new.groupby("Social_Network")["Click_Probability"].mean().reset_index()
            fig_soc = px.bar(
                soc_qual,
                x="Social_Network",
                y="Click_Probability",
                color="Click_Probability",
                color_continuous_scale="Teal",
                labels={"Click_Probability": "Avg Probability"},
                range_y=[0, 1],
            )
            fig_soc.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_soc, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Carrier Quality**")
            car_qual = df_new.groupby("Carrier")["Click_Probability"].mean().reset_index()
            fig_car = px.bar(
                car_qual,
                x="Carrier",
                y="Click_Probability",
                color="Click_Probability",
                color_continuous_scale="Teal",
                labels={"Click_Probability": "Avg Probability"},
                range_y=[0, 1],
            )
            fig_car.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_car, width='stretch')

        with c4:
            st.markdown("**Restaurant Type Quality**")
            rest_qual = df_new.groupby("Restaurant_Type")["Click_Probability"].mean().reset_index()
            fig_rest = px.bar(
                rest_qual,
                x="Click_Probability",
                y="Restaurant_Type",
                orientation="h",
                color="Click_Probability",
                color_continuous_scale="Teal",
                labels={"Click_Probability": "Avg Probability"},
                range_x=[0, 1],
            )
            fig_rest.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_rest, width='stretch')

        st.divider()
        st.subheader("4. Export Results")
        st.write("Top 5 highest potential users:")
        top_leads = df_new.sort_values(by="Click_Probability", ascending=False).head(5).copy()
        top_leads["Click_Probability"] = top_leads["Click_Probability"].map(lambda value: f"{value:.1%}")
        st.dataframe(top_leads)

        csv_data = df_new.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Predictions CSV",
            data=csv_data,
            file_name="deliveroo_batch_predictions.csv",
            mime="text/csv",
        )
