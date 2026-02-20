"""Model performance tab."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

from deliveroo_click_prediction.config import (
    SESSION_KEY_METRICS,
    SESSION_KEY_MODEL,
    SESSION_KEY_Y_PRED,
    SESSION_KEY_Y_PROB,
    SESSION_KEY_Y_TEST,
)


def render_model_metrics_tab() -> None:
    """Render model diagnostics."""
    st.header("Model Performance Evaluation")

    if SESSION_KEY_METRICS not in st.session_state:
        st.info("Go to the sidebar and click 'Train Model' to generate results.")
        return

    metrics = st.session_state[SESSION_KEY_METRICS]
    model = st.session_state[SESSION_KEY_MODEL]
    y_test = st.session_state[SESSION_KEY_Y_TEST]
    y_pred = st.session_state[SESSION_KEY_Y_PRED]
    y_prob = st.session_state[SESSION_KEY_Y_PROB]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC AUC Score", f"{metrics['AUC']:.3f}")
    c2.metric("F1 Score", f"{metrics['F1_score']:.2%}")
    c3.metric("Precision", f"{metrics['Precision']:.2%}")
    c4.metric("Recall", f"{metrics['Recall']:.2%}")
    c5.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    st.divider()

    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Teal",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Click", "Click"],
            y=["No Click", "Click"],
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, width='stretch')

    with col_viz2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = px.area(
            x=fpr,
            y=tpr,
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
        )
        fig_roc.add_shape(
            type="line",
            line=dict(dash="dash", color="gray"),
            x0=0,
            x1=1,
            y0=0,
            y1=1,
        )
        fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_roc, width='stretch')

    st.divider()
    col_adv1, col_adv2 = st.columns(2)

    with col_adv1:
        st.subheader("Precision-Recall Curve")
        st.caption("Better than ROC for rare events (like ad clicks).")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        fig_pr = px.area(
            x=rec,
            y=prec,
            labels=dict(x="Recall", y="Precision"),
            title=f"PR AUC: {average_precision_score(y_test, y_prob):.3f}",
        )
        baseline = y_test.mean()
        fig_pr.add_hline(y=baseline, line_dash="dash", line_color="red", annotation_text="Random baseline")
        st.plotly_chart(fig_pr, width='stretch')

    with col_adv2:
        st.subheader("Model Confidence Histogram")
        st.caption("Peaks near 0 and 1 indicate stronger discrimination.")
        hist_df = pd.DataFrame({"Probability": y_prob, "Actual Outcome": y_test})
        hist_df["Actual Outcome"] = hist_df["Actual Outcome"].map({0: "No Click", 1: "Click"})
        fig_hist = px.histogram(
            hist_df,
            x="Probability",
            color="Actual Outcome",
            nbins=50,
            barmode="overlay",
            color_discrete_map={"No Click": "red", "Click": "teal"},
            opacity=0.6,
        )
        st.plotly_chart(fig_hist, width='stretch')

    st.subheader("Feature Importance")
    try:
        xgb_model = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        importances = xgb_model.feature_importances_

        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(10)
        fig_imp = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text_auto=".3f",
            color="Importance",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, width='stretch')
    except Exception:
        st.warning("Could not extract feature importance.")
