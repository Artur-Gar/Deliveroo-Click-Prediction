"""Business insight tab based on historical training data."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from deliveroo_click_prediction.config import SESSION_KEY_DF_TRAIN, TARGET_COL


def render_business_insights_tab() -> None:
    """Render historical conversion insights."""
    st.header("Historical Business Insights")
    st.info("These insights are based on historical training data.")

    if SESSION_KEY_DF_TRAIN not in st.session_state:
        st.warning("Please load the 'ClickTraining' dataset in the sidebar first.")
        return

    df = st.session_state[SESSION_KEY_DF_TRAIN]
    target = TARGET_COL

    total_obs = len(df)
    total_conversions = df[target].sum()
    conversion_rate = df[target].mean()
    missed_opportunities = total_obs - total_conversions

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Historical Conversion Rate", f"{conversion_rate:.1%}")
    kpi2.metric("Total Historical Users", f"{total_obs:,}")
    kpi3.metric("Missed Opportunities", f"{missed_opportunities:,}")
    kpi4.metric("Avg Previous Orders", f"{df['Number_of_Previous_Orders'].mean():.1f}")
    st.divider()

    st.subheader("1. Segmentation: Who are we targeting?")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**By Social Network**")
        social_conv = df.groupby("Social_Network", observed=False)[target].mean().reset_index()
        fig_social = px.bar(
            social_conv,
            x="Social_Network",
            y=target,
            color=target,
            color_continuous_scale="RdYlGn",
            range_y=[0, 1],
        )
        fig_social.add_hline(y=conversion_rate, line_dash="dash", line_color="black")
        fig_social.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_social, width='stretch')

    with c2:
        st.markdown("**By Region**")
        region_conv = (
            df.groupby("Region", observed=False)[target].mean().reset_index().sort_values(by=target)
        )
        fig_region = px.bar(
            region_conv,
            x=target,
            y="Region",
            orientation="h",
            color=target,
            color_continuous_scale="RdYlGn",
            range_x=[0, 1],
        )
        fig_region.add_vline(x=conversion_rate, line_dash="dash", line_color="black")
        fig_region.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig_region, width='stretch')

    with c3:
        st.markdown("**By Mobile Carrier**")
        carrier_conv = (
            df.groupby("Carrier", observed=False)[target].mean().reset_index().sort_values(by=target)
        )
        fig_carrier = px.bar(
            carrier_conv,
            x="Carrier",
            y=target,
            color=target,
            color_continuous_scale="RdYlGn",
            range_y=[0, 1],
        )
        fig_carrier.add_hline(y=conversion_rate, line_dash="dash", line_color="black")
        fig_carrier.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_carrier, width='stretch')

    st.divider()
    st.subheader("2. Timing: When do they click?")
    c4, c5 = st.columns(2)

    with c4:
        st.markdown("**Conversion by Weekday**")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_conv = (
            df.groupby("Weekday", observed=False)[target].mean().reindex(day_order).reset_index()
        )
        fig_week = px.bar(
            weekday_conv,
            x="Weekday",
            y=target,
            color=target,
            color_continuous_scale="Viridis",
            range_y=[0, 1],
        )
        fig_week.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_week, width='stretch')

    df_chart = df.copy()
    df_chart["Status"] = df_chart[target].map({0: "No Click", 1: "Click"})
    with c5:
        st.markdown("**Impact of Time of Day**")
        fig_daytime = px.box(
            df_chart,
            x="Status",
            y="Daytime",
            color="Status",
            color_discrete_map={"No Click": "#EF553B", "Click": "#00CCBC"},
        )
        st.plotly_chart(fig_daytime, width='stretch')

    st.divider()
    st.subheader("3. User Behavior")
    c6, c7 = st.columns(2)

    with c6:
        st.markdown("**Time on Previous Website**")
        fig_time = px.box(
            df_chart,
            x="Status",
            y="Time_On_Previous_Website",
            color="Status",
            color_discrete_map={"No Click": "#EF553B", "Click": "#00CCBC"},
            points=False,
        )
        st.plotly_chart(fig_time, width='stretch')

    with c7:
        st.markdown("**Interest by Restaurant Type**")
        rest_conv = (
            df.groupby("Restaurant_Type", observed=False)[target].mean().reset_index().sort_values(by=target)
        )
        fig_rest = px.bar(
            rest_conv,
            x="Restaurant_Type",
            y=target,
            color=target,
            color_continuous_scale="RdYlGn",
        )
        fig_rest.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_rest, width='stretch')

    st.divider()
    st.subheader("4. Conversion Rate by Customer Loyalty")
    st.caption(
        "Conversion likelihood by number of previous orders to identify high-value repeat customers."
    )

    df_loyalty = df.copy()
    df_loyalty["Orders_Bucket"] = pd.cut(
        df_loyalty["Number_of_Previous_Orders"],
        bins=[-1, 0, 1, 3, 5, 10, np.inf],
        labels=["0 (New)", "1", "2-3", "4-5", "6-10", "10+"],
    )

    loyalty_conv = df_loyalty.groupby("Orders_Bucket", observed=False)[target].mean().reset_index()
    fig_loyalty = px.bar(
        loyalty_conv,
        x="Orders_Bucket",
        y=target,
        color=target,
        color_continuous_scale="RdYlGn",
        range_y=[0, 1],
        labels={"Orders_Bucket": "Number of Previous Orders", target: "Conversion Rate"},
        text=loyalty_conv[target].apply(lambda x: f"{x:.1%}"),
    )
    fig_loyalty.add_hline(y=conversion_rate, line_dash="dash", line_color="black", annotation_text="Overall Avg")
    fig_loyalty.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_loyalty, width='stretch')

    best_bucket = loyalty_conv.loc[loyalty_conv[target].idxmax(), "Orders_Bucket"]
    best_rate = loyalty_conv[target].max()
    st.success(
        f"Users with {best_bucket} previous orders show the highest conversion rate ({best_rate:.1%})."
    )

    st.divider()
    st.subheader("5. Geographic Disparity: Conversion Rate by Region")
    st.caption("Regional differences in conversion performance.")

    region_conv = df.groupby("Region", observed=False)[target].mean().reset_index().sort_values(by=target)
    fig_geo = px.bar(
        region_conv,
        x=target,
        y="Region",
        orientation="h",
        color=target,
        color_continuous_scale="RdYlGn",
        labels={target: "Conversion Rate"},
        text=region_conv[target].apply(lambda x: f"{x:.1%}"),
    )
    fig_geo.add_vline(x=conversion_rate, line_dash="dash", line_color="black", annotation_text="Overall Avg")
    fig_geo.update_layout(xaxis_tickformat=".0%", height=450, showlegend=False)
    st.plotly_chart(fig_geo, width='stretch')

    st.success("Regional differences are modest versus other behavioral/segment drivers.")
