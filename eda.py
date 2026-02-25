# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from EDA import (
    executive_summary,
    numeric_summary,
    split_columns,
    column_intelligence,
    dataset_overview,
    top_categorical_insights,
    auto_plots,
    data_quality_bullets,
    detect_quality_issues,
    plot_missingness_bar,
    recommendations,
    correlation_insights,
    correlation_heatmap,
    outlier_report,
    data_dictionary,
)

st.set_page_config(page_title="Auto-EDA Platform", layout="wide")

st.title("Auto-EDA Generation Platform")
st.write("Upload a CSV file to generate an executive-style data summary.")

# Sidebar navigation (1)
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Quality", "Insights", "Visuals", "Data Dictionary"],
    index=0
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Preview
st.subheader("Preview of Data")
st.dataframe(df.head(20), use_container_width=True)

numeric_cols, categorical_cols, datetime_cols = split_columns(df)

# -------------------------
# OVERVIEW PAGE
# -------------------------
if page == "Overview":
    tabs = st.tabs(["Overview", "Executive Summary", "Recommendations"])

    with tabs[0]:
        st.subheader("Dataset Overview")
        for b in dataset_overview(df):
            st.markdown(f"- {b}")

    with tabs[1]:
        st.subheader("Executive Summary")
        for b in executive_summary(df):
            st.markdown(f"- {b}")

    with tabs[2]:
        st.subheader("What to do next (Actionable!)")
        rec = recommendations(df)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Fix now")
            if rec["fix_now"]:
                for b in rec["fix_now"]:
                    st.markdown(f"- {b}")
            else:
                st.success("No urgent fixes detected.")

        with c2:
            st.markdown("### Analyze next")
            for b in rec["analyze_next"]:
                st.markdown(f"- {b}")

        with c3:
            st.markdown("### Modeling readiness")
            for b in rec["modeling_ready"]:
                st.markdown(f"- {b}")

# -------------------------
# QUALITY PAGE
# -------------------------
elif page == "Quality":
    tabs = st.tabs(["Business Summary", "Missingness Chart", "Detailed Flags"])

    with tabs[0]:
        st.subheader("Data Quality & Risks")
        for b in data_quality_bullets(df):
            st.markdown(f"- {b}")

    with tabs[1]:
        st.subheader("Top Missing Columns")
        fig = plot_missingness_bar(df, top_n=10)
        if fig is None:
            st.info("No missingness detected.")
        else:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tabs[2]:
        st.subheader("Detailed quality flags")
        issues = detect_quality_issues(df)
        if not issues:
            st.success("No major quality issues detected.")
        else:
            for it in issues:
                if it.level == "critical":
                    st.error(it.message)
                elif it.level == "warning":
                    st.warning(it.message)
                else:
                    st.info(it.message)

# -------------------------
# INSIGHTS PAGE
# -------------------------
elif page == "Insights":
    tabs = st.tabs(["Top Categories", "Outliers", "Correlation", "Numeric Summary"])

    with tabs[0]:
        st.subheader("Top Categories (What Actually Dominates!)")
        cat_insights = top_categorical_insights(df)
        if not cat_insights:
            st.info("No low-cardinality categorical columns found.")
        else:
            for col, info in cat_insights.items():
                with st.expander(col, expanded=False):
                    st.dataframe(info["distribution"], use_container_width=True)
                    for b in info["insights"]:
                        st.markdown(f"- {b}")

    with tabs[1]:
        st.subheader("Outlier Report (IQR based)")
        rep = outlier_report(df, numeric_cols)
        if rep.empty:
            st.info("No numeric columns (or not enough data) for outlier report.")
        else:
            st.dataframe(rep, use_container_width=True)

    with tabs[2]:
        st.subheader("Correlation Insights")
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation.")
        else:
            heat = correlation_heatmap(df, numeric_cols)
            if heat is not None:
                st.pyplot(heat, use_container_width=True)
                plt.close(heat)

            bullets = correlation_insights(df, numeric_cols, top_k=6)
            if bullets:
                st.markdown("**Top relationships**")
                for b in bullets:
                    st.markdown(f"- {b}")
            else:
                st.info("No strong correlations detected (≥ 0.5 threshold).")

    with tabs[3]:
        st.subheader("Numeric Summary")
        ns = numeric_summary(df, numeric_cols)
        if ns.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(ns, use_container_width=True)

# -------------------------
# VISUALS PAGE
# -------------------------
elif page == "Visuals":
    st.subheader("Visual Insights")

    plot_mode = st.radio("Mode", ["Auto-selected", "Pick columns"], horizontal=True)

    if plot_mode == "Auto-selected":
        plots = auto_plots(df)
        if not plots["numeric"] and not plots["categorical"]:
            st.info("No suitable columns found for auto visuals.")
        else:
            cols = st.columns(2)
            i = 0

            for col, fig in plots["numeric"].items():
                with cols[i % 2]:
                    st.markdown(f"**Numeric distribution: {col}**")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption("How to read: look for skew, unusual spikes, and outliers.")
                i += 1

            for col, fig in plots["categorical"].items():
                with cols[i % 2]:
                    st.markdown(f"**Category share: {col}**")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption("How to read: if one bar dominates, the dataset is imbalanced for that category.")
                i += 1

    else:
        top_n = st.slider("Top N categories", 3, 15, 8)
        normalize = st.toggle("Show percent (instead of count)", value=True)

        pick_cat = st.multiselect("Categorical columns", options=categorical_cols, default=categorical_cols[:2])
        pick_num = st.multiselect("Numeric columns", options=numeric_cols, default=numeric_cols[:2])

        for col in pick_cat:
            vc = df[col].fillna("<<MISSING>>").value_counts().head(top_n)
            if vc.empty:
                continue

            if normalize:
                vc = (vc / vc.sum() * 100).round(1)

            fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=160)
            vc.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"{col} (Top {top_n})")
            ax.set_xlabel("Percent (%)" if normalize else "Count")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        for col in pick_num:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 20:
                st.info(f"Not enough values to plot: {col}")
                continue

            fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=160)
            ax.hist(s, bins=30)
            ax.set_title(f"Distribution: {col}")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# -------------------------
# DATA DICTIONARY PAGE
# -------------------------
elif page == "Data Dictionary":
    st.subheader("Data Dictionary")
    dd = data_dictionary(df)
    st.dataframe(dd, use_container_width=True)

    st.download_button(
        "Download data dictionary (CSV)",
        dd.to_csv(index=False).encode("utf-8"),
        file_name="data_dictionary.csv",
        mime="text/csv"
    )
