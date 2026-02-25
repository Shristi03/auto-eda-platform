# app.py
import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from eda import (
    split_columns,
    column_intelligence,
    data_quality_stats,
    data_quality_bullets,
    data_health_score,
    detect_quality_issues,
    plot_missingness_bar,
    dataset_overview,
    executive_summary,
    numeric_summary,
    top_categorical_insights,
    correlation_heatmap,
    correlation_pairs,
    redundancy_report,
    outlier_report,
    pii_detection_heuristics,
    recommendations_v2,
    auto_plots,
    data_dictionary,
)

# =========================
# Page Config (ONLY ONCE)
# =========================
st.set_page_config(
    page_title="Auto-EDA Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Dark UI polish + IMPORTANT FIX for "top cut off"
# =========================
st.markdown(
    """
    <style>
    /* --- FIX: Streamlit header overlap --- */
    /* This ensures content never sits under the top Streamlit toolbar/header */
    div[data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 3.75rem !important;   /* <-- key fix */
        padding-bottom: 2rem !important;
    }

    /* Optional: make header slightly transparent so it feels clean */
    header[data-testid="stHeader"] {
        background: rgba(11, 15, 20, 0.65) !important;
        backdrop-filter: blur(6px);
    }

    h1, h2, h3, h4 { font-family: "Inter", "Segoe UI", sans-serif; font-weight: 700; }
    p, li, div { font-size: 15.5px; line-height: 1.55; }
    .small-muted { color: #9BA3AF; font-size: 13px; }
    .pill { display:inline-block; padding:3px 10px; border-radius:999px; border:1px solid #30363D; margin-right:8px; }
    </style>
    """,
    unsafe_allow_html=True
)

def section_card(title: str, emoji: str = "", subtitle: str | None = None):
    heading = f"{title} {emoji}".strip()
    sub = f"<div class='small-muted'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div style="
            background-color:#161B22;
            padding:16px 16px 10px 16px;
            border-radius:14px;
            margin:10px 0 16px 0;
            border:1px solid #30363D;">
            <h3 style="margin:0 0 6px 0;">{heading}</h3>
            {sub}
        </div>
        """,
        unsafe_allow_html=True
    )

def card_row(title: str, why: str, risk: str, effort: str):
    st.markdown(
        f"""
        <div style="background:#0f141b;border:1px solid #30363D;border-radius:14px;padding:14px;margin:10px 0;">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
            <div style="font-weight:700;font-size:16px;">{title}</div>
            <div class="pill">{effort} effort</div>
          </div>
          <div class="small-muted" style="margin-top:8px;"><b>Why:</b> {why}</div>
          <div class="small-muted" style="margin-top:6px;"><b>Risk:</b> {risk}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Sidebar
# =========================
st.sidebar.title("All About your Dataset!")
st.sidebar.caption("Upload any CSV → get explainable, executive-grade EDA.")

PAGE = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard 🧭",
        "Preview 📌",
        "Quality & Risks ⚠️",
        "Insights 🔍",
        "Visuals 📈",
        "Column Inspector 🧠",
        "PII & Governance 🔐",
        "Data Dictionary 📚",
        "Export Center ⬇️",
    ],
    index=0
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================
# No file state
# =========================
if uploaded_file is None:
    st.title("Auto-EDA & Insights 📊")
    st.write("Upload a CSV file from the sidebar to begin.")
    st.info("Tip: Start with **Dashboard 🧭** after uploading.")
    st.stop()

# =========================
# Read CSV (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

df = load_csv(uploaded_file)

# =========================
# BIG: visible anchor/title AFTER upload (prevents weird “cut” feel)
# =========================
st.title("Auto-EDA & Insight Generation Platform 📊")
st.caption("Executive-grade EDA • Explainable insights • Governance-aware outputs")
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# =========================
# Cached computations
# =========================
@st.cache_data(show_spinner=False)
def cached_splits(dfx: pd.DataFrame):
    return split_columns(dfx)

@st.cache_data(show_spinner=False)
def cached_colinfo(dfx: pd.DataFrame):
    return column_intelligence(dfx)

@st.cache_data(show_spinner=False)
def cached_quality_stats(dfx: pd.DataFrame):
    return data_quality_stats(dfx)

@st.cache_data(show_spinner=False)
def cached_health(dfx: pd.DataFrame):
    return data_health_score(dfx)

@st.cache_data(show_spinner=False)
def cached_outliers(dfx: pd.DataFrame, numeric_cols: list[str]):
    return outlier_report(dfx, numeric_cols)

@st.cache_data(show_spinner=False)
def cached_redundancy(dfx: pd.DataFrame, numeric_cols: list[str]):
    return redundancy_report(dfx, numeric_cols, threshold=0.9)

@st.cache_data(show_spinner=False)
def cached_corr_pairs(dfx: pd.DataFrame, numeric_cols: list[str]):
    return correlation_pairs(dfx, numeric_cols, min_abs=0.5, max_pairs=25)

@st.cache_data(show_spinner=False)
def cached_pii(dfx: pd.DataFrame):
    return pii_detection_heuristics(dfx)

@st.cache_data(show_spinner=False)
def cached_recs(dfx: pd.DataFrame):
    return recommendations_v2(dfx)

numeric_cols, categorical_cols, datetime_cols = cached_splits(df)
col_info = cached_colinfo(df)
qstats = cached_quality_stats(df)
health = cached_health(df)

# Header strip (taskbar feel)
st.markdown(
    f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 16px 0;">
      <span class="pill">Rows: <b>{qstats['nrows']:,}</b></span>
      <span class="pill">Cols: <b>{qstats['ncols']:,}</b></span>
      <span class="pill">Missing: <b>{qstats['missing_overall_pct']:.2f}%</b></span>
      <span class="pill">Duplicates: <b>{qstats['dup_rows']:,}</b></span>
      <span class="pill">Health: <b>{health['score']}/100 ({health['grade']})</b></span>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# DASHBOARD
# =========================
if PAGE == "Dashboard 🧭":
    section_card("Executive Dashboard", "🧭", "A quick, boardroom-ready view of what matters.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Health Score", f"{health['score']}/100", health["label"])
    c2.metric("Rows", f"{qstats['nrows']:,}")
    c3.metric("Columns", f"{qstats['ncols']:,}")
    c4.metric("Missing %", f"{qstats['missing_overall_pct']:.2f}%")
    c5.metric("Duplicates", f"{qstats['dup_rows']:,}")

    st.markdown("---")

    left, right = st.columns([1.15, 1])

    with left:
        section_card("Top Risks", "⚠️", "The stuff that can break analysis if ignored.")
        issues = detect_quality_issues(df)
        if not issues:
            st.success("No major quality risks detected.")
        else:
            for it in issues[:8]:
                if it.level == "critical":
                    st.error(it.message)
                elif it.level == "warning":
                    st.warning(it.message)
                else:
                    st.info(it.message)

    with right:
        section_card("Missingness Snapshot", "🧩", "A fast visual of where the holes are.")
        fig = plot_missingness_bar(df, top_n=10)
        if fig is None:
            st.info("No missingness detected.")
        else:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")
    section_card("Next Best Actions", "✅", "What to fix first, then what to analyze.")
    rec = cached_recs(df)

    colA, colB, colC = st.columns(3)
    with colA:
        st.subheader("Fix Now ✅")
        if not rec["fix_now"]:
            st.success("No urgent fixes detected.")
        for r in rec["fix_now"][:4]:
            card_row(r["title"], r["why"], r["risk"], r["effort"])
    with colB:
        st.subheader("Analyze Next 🔍")
        for r in rec["analyze_next"][:4]:
            card_row(r["title"], r["why"], r["risk"], r["effort"])
    with colC:
        st.subheader("Modeling Ready 🧪")
        for r in rec["modeling_ready"][:4]:
            card_row(r["title"], r["why"], r["risk"], r["effort"])

# =========================
# PREVIEW
# =========================
elif PAGE == "Preview 📌":
    section_card("Preview", "📌", "A quick look at what you uploaded.")
    st.dataframe(df.head(50), use_container_width=True)

    section_card("Column Type Counts", "🧭", "How your dataset is shaped.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric", len(numeric_cols))
    c2.metric("Categorical/Text", len(categorical_cols))
    c3.metric("Date/Time", len(datetime_cols))

# =========================
# QUALITY
# =========================
elif PAGE == "Quality & Risks ⚠️":
    tabs = st.tabs(["Summary", "Missingness", "Flags", "Score"])

    with tabs[0]:
        section_card("Quality Summary", "⚠️", "Plain English + business risk.")
        for b in data_quality_bullets(df):
            st.markdown(f"- {b}")

    with tabs[1]:
        section_card("Missingness Chart", "🧩", "What’s missing and how much.")
        fig = plot_missingness_bar(df, top_n=10)
        if fig is None:
            st.info("No missingness detected.")
        else:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tabs[2]:
        section_card("Detailed Flags", "🔎", "Machine-generated alerts.")
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

    with tabs[3]:
        section_card("Data Health Score", "🧪", "A simple 0–100 health rating.")
        st.metric("Score", f"{health['score']}/100", f"Grade {health['grade']} • {health['label']}")
        st.markdown(
            "- **What it means:** Higher score = cleaner, more reliable dataset.\n"
            "- **What to do:** Improve missingness, duplicates, and constant/mostly-missing fields to boost score."
        )

# =========================
# INSIGHTS
# =========================
elif PAGE == "Insights 🔍":
    tabs = st.tabs(["Overview", "Categories", "Numeric", "Correlation", "Redundancy", "Outliers"])

    with tabs[0]:
        section_card("Dataset Overview", "🧾", "A readable story of what this dataset looks like.")
        for b in dataset_overview(df):
            st.markdown(f"- {b}")

        section_card("Executive Summary", "🧠", "Top-level bullets for stakeholders.")
        for b in executive_summary(df):
            st.markdown(f"- {b}")

    with tabs[1]:
        section_card("Category Insights", "🏷️", "What dominates and what’s imbalanced.")
        cat_insights = top_categorical_insights(df)
        if not cat_insights:
            st.info("No low-cardinality categorical columns found.")
        else:
            for col, info in cat_insights.items():
                with st.expander(f"{col}", expanded=False):
                    st.dataframe(info["distribution"], use_container_width=True)
                    for b in info["insights"]:
                        st.markdown(f"- {b}")
                    st.caption("Dominance can skew analysis. Consider grouping rare categories + validating imbalance.")

    with tabs[2]:
        section_card("Numeric Summary", "📐", "Stats for numeric columns.")
        ns = numeric_summary(df, numeric_cols)
        if ns.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(ns, use_container_width=True)

    with tabs[3]:
        section_card("Correlation", "🔗", "How numeric columns move together.")
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation.")
        else:
            heat = correlation_heatmap(df, numeric_cols)
            if heat is not None:
                st.pyplot(heat, use_container_width=True)
                plt.close(heat)

            pairs = cached_corr_pairs(df, numeric_cols)
            if pairs.empty:
                st.info("No strong correlations detected (|corr| ≥ 0.5).")
            else:
                st.dataframe(pairs, use_container_width=True)
                st.caption("Correlation near ±1.0 means the two columns move together. Check redundancy/leakage.")

    with tabs[4]:
        section_card("Redundancy Detector", "🧩", "Flags near-duplicate numeric fields.")
        red = cached_redundancy(df, numeric_cols)
        if red.empty:
            st.info("No high-redundancy pairs detected (|corr| ≥ 0.9).")
        else:
            st.dataframe(red, use_container_width=True)

    with tabs[5]:
        section_card("Outliers (IQR)", "📍", "Explained + actionable.")
        rep = cached_outliers(df, numeric_cols)
        if rep.empty:
            st.info("No numeric columns (or not enough data) for outlier report.")
        else:
            st.dataframe(rep, use_container_width=True)
            st.caption("IQR outliers are values far beyond the typical range. They can be rare events or errors.")

# =========================
# VISUALS
# =========================
elif PAGE == "Visuals 📈":
    section_card("Visual Insights", "📈", "Auto-selected charts that actually help.")

    plot_mode = st.radio("Mode", ["Auto-selected", "Pick columns"], horizontal=True)

    if plot_mode == "Auto-selected":
        plots = auto_plots(df)
        if not plots.get("numeric") and not plots.get("categorical"):
            st.info("No suitable columns found for auto visuals.")
        else:
            grid = st.columns(2)
            i = 0
            for col, fig in plots.get("numeric", {}).items():
                with grid[i % 2]:
                    st.markdown(f"**Distribution: {col}**")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption("Spikes = common values; long tail = skew; isolated bars = potential outliers.")
                i += 1

            for col, fig in plots.get("categorical", {}).items():
                with grid[i % 2]:
                    st.markdown(f"**Category share: {col}**")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption("One huge bar = dominance. Consider grouping rare categories.")
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

            fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=160)
            vc.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"{col}")
            ax.set_xlabel("Percent (%)" if normalize else "Count")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        for col in pick_num:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 20:
                st.info(f"Not enough values to plot: {col}")
                continue

            fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=160)
            ax.hist(s, bins=30)
            ax.set_title(f"{col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# =========================
# COLUMN INSPECTOR
# =========================
elif PAGE == "Column Inspector 🧠":
    section_card("Column Inspector", "🧠", "Search, filter, and understand columns without drowning.")

    search = st.text_input("Search column name", "")
    type_filter = st.multiselect(
        "Filter by detected type",
        options=sorted({meta["type"] for meta in col_info.values()}),
        default=[]
    )
    miss_min = st.slider("Minimum missingness (%)", 0, 100, 0)
    uniq_max = st.slider("Maximum unique values (optional)", 0, 5000, 5000)

    cols = []
    for c, meta in col_info.items():
        if search and search.lower() not in c.lower():
            continue
        if type_filter and meta["type"] not in type_filter:
            continue
        if meta["signals"]["missing_pct"] < miss_min:
            continue
        if meta["signals"]["nunique"] > uniq_max:
            continue
        cols.append(c)

    st.caption(f"Showing {len(cols)} columns.")
    for c in cols:
        meta = col_info[c]
        with st.expander(f"{c} — {meta['type']}", expanded=False):
            st.markdown("**Quick meaning**")
            for b in meta["insights"]:
                st.markdown(f"- {b}")
            if meta.get("sample_values"):
                st.markdown("**Sample values**")
                st.write(meta["sample_values"])

# =========================
# PII & Governance
# =========================
elif PAGE == "PII & Governance 🔐":
    section_card("PII Heuristics", "🔐", "Not perfect detection — a safety alert layer (bank-friendly).")

    pii = cached_pii(df)
    if pii.empty:
        st.success("No obvious PII-like signals detected by heuristics.")
    else:
        st.warning("Potential sensitive fields detected. Review before exporting/sharing.")
        st.dataframe(pii, use_container_width=True)
        st.caption("Mask/redact, restrict downloads, and follow governance rules.")

# =========================
# DATA DICTIONARY
# =========================
elif PAGE == "Data Dictionary 📚":
    section_card("Data Dictionary", "📚", "One row per column: type, missingness, samples, suggested treatment.")
    dd = data_dictionary(df)
    st.dataframe(dd, use_container_width=True)

# =========================
# EXPORT CENTER
# =========================
elif PAGE == "Export Center ⬇️":
    section_card("Export Center", "⬇️", "Download outputs like a real enterprise tool.")

    dd = data_dictionary(df)
    outliers_df = cached_outliers(df, numeric_cols)
    red = cached_redundancy(df, numeric_cols)
    pairs = cached_corr_pairs(df, numeric_cols)
    pii = cached_pii(df)
    rec = cached_recs(df)

    st.download_button(
        "Download Data Dictionary (CSV) 📚",
        dd.to_csv(index=False).encode("utf-8"),
        file_name="data_dictionary.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Outlier Report (CSV) 📍",
        (outliers_df.to_csv(index=False).encode("utf-8") if not outliers_df.empty else b""),
        file_name="outlier_report.csv",
        mime="text/csv",
        disabled=outliers_df.empty
    )

    st.download_button(
        "Download Correlation Pairs (CSV) 🔗",
        (pairs.to_csv(index=False).encode("utf-8") if not pairs.empty else b""),
        file_name="correlation_pairs.csv",
        mime="text/csv",
        disabled=pairs.empty
    )

    st.download_button(
        "Download Redundancy Report (CSV) 🧩",
        (red.to_csv(index=False).encode("utf-8") if not red.empty else b""),
        file_name="redundancy_report.csv",
        mime="text/csv",
        disabled=red.empty
    )

    st.download_button(
        "Download PII Heuristics (CSV) 🔐",
        (pii.to_csv(index=False).encode("utf-8") if not pii.empty else b""),
        file_name="pii_heuristics.csv",
        mime="text/csv",
        disabled=pii.empty
    )

    report = {
        "quality_stats": qstats,
        "health": health,
        "recommendations_v2": rec,
    }
    st.download_button(
        "Download Quality + Recommendations (JSON) ✅",
        json.dumps(report, indent=2).encode("utf-8"),
        file_name="eda_report.json",
        mime="application/json"
    )
