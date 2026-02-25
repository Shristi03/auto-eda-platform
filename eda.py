# EDA.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Matplotlib defaults (important for Streamlit sizing)
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


@dataclass
class DataQualityIssue:
    level: str  # "info" | "warning" | "critical"
    message: str


# ----------------------------
# Helpers
# ----------------------------
def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    datetime_cols: List[str] = []
    categorical_cols: List[str] = non_numeric.copy()

    # best-effort datetime detection from object/string columns
    for c in non_numeric:
        s = df[c]
        if s.dtype == "object" or "string" in str(s.dtype):
            sample = s.dropna().astype(str).head(50)
            if len(sample) >= 5:
                try:
                    parsed = pd.to_datetime(sample, errors="raise")
                    if parsed.notna().all():
                        datetime_cols.append(c)
                except Exception:
                    pass

    # remove datetime from categorical
    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]

    return numeric_cols, categorical_cols, datetime_cols


def column_intelligence(df: pd.DataFrame, max_unique_preview: int = 5) -> dict:
    result: Dict[str, dict] = {}
    n = len(df)

    for col in df.columns:
        s = df[col]
        missing_pct = float(s.isna().mean() * 100)
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)

        # date-like check
        date_like = False
        if s.dtype == "object" or "string" in dtype:
            sample = s.dropna().astype(str).head(50)
            if len(sample) >= 5:
                try:
                    parsed = pd.to_datetime(sample, errors="raise")
                    date_like = parsed.notna().all()
                except Exception:
                    date_like = False

        # id-like check
        id_like = False
        if n > 0 and missing_pct < 5:
            uniq_ratio = nunique / max(n, 1)
            if uniq_ratio > 0.9 and nunique > 20:
                id_like = True

        constant_like = nunique <= 1

        if missing_pct >= 60:
            col_type = "mostly-missing"
        elif constant_like:
            col_type = "constant"
        elif date_like:
            col_type = "date/time"
        elif id_like:
            col_type = "identifier"
        elif pd.api.types.is_numeric_dtype(s):
            col_type = "numeric"
        else:
            col_type = "categorical/text"

        sample_vals = s.dropna().astype("string").head(max_unique_preview).tolist()

        insights: List[str] = []
        insights.append(f"Detected type: **{col_type}** (dtype: {dtype}).")
        insights.append(f"Missing values: **{missing_pct:.1f}%**.")
        insights.append(f"Unique values: **{nunique:,}**.")

        if col_type == "mostly-missing":
            insights.append("Heavily missing → consider dropping or investigate upstream source.")
        if col_type == "constant":
            insights.append("No variation → usually safe to drop (adds no analytical signal).")
        if col_type == "identifier":
            insights.append("ID/key-like → good for joins/audit trails, usually not predictive.")
        if col_type == "categorical/text":
            if nunique <= 20:
                insights.append("Category field → check dominance/imbalance and rare categories.")
            else:
                insights.append("High-cardinality text/category → consider grouping/encoding or NLP if needed.")
        if col_type == "numeric":
            s_num = pd.to_numeric(s, errors="coerce")
            valid = s_num.dropna()
            if len(valid) >= 5:
                mn, mx = float(valid.min()), float(valid.max())
                sd = float(valid.std())
                insights.append(f"Range: **{mn:.3g} → {mx:.3g}**, Std Dev: **{sd:.3g}**.")
                if valid.nunique() <= 3:
                    insights.append("Very few unique numeric values → behaves like a category.")
                elif abs(sd) < 1e-12:
                    insights.append("No spread → treat as constant.")
        if col_type == "date/time":
            insights.append("Time field → enables trend/seasonality analysis and time-based grouping.")

        result[col] = {
            "type": col_type,
            "signals": {"missing_pct": missing_pct, "nunique": nunique, "dtype": dtype},
            "insights": insights,
            "sample_values": sample_vals,
        }

    return result


# ----------------------------
# Quality
# ----------------------------
def detect_quality_issues(df: pd.DataFrame) -> List[DataQualityIssue]:
    issues: List[DataQualityIssue] = []

    if df.shape[0] == 0 or df.shape[1] == 0:
        issues.append(DataQualityIssue("critical", "Dataset is empty."))
        return issues

    dup = int(df.duplicated().sum())
    if dup > 0:
        issues.append(DataQualityIssue("warning", f"{dup:,} duplicate rows found. Confirm if duplicates are valid."))

    miss_pct = df.isna().mean() * 100
    high = miss_pct[miss_pct >= 40].sort_values(ascending=False)
    med = miss_pct[(miss_pct >= 20) & (miss_pct < 40)].sort_values(ascending=False)

    for col, pct in high.items():
        issues.append(DataQualityIssue("critical", f"High missingness: '{col}' is {pct:.1f}% missing."))

    for col, pct in med.items():
        issues.append(DataQualityIssue("warning", f"Missingness: '{col}' is {pct:.1f}% missing."))

    for col in df.columns:
        nun = int(df[col].nunique(dropna=True))
        if nun <= 1:
            issues.append(DataQualityIssue("info", f"Column '{col}' is constant (1 unique value)."))

    return issues


def data_quality_bullets(df: pd.DataFrame) -> List[str]:
    """
    Business-friendly summary of quality risks.
    """
    bullets: List[str] = []
    nrows, ncols = df.shape
    bullets.append(f"Dataset size: **{nrows:,} rows × {ncols:,} columns**.")

    miss_pct = df.isna().mean() * 100
    worst = miss_pct.sort_values(ascending=False)
    top_missing = worst[worst >= 20].head(5)

    if not top_missing.empty:
        bullets.append(
            "High missingness (can bias analysis): " +
            ", ".join([f"**{c} ({p:.1f}%)**" for c, p in top_missing.items()])
        )
    else:
        bullets.append("Missingness looks manageable (no column ≥ 20% missing).")

    dup = int(df.duplicated().sum())
    bullets.append(f"Duplicate rows: **{dup:,}**." if dup > 0 else "No duplicate rows detected.")

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        bullets.append(
            "Constant columns (usually safe to drop): " +
            ", ".join([f"**{c}**" for c in constant_cols[:8]]) +
            ("..." if len(constant_cols) > 8 else "")
        )

    info = column_intelligence(df)
    id_cols = [c for c, meta in info.items() if meta["type"] == "identifier"]
    if id_cols:
        bullets.append(
            "Likely identifier fields (joins/audit trails, not predictive): " +
            ", ".join([f"**{c}**" for c in id_cols[:8]]) +
            ("..." if len(id_cols) > 8 else "")
        )

    bullets.append("Fix order: resolve high-missing fields → confirm dataset grain → drop constant/ID-like fields → then model/segment.")
    return bullets


def plot_missingness_bar(df: pd.DataFrame, top_n: int = 10):
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    if miss.empty:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=160)
    miss.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top Missing Columns (%)", fontsize=11)
    ax.set_xlabel("Missing %", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    return fig


# ----------------------------
# Summaries
# ----------------------------
def numeric_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()

    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isna().sum()
    desc["missing_pct"] = (df[numeric_cols].isna().mean() * 100).round(2)
    desc["skew"] = df[numeric_cols].skew(numeric_only=True)
    return desc.reset_index().rename(columns={"index": "column"})


def dataset_overview(df: pd.DataFrame) -> List[str]:
    nrows, ncols = df.shape
    bullets = [f"This dataset has **{nrows:,} rows** and **{ncols:,} columns**."]

    col_info = column_intelligence(df)
    type_counts: Dict[str, int] = {}
    for _, info in col_info.items():
        t = info["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    bullets.append(
        f"Detected: "
        f"**{type_counts.get('numeric', 0)} numeric**, "
        f"**{type_counts.get('categorical/text', 0)} categorical/text**, "
        f"**{type_counts.get('date/time', 0)} date/time**."
    )

    id_cols = [c for c, info in col_info.items() if info["type"] == "identifier"]
    if id_cols:
        bullets.append("Likely **identifier** columns: " + ", ".join(id_cols[:6]) + ("..." if len(id_cols) > 6 else ""))

    missing_cols = sorted(
        [(c, col_info[c]["signals"]["missing_pct"]) for c in col_info],
        key=lambda x: x[1],
        reverse=True,
    )
    top_missing = [(c, p) for c, p in missing_cols if p >= 20][:3]
    if top_missing:
        bullets.append("Highest missingness columns: " + ", ".join([f"**{c} ({p:.1f}%)**" for c, p in top_missing]) + ".")

    const_cols = [c for c, info in col_info.items() if info["type"] == "constant"]
    if const_cols:
        bullets.append("Constant columns (safe to drop): " + ", ".join(const_cols[:6]) + ("..." if len(const_cols) > 6 else ""))

    dup = int(df.duplicated().sum())
    if dup > 0:
        bullets.append(f"Duplicate rows detected: **{dup:,}** (confirm business rules).")

    bullets.append("Recommended next steps:")
    bullets.append("• Confirm dataset **grain** (one row per what?).")
    bullets.append("• Decide the **goal**: monitoring, segmentation, forecasting, or prediction.")
    bullets.append("• Treat high-missing fields carefully (drop, impute, or fix upstream).")
    return bullets


def executive_summary(df: pd.DataFrame) -> List[str]:
    bullets: List[str] = [f"Dataset has {df.shape[0]:,} rows and {df.shape[1]:,} columns."]

    miss = (df.isna().mean() * 100)
    high_miss = miss[miss >= 20].sort_values(ascending=False)
    if not high_miss.empty:
        top = ", ".join([f"{c} ({high_miss[c]:.1f}%)" for c in high_miss.index[:4]])
        bullets.append(f"Missingness risk: {top}" + ("..." if len(high_miss) > 4 else ""))

    dup = int(df.duplicated().sum())
    if dup > 0:
        bullets.append(f"{dup:,} duplicate rows found → confirm whether duplicates are valid events or data issues.")

    col_info = column_intelligence(df)
    possible_targets = [
        c for c, info in col_info.items()
        if info["type"] in ("categorical/text", "numeric")
        and info["signals"]["missing_pct"] < 10
        and info["signals"]["nunique"] in (2, 3, 4, 5)
    ]
    if possible_targets:
        bullets.append(
            "Possible outcome/label columns (low-cardinality): " +
            ", ".join(possible_targets[:3]) +
            ("..." if len(possible_targets) > 3 else "")
        )

    bullets.append("Validate outliers: separate true rare events from data entry/system errors.")
    bullets.append("For governance: identify sensitive/PII-like columns before sharing/exporting.")
    return bullets


def top_categorical_insights(df: pd.DataFrame, max_cols: int = 6, top_n: int = 8) -> dict:
    insights: Dict[str, dict] = {}
    col_info = column_intelligence(df)

    candidates = [
        c for c, info in col_info.items()
        if info["type"] == "categorical/text" and info["signals"]["nunique"] <= 20
    ][:max_cols]

    for col in candidates:
        vc = (
            df[col]
            .fillna("<<MISSING>>")
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .head(top_n)
        )

        bullets: List[str] = []
        top_val = str(vc.index[0])
        top_pct = float(vc.iloc[0])

        if top_pct >= 70:
            bullets.append(f"One category dominates: **{top_val} ({top_pct:.1f}%)** → highly imbalanced.")
        elif top_pct >= 40:
            bullets.append("A small number of categories dominate the distribution.")
        else:
            bullets.append("Categories are relatively balanced.")

        insights[col] = {
            "distribution": vc.reset_index().rename(columns={"index": col, col: "percent"}),
            "insights": bullets,
        }

    return insights


# ----------------------------
# Correlation + Outliers
# ----------------------------
def correlation_insights(df: pd.DataFrame, numeric_cols: List[str], top_k: int = 6) -> List[str]:
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr(numeric_only=True).abs()
    np.fill_diagonal(corr.values, 0.0)

    pairs = []
    for i, a in enumerate(numeric_cols):
        for j, b in enumerate(numeric_cols):
            if j <= i:
                continue
            pairs.append((a, b, float(corr.loc[a, b])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = [p for p in pairs if not np.isnan(p[2]) and p[2] >= 0.5][:top_k]

    out: List[str] = []
    for a, b, v in pairs:
        out.append(f"Strong relationship: **{a} ↔ {b}** (|corr| ≈ {v:.2f}). Consider redundancy/leakage.")
    return out


def correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 12):
    if len(numeric_cols) < 2:
        return None

    use_cols = numeric_cols[:max_cols]
    corr = df[use_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=160)
    im = ax.imshow(corr.values)
    ax.set_title("Correlation Heatmap (numeric)", fontsize=11)

    ax.set_xticks(range(len(use_cols)))
    ax.set_yticks(range(len(use_cols)))
    ax.set_xticklabels(use_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(use_cols, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def outlier_report(df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 10) -> pd.DataFrame:
    rows = []
    for col in numeric_cols[:max_cols]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 30:
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        out = s[(s < lo) | (s > hi)]

        rows.append({
            "column": col,
            "outlier_count": int(out.shape[0]),
            "outlier_pct": round(float(out.shape[0] / max(len(s), 1) * 100), 2),
            "min": float(s.min()),
            "max": float(s.max()),
            "notes": "High outliers may be true rare events or data errors."
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["outlier_pct", "outlier_count"], ascending=False)


# ----------------------------
# Recommendations engine
# ----------------------------
def recommendations(df: pd.DataFrame) -> dict:
    actions = {"fix_now": [], "analyze_next": [], "modeling_ready": []}
    info = column_intelligence(df)

    dup = int(df.duplicated().sum())
    if dup > 0:
        actions["fix_now"].append(f"Remove/resolve **{dup:,} duplicate rows** (define business de-dup rules).")

    high_missing = [(c, info[c]["signals"]["missing_pct"]) for c in info if info[c]["signals"]["missing_pct"] >= 40]
    high_missing.sort(key=lambda x: x[1], reverse=True)
    for c, p in high_missing[:8]:
        actions["fix_now"].append(f"**{c}** has **{p:.1f}% missing** → fix upstream, drop, or impute (document why).")

    constant_cols = [c for c, meta in info.items() if meta["type"] == "constant"]
    if constant_cols:
        actions["fix_now"].append(
            "Drop constant columns: " +
            ", ".join([f"**{c}**" for c in constant_cols[:10]]) +
            ("..." if len(constant_cols) > 10 else "")
        )

    low_card_cats = [c for c, meta in info.items() if meta["type"] == "categorical/text" and meta["signals"]["nunique"] <= 12]
    if low_card_cats:
        actions["analyze_next"].append(
            "Start with category dominance/imbalance for: " +
            ", ".join([f"**{c}**" for c in low_card_cats[:8]])
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        actions["analyze_next"].append("Check outliers + distribution for numeric columns (histogram + IQR report).")

    id_cols = [c for c, meta in info.items() if meta["type"] == "identifier"]
    if id_cols:
        actions["modeling_ready"].append(
            "Exclude identifier columns from features: " +
            ", ".join([f"**{c}**" for c in id_cols[:10]]) +
            ("..." if len(id_cols) > 10 else "")
        )

    date_cols = [c for c, meta in info.items() if meta["type"] == "date/time"]
    if date_cols:
        actions["modeling_ready"].append(
            "Parse date/time columns and create time-features (month, day-of-week, trend)."
        )

    actions["modeling_ready"].append(
        "Before modeling: split train/test, handle missingness consistently, encode categoricals (one-hot/target), and document assumptions."
    )

    return actions


# ----------------------------
# Visuals
# ----------------------------
def auto_plots(df: pd.DataFrame, max_numeric: int = 3, max_categorical: int = 4) -> Dict[str, Dict[str, plt.Figure]]:
    plots: Dict[str, Dict[str, plt.Figure]] = {"numeric": {}, "categorical": {}}

    numeric_cols, _, _ = split_columns(df)

    for col in numeric_cols[: max_numeric * 2]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 20:
            continue
        if s.nunique() <= 5:
            continue

        fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=160)
        ax.hist(s, bins=30)
        ax.set_title(f"Distribution: {col}", fontsize=10)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        fig.tight_layout()
        plots["numeric"][col] = fig

        if len(plots["numeric"]) >= max_numeric:
            break

    info = column_intelligence(df)
    cat_cols = [
        c for c, meta in info.items()
        if meta["type"] == "categorical/text" and meta["signals"]["nunique"] <= 12
    ][: max_categorical * 2]

    for col in cat_cols:
        vc = df[col].fillna("<<MISSING>>").value_counts(normalize=True).mul(100).round(1).head(8)
        if len(vc) < 2:
            continue

        fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=160)
        vc.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Category share: {col}", fontsize=10)
        ax.set_xlabel("Percent (%)", fontsize=9)
        ax.set_ylabel("")
        fig.tight_layout()
        plots["categorical"][col] = fig

        if len(plots["categorical"]) >= max_categorical:
            break

    return plots


# ----------------------------
# Data Dictionary
# ----------------------------
def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    info = column_intelligence(df)
    rows = []

    for col, meta in info.items():
        col_type = meta["type"]
        rows.append({
            "column": col,
            "detected_type": col_type,
            "dtype": meta["signals"]["dtype"],
            "missing_pct": round(meta["signals"]["missing_pct"], 2),
            "unique_values": meta["signals"]["nunique"],
            "sample_values": ", ".join([str(x) for x in meta["sample_values"][:5]]),
            "suggested_treatment": (
                "Drop" if col_type in ("constant", "mostly-missing") else
                "Exclude from features (ID)" if col_type == "identifier" else
                "Parse + feature engineering" if col_type == "date/time" else
                "Encode / group categories" if col_type == "categorical/text" else
                "Scale/check outliers" if col_type == "numeric" else
                "Review"
            )
        })

    dd = pd.DataFrame(rows)
    return dd.sort_values(["missing_pct", "unique_values"], ascending=[False, False])
