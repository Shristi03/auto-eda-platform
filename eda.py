# eda.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Matplotlib defaults (Streamlit sizing)
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


# ============================
# Helpers
# ============================
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

    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]
    return numeric_cols, categorical_cols, datetime_cols


def _is_id_like(series: pd.Series) -> bool:
    n = max(len(series), 1)
    missing_pct = float(series.isna().mean() * 100)
    if missing_pct >= 5:
        return False
    nunique = int(series.nunique(dropna=True))
    if nunique <= 20:
        return False
    uniq_ratio = nunique / n
    return uniq_ratio > 0.9


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


# ============================
# Quality
# ============================
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


def data_quality_stats(df: pd.DataFrame) -> dict:
    nrows, ncols = df.shape
    dup_rows = int(df.duplicated().sum())

    missing_overall_pct = 0.0
    if nrows > 0 and ncols > 0:
        missing_overall_pct = float(df.isna().sum().sum() / (nrows * ncols) * 100)

    numeric_cols, categorical_cols, datetime_cols = split_columns(df)
    info = column_intelligence(df)
    id_cols = [c for c, meta in info.items() if meta["type"] == "identifier"]
    constant_cols = [c for c, meta in info.items() if meta["type"] == "constant"]
    mostly_missing_cols = [c for c, meta in info.items() if meta["type"] == "mostly-missing"]

    return {
        "nrows": int(nrows),
        "ncols": int(ncols),
        "dup_rows": int(dup_rows),
        "missing_overall_pct": float(round(missing_overall_pct, 2)),
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
        "datetime_cols": len(datetime_cols),
        "id_cols": len(id_cols),
        "constant_cols": len(constant_cols),
        "mostly_missing_cols": len(mostly_missing_cols),
    }


def data_health_score(df: pd.DataFrame) -> dict:
    """
    Simple 0–100 score. Higher = healthier.
    Penalizes missingness, duplicates, constant cols, mostly-missing cols, and heavy outliers.
    """
    stats = data_quality_stats(df)
    nrows = stats["nrows"]
    ncols = stats["ncols"]

    score = 100.0

    # Missingness penalty (overall)
    score -= min(stats["missing_overall_pct"] * 0.6, 35)

    # Duplicates penalty
    if nrows > 0:
        dup_rate = stats["dup_rows"] / max(nrows, 1)
        score -= min(dup_rate * 100 * 0.4, 15)

    # Constant / mostly-missing penalties
    if ncols > 0:
        score -= min((stats["constant_cols"] / ncols) * 100 * 0.25, 10)
        score -= min((stats["mostly_missing_cols"] / ncols) * 100 * 0.35, 15)

    score = float(max(0.0, min(100.0, score)))

    # Grade
    if score >= 85:
        grade = "A"
        label = "Strong"
    elif score >= 70:
        grade = "B"
        label = "Good"
    elif score >= 55:
        grade = "C"
        label = "Needs attention"
    else:
        grade = "D"
        label = "High risk"

    return {"score": round(score, 1), "grade": grade, "label": label}


def data_quality_bullets(df: pd.DataFrame) -> List[str]:
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

    info = column_intelligence(df)
    constant_cols = [c for c, meta in info.items() if meta["type"] == "constant"]
    if constant_cols:
        bullets.append(
            "Constant columns (usually safe to drop): " +
            ", ".join([f"**{c}**" for c in constant_cols[:8]]) +
            ("..." if len(constant_cols) > 8 else "")
        )

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


# ============================
# Summaries
# ============================
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


# ============================
# Correlation + Redundancy
# ============================
def correlation_pairs(df: pd.DataFrame, numeric_cols: List[str], min_abs: float = 0.5, max_pairs: int = 25) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            v = corr.loc[a, b]
            if pd.isna(v):
                continue
            if abs(v) >= min_abs:
                pairs.append((a, b, float(v), float(abs(v))))

    if not pairs:
        return pd.DataFrame()

    out = pd.DataFrame(pairs, columns=["col_a", "col_b", "corr", "abs_corr"]).sort_values("abs_corr", ascending=False)
    return out.head(max_pairs).reset_index(drop=True)


def correlation_insights(df: pd.DataFrame, numeric_cols: List[str], top_k: int = 6) -> List[str]:
    pairs = correlation_pairs(df, numeric_cols, min_abs=0.5, max_pairs=top_k)
    if pairs.empty:
        return []

    out: List[str] = []
    for _, r in pairs.iterrows():
        a, b, v = r["col_a"], r["col_b"], r["corr"]
        out.append(f"Strong relationship: **{a} ↔ {b}** (corr ≈ {v:.2f}). Consider redundancy/leakage.")
    return out


def redundancy_report(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.9) -> pd.DataFrame:
    pairs = correlation_pairs(df, numeric_cols, min_abs=threshold, max_pairs=200)
    if pairs.empty:
        return pd.DataFrame()

    def _label(v: float) -> str:
        if abs(v) >= 0.97:
            return "Near-duplicate"
        if abs(v) >= 0.93:
            return "Very high redundancy"
        return "High redundancy"

    pairs["risk_label"] = pairs["corr"].apply(_label)
    pairs["why_it_matters"] = "Two fields move together. Keeping both can add noise or multicollinearity."
    pairs["what_to_do"] = "Keep one, or combine them, or validate if one leaks target info."
    return pairs.reset_index(drop=True)


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


# ============================
# Outliers
# ============================
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
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "min": float(s.min()),
            "max": float(s.max()),
            "what_it_means": "Outliers are values far from most of the data (could be rare events or errors).",
            "what_to_do": "Validate with business context; consider capping/winsorizing only if justified."
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["outlier_pct", "outlier_count"], ascending=False).reset_index(drop=True)


# ============================
# PII heuristics
# ============================
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PHONE_RE = re.compile(r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
_SSN_RE = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")
_CC_RE = re.compile(r"^\d{13,19}$")

_PII_NAME_HINTS = [
    "email", "e-mail", "mail",
    "phone", "mobile", "contact",
    "ssn", "social", "taxid", "tin",
    "dob", "birth",
    "address", "street", "zip", "zipcode", "postal",
    "passport", "pan",
    "credit", "card",
    "name", "first", "last",
    "account", "acct",
]

def pii_detection_heuristics(df: pd.DataFrame, sample_n: int = 200) -> pd.DataFrame:
    """
    Safe heuristics: flags columns that *look* like PII.
    This is NOT perfect detection — it's an alerting layer.
    """
    rows = []
    for col in df.columns:
        col_l = col.lower()
        name_hint = any(h in col_l for h in _PII_NAME_HINTS)

        s = df[col]
        s2 = s.dropna()
        if s2.empty:
            continue

        # sample
        sample = s2.astype(str).head(sample_n).tolist()

        email_hits = sum(1 for v in sample if _EMAIL_RE.match(v.strip()))
        phone_hits = sum(1 for v in sample if _PHONE_RE.match(v.strip()))
        ssn_hits = sum(1 for v in sample if _SSN_RE.match(v.strip()))

        # credit-card-like: numeric strings length 13-19 (we DO NOT validate luhn here)
        cc_hits = 0
        for v in sample:
            vv = re.sub(r"\s|-", "", v.strip())
            if vv.isdigit() and _CC_RE.match(vv):
                cc_hits += 1

        hit_total = email_hits + phone_hits + ssn_hits + cc_hits

        # score
        score = 0
        if name_hint:
            score += 2
        if email_hits > 0:
            score += 3
        if phone_hits > 0:
            score += 2
        if ssn_hits > 0:
            score += 4
        if cc_hits > 0:
            score += 4

        if score <= 0:
            continue

        risk = "Low"
        if score >= 6:
            risk = "High"
        elif score >= 3:
            risk = "Medium"

        rows.append({
            "column": col,
            "name_hint": bool(name_hint),
            "email_like_hits": int(email_hits),
            "phone_like_hits": int(phone_hits),
            "ssn_like_hits": int(ssn_hits),
            "card_like_hits": int(cc_hits),
            "risk": risk,
            "what_to_do": "Mask/redact before sharing; confirm governance policy; restrict exports."
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["risk", "ssn_like_hits", "card_like_hits", "email_like_hits"], ascending=[True, False, False, False])
    # put High first
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    out["risk_rank"] = out["risk"].map(risk_order).fillna(9).astype(int)
    out = out.sort_values(["risk_rank"]).drop(columns=["risk_rank"]).reset_index(drop=True)
    return out


# ============================
# Recommendations Engine v2
# ============================
def recommendations_v2(df: pd.DataFrame) -> dict:
    """
    Returns structured cards:
    { "fix_now": [card...], "analyze_next": [...], "modeling_ready": [...] }
    """
    info = column_intelligence(df)
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    def card(title: str, why: str, risk: str, effort: str = "Low") -> dict:
        return {"title": title, "why": why, "risk": risk, "effort": effort}

    out = {"fix_now": [], "analyze_next": [], "modeling_ready": []}

    # duplicates
    dup = int(df.duplicated().sum())
    if dup > 0:
        out["fix_now"].append(card(
            title=f"Resolve {dup:,} duplicate rows",
            why="Duplicates can inflate counts and bias averages/correlations.",
            risk="You may draw the wrong conclusions (or build unstable models).",
            effort="Medium"
        ))

    # missingness
    high_missing = [(c, info[c]["signals"]["missing_pct"]) for c in info if info[c]["signals"]["missing_pct"] >= 40]
    high_missing.sort(key=lambda x: x[1], reverse=True)
    for c, p in high_missing[:8]:
        out["fix_now"].append(card(
            title=f"Fix high missingness in {c} ({p:.1f}%)",
            why="High missingness can distort analysis and reduce usable sample size.",
            risk="Downstream metrics/models may become biased or unreliable.",
            effort="High"
        ))

    # constant
    constant_cols = [c for c, meta in info.items() if meta["type"] == "constant"]
    if constant_cols:
        out["fix_now"].append(card(
            title="Drop constant columns",
            why="They add no signal and can clutter analysis.",
            risk="Noise and wasted effort; sometimes breaks encoding pipelines.",
            effort="Low"
        ))

    # category dominance
    low_card_cats = [c for c, meta in info.items() if meta["type"] == "categorical/text" and meta["signals"]["nunique"] <= 12]
    if low_card_cats:
        out["analyze_next"].append(card(
            title="Check category dominance / imbalance",
            why="If one category dominates, comparisons and models can be biased.",
            risk="You may overfit to dominant groups and miss minority behavior.",
            effort="Low"
        ))

    # numeric dist/outliers
    if numeric_cols:
        out["analyze_next"].append(card(
            title="Review numeric distributions and outliers",
            why="Skew and outliers change averages and correlations.",
            risk="Metrics can be misleading; models may become unstable.",
            effort="Medium"
        ))

    # redundancy
    red = redundancy_report(df, numeric_cols, threshold=0.9)
    if not red.empty:
        out["analyze_next"].append(card(
            title="Investigate highly correlated features (redundancy)",
            why="Very high correlation can mean duplicate or leaking features.",
            risk="Multicollinearity and leakage can inflate performance unrealistically.",
            effort="Medium"
        ))

    # pii
    pii = pii_detection_heuristics(df)
    if not pii.empty:
        out["fix_now"].append(card(
            title="Review potential sensitive/PII-like columns",
            why="Financial environments require strict data handling policies.",
            risk="Compliance and privacy risk if shared/exported improperly.",
            effort="Low"
        ))

    # modeling readiness
    id_cols = [c for c, meta in info.items() if meta["type"] == "identifier"]
    if id_cols:
        out["modeling_ready"].append(card(
            title="Exclude ID-like columns from features",
            why="IDs usually don’t generalize and can cause leakage.",
            risk="Overfitting and poor real-world performance.",
            effort="Low"
        ))

    if datetime_cols:
        out["modeling_ready"].append(card(
            title="Engineer time features from date columns",
            why="Time fields enable trends, seasonality, and time-based grouping.",
            risk="You may miss time-driven patterns.",
            effort="Medium"
        ))

    out["modeling_ready"].append(card(
        title="Document assumptions + preprocessing choices",
        why="Explainability and governance depend on clear documentation.",
        risk="Hard to audit or reproduce results later.",
        effort="Low"
    ))

    return out


# ============================
# Visuals
# ============================
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


# ============================
# Data Dictionary
# ============================
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
            "unique_values": meta["signals"]["nunique"],st
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
    return dd.sort_values(["missing_pct", "unique_values"], ascending=[False, False]).reset_index(drop=True)
