from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent / "data"

DIMENSION_METRICS = {
    "Consistency": ["C_out", "C_traj_d", "C_traj_s", "C_res"],
    "Predictability": ["P_cal", "P_auroc", "P_brier"],
    "Robustness": ["R_fault", "R_struct", "R_prompt"],
    "Safety": ["S_harm", "S_comp", "S_safety"],
}

# Dimensions included in R_Overall (Safety is reported separately)
_OVERALL_DIMS = ["Consistency", "Predictability", "Robustness"]

ABSTENTION_METRICS = ["A_rate", "A_prec", "A_rec", "A_sel", "A_cal"]

RESOURCE_CV_METRICS = [
    "mean_time_cv", "mean_cost_cv", "mean_api_calls_cv",
    "mean_actions_cv", "mean_errors_cv", "mean_call_latency_cv", "mean_conf_cv",
]

# The analysis pipeline emits descriptive metric names; the UI (templates, LaTeX
# labels, leaderboard configs) uses compact codes. Translate at the data-loading
# boundary so everything downstream stays unchanged.
_METRIC_RENAME = {
    "consistency_outcome": "C_out",
    "consistency_trajectory_distribution": "C_traj_d",
    "consistency_trajectory_sequence": "C_traj_s",
    "consistency_confidence": "C_conf",
    "consistency_resource": "C_res",
    "predictability_rate_confidence_correlation": "P_rc",
    "predictability_calibration": "P_cal",
    "predictability_roc_auc": "P_auroc",
    "predictability_brier_score": "P_brier",
    "robustness_fault_injection": "R_fault",
    "robustness_structural": "R_struct",
    "robustness_prompt_variation": "R_prompt",
    "safety_harm_severity": "S_harm",
    "safety_compliance": "S_comp",
    "safety_score": "S_safety",
    "abstention_rate": "A_rate",
    "abstention_precision": "A_prec",
    "abstention_recall": "A_rec",
    "abstention_selective_accuracy": "A_sel",
    "abstention_calibration": "A_cal",
}


def _rename_metrics(obj):
    """Recursively rename long metric keys to their short UI codes."""
    if isinstance(obj, dict):
        return {_METRIC_RENAME.get(k, k): _rename_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_metrics(x) for x in obj]
    return obj

PROVIDER_COLORS = {
    "openai": "#10a37f",
    "anthropic": "#d4a574",
    "google": "#4285f4",
    "unknown": "#888888",
}

# Model release dates keyed by raw agent name
MODEL_METADATA = {
    'taubench_toolcalling_gpt_4_turbo': {'date': '2024-04-09'},
    'taubench_toolcalling_gpt_4o_mini': {'date': '2024-07-18'},
    'taubench_toolcalling_gpt_o1': {'date': '2024-12-05'},
    'taubench_toolcalling_gpt_5_2': {'date': '2025-12-11'},
    'taubench_toolcalling_gpt_5_2_xhigh': {'date': '2025-12-11'},
    'taubench_toolcalling_gpt_5_2_medium': {'date': '2025-12-11'},
    'taubench_toolcalling_gpt_5_5': {'date': '2026-04-23'},
    'taubench_toolcalling_gemini_2_flash': {'date': '2024-12-11'},
    'taubench_toolcalling_gemini_2_5_flash': {'date': '2025-03-25'},
    'taubench_toolcalling_gemini_2_5_pro': {'date': '2025-04-17'},
    'taubench_toolcalling_gemini_3_pro': {'date': '2025-11-18'},
    'taubench_toolcalling_gemini_3_1_pro': {'date': '2026-02-19'},
    'taubench_toolcalling_gemini_3_5_flash': {'date': '2026-05-19'},
    'taubench_toolcalling_claude_haiku_3': {'date': '2024-03-13'},
    'taubench_toolcalling_claude_haiku_3_5': {'date': '2024-10-22'},
    'taubench_toolcalling_claude_sonnet_3_7': {'date': '2025-02-24'},
    'taubench_toolcalling_claude_sonnet_4': {'date': '2025-05-22'},
    'taubench_toolcalling_claude_sonnet_4_5': {'date': '2025-09-29'},
    'taubench_toolcalling_claude_opus_4_5': {'date': '2025-11-24'},
    'taubench_toolcalling_claude_opus_4_7': {'date': '2026-04-16'},
    'taubench_fewshot_gpt_4_turbo': {'date': '2024-04-09'},
    'taubench_fewshot_gpt_4o_mini': {'date': '2024-07-18'},
    'taubench_fewshot_gpt_o1': {'date': '2024-12-05'},
    'taubench_fewshot_gpt_5_2': {'date': '2025-12-11'},
    'taubench_fewshot_gpt_5_2_xhigh': {'date': '2025-12-11'},
    'taubench_fewshot_gemini_2_flash': {'date': '2024-12-11'},
    'taubench_fewshot_gemini_2_5_flash': {'date': '2025-03-25'},
    'taubench_fewshot_gemini_2_5_pro': {'date': '2025-04-17'},
    'taubench_fewshot_gemini_3_pro': {'date': '2025-11-18'},
    'taubench_fewshot_claude_haiku_3_5': {'date': '2024-10-22'},
    'taubench_fewshot_claude_sonnet_3_7': {'date': '2025-02-24'},
    'taubench_fewshot_claude_sonnet_4_5': {'date': '2025-09-29'},
    'taubench_fewshot_claude_opus_4_5': {'date': '2025-11-24'},
    'gaia_generalist_gpt_4_turbo': {'date': '2024-04-09'},
    'gaia_generalist_gpt_4o_mini': {'date': '2024-07-18'},
    'gaia_generalist_gpt_o1': {'date': '2024-12-05'},
    'gaia_generalist_gpt_5_2': {'date': '2025-12-11'},
    'gaia_generalist_gpt_5_2_medium': {'date': '2025-12-11'},
    'gaia_generalist_gpt_5_5': {'date': '2026-04-23'},
    'gaia_generalist_gemini_2_flash': {'date': '2024-12-11'},
    'gaia_generalist_gemini_2_5_flash': {'date': '2025-03-25'},
    'gaia_generalist_gemini_2_5_pro': {'date': '2025-04-17'},
    'gaia_generalist_gemini_3_1_pro': {'date': '2026-02-19'},
    'gaia_generalist_gemini_3_5_flash': {'date': '2026-05-19'},
    'gaia_generalist_claude_haiku_3': {'date': '2024-03-13'},
    'gaia_generalist_claude_haiku_3_5': {'date': '2024-10-22'},
    'gaia_generalist_claude_sonnet_3_7': {'date': '2025-02-24'},
    'gaia_generalist_claude_sonnet_4': {'date': '2025-05-22'},
    'gaia_generalist_claude_sonnet_4_5': {'date': '2025-09-29'},
    'gaia_generalist_claude_opus_4_5': {'date': '2025-11-24'},
    'gaia_generalist_claude_opus_4_7': {'date': '2026-04-16'},
}

PROVIDER_SHAPES = {
    "openai": "circle",
    "google": "rectRot",
    "anthropic": "triangle",
    "unknown": "cross",
}

def _guess_provider(agent_name: str) -> str:
    name = agent_name.lower()
    if any(k in name for k in ("gpt", "o1", "o3", "openai")):
        return "openai"
    if any(k in name for k in ("claude", "anthropic", "haiku", "sonnet", "opus")):
        return "anthropic"
    if any(k in name for k in ("gemini", "google")):
        return "google"
    return "unknown"


_DISPLAY_NAMES = {
    "gpt_4_turbo": "GPT-4 Turbo",
    "gpt_4o_mini": "GPT-4o Mini",
    "gpt_o1": "O1",
    "gpt_5_2": "GPT-5.2",
    "gpt_5_2_xhigh": "GPT-5.2 (xhigh)",
    "gpt_5_2_medium": "GPT-5.2 (medium)",
    "gpt_5_5": "GPT-5.5",
    "gemini_2_flash": "Gemini 2.0 Flash",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "gemini_3_pro": "Gemini 3.0 Pro",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gemini_3_5_flash": "Gemini 3.5 Flash",
    "claude_haiku_3_5": "Claude 3.5 Haiku",
    "claude_haiku_3": "Claude 3 Haiku",
    "claude_sonnet_3_7": "Claude 3.7 Sonnet",
    "claude_sonnet_4_5": "Claude Sonnet 4.5",
    "claude_sonnet_4": "Claude Sonnet 4",
    "claude_opus_4_5": "Claude Opus 4.5",
    "claude_opus_4_7": "Claude Opus 4.7",
}


import re

def _slugify(name: str) -> str:
    """Convert a display name to a URL-safe slug."""
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')


def _clean_agent_name(raw: str) -> str:
    """Make agent names human-readable."""
    # Try explicit mapping by matching the model suffix
    for suffix, nice in _DISPLAY_NAMES.items():
        if raw.endswith(suffix):
            return nice
    # Fallback
    parts = raw.split("_")
    model_keywords = {"gpt", "claude", "gemini", "o1", "o3", "llama"}
    start = 0
    for i, p in enumerate(parts):
        if p.lower() in model_keywords:
            start = i
            break
    name_parts = parts[start:]
    return " ".join(p.capitalize() if len(p) > 1 else p.upper() for p in name_parts) if name_parts else raw


def load_benchmarks() -> dict[str, pd.DataFrame]:
    """Return {benchmark_name: DataFrame} for each discovered benchmark."""
    benchmarks = {}
    for d in sorted(ANALYSIS_DIR.iterdir()):
        csv_path = d / "reliability_metrics.csv"
        if d.is_dir() and csv_path.exists():
            df = pd.read_csv(csv_path).rename(columns=_METRIC_RENAME)
            df["provider"] = df["agent"].apply(_guess_provider)
            df["display_name"] = df["agent"].apply(_clean_agent_name)
            df["slug"] = df["display_name"].apply(_slugify)
            df["release_date"] = df["agent"].map(
                lambda x: MODEL_METADATA.get(x, {}).get("date", None)
            )
            # Compute dimension aggregates matching analyze_reliability.py
            # R_Con: weighted 1/3 each for outcome, trajectory (mean of d,s), resource
            if all(c in df.columns for c in ["C_out", "C_traj_d", "C_traj_s", "C_res"]):
                c_traj = df[["C_traj_d", "C_traj_s"]].mean(axis=1)
                df["dim_Consistency"] = (1/3) * df["C_out"] + (1/3) * c_traj + (1/3) * df["C_res"]
            else:
                present = [c for c in DIMENSION_METRICS["Consistency"] if c in df.columns]
                df["dim_Consistency"] = df[present].mean(axis=1) if present else np.nan
            # R_Pred = P_brier (Brier score captures both calibration and discrimination)
            if "P_brier" in df.columns:
                df["dim_Predictability"] = df["P_brier"]
            else:
                present = [c for c in DIMENSION_METRICS["Predictability"] if c in df.columns]
                df["dim_Predictability"] = df[present].mean(axis=1) if present else np.nan
            # R_Rob = mean(R_fault, R_struct, R_prompt)
            present = [c for c in DIMENSION_METRICS["Robustness"] if c in df.columns]
            df["dim_Robustness"] = df[present].mean(axis=1) if present else np.nan
            # R_Saf = S_safety
            if "S_safety" in df.columns:
                df["dim_Safety"] = df["S_safety"]
            else:
                present = [c for c in DIMENSION_METRICS["Safety"] if c in df.columns]
                df["dim_Safety"] = df[present].mean(axis=1) if present else np.nan
            # R_Overall = mean(R_Con, R_Pred, R_Rob) — Safety excluded
            df["overall_reliability"] = df[[f"dim_{d}" for d in _OVERALL_DIMS]].mean(axis=1)
            benchmarks[d.name] = df
    return benchmarks


def get_all_agents(benchmarks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate agent metrics across benchmarks."""
    rows = []
    for bname, df in benchmarks.items():
        for _, row in df.iterrows():
            rows.append({"benchmark": bname, **row.to_dict()})
    return pd.DataFrame(rows)


def agent_summary(all_agents: pd.DataFrame) -> pd.DataFrame:
    """One row per unique display_name with averaged metrics."""
    num_cols = all_agents.select_dtypes(include=[np.number]).columns.tolist()
    summary = all_agents.groupby("display_name")[num_cols].mean().reset_index().sort_values("overall_reliability", ascending=False)
    # Re-attach provider and slug (take first occurrence per display_name)
    first = all_agents.drop_duplicates("display_name").set_index("display_name")
    summary["provider"] = summary["display_name"].map(first["provider"])
    summary["slug"] = summary["display_name"].map(first["slug"])
    return summary


# Mapping from dimension name to JSON filename
DIMENSION_JSON_FILES = {
    "Predictability": "predictability_detailed.json",
    "Consistency": "consistency_detailed.json",
    "Robustness": "robustness_detailed.json",
    "Safety": "safety_detailed.json",
    "Abstention": "abstention_detailed.json",
}


def compute_trend_data(df: pd.DataFrame, y_col: str) -> dict | None:
    """Compute scatter points and linear trend for a metric column.

    Returns a dict with 'points' (list of {x, y, label, provider}) and
    'trend' ({slope, intercept, r, p}) or None if insufficient data.
    """
    from scipy import stats as sp_stats

    sub = df.dropna(subset=[y_col, "release_date"]).copy()
    if len(sub) < 3:
        return None
    sub["_ts"] = pd.to_datetime(sub["release_date"])
    # Convert dates to fractional years for regression
    ref = pd.Timestamp("2024-01-01")
    sub["_years"] = (sub["_ts"] - ref).dt.total_seconds() / (365.25 * 86400)

    slope, intercept, r_value, p_value, _ = sp_stats.linregress(sub["_years"], sub[y_col])

    points = []
    for _, row in sub.iterrows():
        p = {
            "x": row["release_date"],
            "y": float(row[y_col]) if not np.isnan(row[y_col]) else None,
            "label": row["display_name"],
            "slug": row.get("slug", ""),
            "provider": row["provider"],
        }
        if "benchmark" in row.index:
            p["benchmark"] = row["benchmark"]
        points.append(p)

    # Trend line endpoints
    x_min, x_max = sub["_years"].min(), sub["_years"].max()
    trend_line = [
        {"x": sub.loc[sub["_years"].idxmin(), "release_date"], "y": float(intercept + slope * x_min)},
        {"x": sub.loc[sub["_years"].idxmax(), "release_date"], "y": float(intercept + slope * x_max)},
    ]
    return {
        "points": points,
        "trend_line": trend_line,
        "r": round(r_value, 2),
        "slope_per_year": round(slope, 2),
        "p": round(p_value, 2),
    }


def compute_accuracy_scatter(df: pd.DataFrame, y_col: str) -> dict | None:
    """Compute reliability vs accuracy scatter with trend line."""
    from scipy import stats as sp_stats

    sub = df.dropna(subset=[y_col, "accuracy"]).copy()
    if len(sub) < 3:
        return None
    slope, intercept, r_value, p_value, _ = sp_stats.linregress(sub["accuracy"], sub[y_col])

    points = []
    for _, row in sub.iterrows():
        p = {
            "x": float(row["accuracy"]),
            "y": float(row[y_col]) if not np.isnan(row[y_col]) else None,
            "label": row["display_name"],
            "slug": row.get("slug", ""),
            "provider": row["provider"],
        }
        if "benchmark" in row.index:
            p["benchmark"] = row["benchmark"]
        points.append(p)

    x_min, x_max = float(sub["accuracy"].min()), float(sub["accuracy"].max())
    trend_line = [
        {"x": x_min, "y": float(intercept + slope * x_min)},
        {"x": x_max, "y": float(intercept + slope * x_max)},
    ]
    return {
        "points": points,
        "trend_line": trend_line,
        "r": round(r_value, 2),
        "slope": round(slope, 2),
        "p": round(p_value, 2),
    }


def load_dimension_detail(benchmark: str, dimension: str) -> dict | None:
    """Load detailed JSON data for a specific benchmark and dimension."""
    filename = DIMENSION_JSON_FILES.get(dimension)
    if not filename:
        return None
    path = ANALYSIS_DIR / benchmark / filename
    if not path.exists():
        return None
    with open(path) as f:
        return {aid: _rename_metrics(data) for aid, data in json.load(f).items()}


def load_agent_benchmark_detail(benchmark: str, agent_key: str) -> dict | None:
    """Load all detailed data for a specific agent on a specific benchmark."""
    result = {}
    found = False

    for dim, filename in DIMENSION_JSON_FILES.items():
        path = ANALYSIS_DIR / benchmark / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if agent_key in data:
            result[dim] = _rename_metrics(data[agent_key])
            found = True
        else:
            result[dim] = None

    # Level-stratified data
    level_path = ANALYSIS_DIR / benchmark / "level_stratified_detailed.json"
    if level_path.exists():
        with open(level_path) as f:
            level_data = json.load(f)
        result["level_stratified"] = level_data.get(agent_key)
    else:
        result["level_stratified"] = None

    return result if found else None
