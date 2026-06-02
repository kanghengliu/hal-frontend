from flask import Blueprint, render_template, abort
import json
import numpy as np
from .data_loader import (
    load_benchmarks, get_all_agents, agent_summary,
    DIMENSION_METRICS, ABSTENTION_METRICS, RESOURCE_CV_METRICS, PROVIDER_COLORS,
    PROVIDER_SHAPES, load_dimension_detail, load_agent_benchmark_detail,
    DIMENSION_JSON_FILES, _guess_provider,
    compute_trend_data, compute_accuracy_scatter,
)

reliability_bp = Blueprint('reliability', __name__,
    template_folder='templates', static_folder='static')


def metric_color(val):
    """Return a CSS color for a 0-1 metric value (smooth gradient).

    ≤0.5 → red, 0.75 → orange, 1.0 → green, with linear interpolation.
    """
    if val is None:
        return "#475569"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "#475569"
    v = max(0.0, min(1.0, v))
    _R, _O, _G = (239, 68, 68), (245, 158, 11), (16, 185, 129)
    if v <= 0.5:
        r, g, b = _R
    elif v <= 0.75:
        t = (v - 0.5) / 0.25
        r, g, b = [int(a + (b - a) * t) for a, b in zip(_R, _O)]
    else:
        t = (v - 0.75) / 0.25
        r, g, b = [int(a + (b - a) * t) for a, b in zip(_O, _G)]
    return f"rgb({r},{g},{b})"


# LaTeX-formatted metric names (rendered by KaTeX on the client)
_METRIC_LATEX = {
    "C_out": r"$C_{\text{out}}$",
    "C_out_global": r"$C_{\text{out}}^{\text{global}}$",
    "C_out_task": r"$C_{\text{out}}^{\text{task}}$",
    "C_traj_d": r"$C_{\text{traj},d}$",
    "C_traj_s": r"$C_{\text{traj},s}$",
    "C_conf": r"$C_{\text{conf}}$",
    "C_res": r"$C_{\text{res}}$",
    "P_rc": r"$P_{\text{rc}}$",
    "P_cal": r"$P_{\text{cal}}$",
    "P_auroc": r"$P_{\text{auroc}}$",
    "P_brier": r"$P_{\text{brier}}$",
    "R_fault": r"$R_{\text{fault}}$",
    "R_struct": r"$R_{\text{struct}}$",
    "R_prompt": r"$R_{\text{prompt}}$",
    "S_harm": r"$S_{\text{harm}}$",
    "S_comp": r"$S_{\text{comp}}$",
    "S_safety": r"$S_{\text{safety}}$",
    "A_rate": r"$A_{\text{rate}}$",
    "A_prec": r"$A_{\text{prec}}$",
    "A_rec": r"$A_{\text{rec}}$",
    "A_sel": r"$A_{\text{sel}}$",
    "A_cal": r"$A_{\text{cal}}$",
    "mean_confidence": "Mean Confidence",
}


def metric_latex(name):
    """Return a LaTeX-formatted metric name for KaTeX rendering."""
    return _METRIC_LATEX.get(name, name)


# Provider logo filenames; logos needing dark-mode inversion are marked
_PROVIDER_LOGOS = {
    "openai": ("openai.svg", False),
    "anthropic": ("anthropic.svg", False),
    "google": ("gemini.svg", False),
}


# Canonical display names for providers
_PROVIDER_DISPLAY = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
}


def provider_display(name):
    """Return the display name for a provider key."""
    return _PROVIDER_DISPLAY.get(name, name.capitalize())


# Benchmark display names (raw directory name → nice name with optional LaTeX)
_BENCH_DISPLAY = {
    "gaia": "GAIA",
    "taubench_airline": "τ-bench (airline, clean)",
    "taubench_airline_original": "τ-bench (airline, original)",
}

_BENCH_SHORT_DESC = {
    "gaia": "Real-world question-answering tasks requiring multi-step reasoning, tool use, and web browsing.",
    "taubench_airline": "Curated 26-task subset of the airline domain with grading and specification issues removed. Used in main results.",
    "taubench_airline_original": "Full 50-task airline customer-service benchmark including tasks with known grading/specification issues.",
}

_BENCH_DESC = {
    "gaia": (
        "GAIA (General AI Assistants) is a benchmark designed to evaluate AI agents on "
        "real-world question-answering tasks that require multi-step reasoning, tool use, "
        "web browsing, and file manipulation. Questions are organized into three difficulty "
        "levels: Level 1 tasks typically require a single tool or a short chain of reasoning, "
        "Level 2 tasks demand combining multiple tools and reasoning over several steps, and "
        "Level 3 tasks involve long-horizon plans with many intermediate actions. "
        "Agents are evaluated on exact-match accuracy against annotated ground-truth answers. "
        "Because each question has a unique, verifiable answer, GAIA is well-suited for "
        "measuring not only correctness but also the reliability of the problem-solving "
        "process — including consistency across repeated runs, calibration of expressed "
        "confidence, and robustness to perturbations in task formatting."
    ),
    "taubench_airline": (
        "τ-bench (airline, clean) evaluates tool-augmented conversational agents on realistic "
        "customer-service scenarios set in an airline domain. This is a curated 26-task subset "
        "of the original 50-task benchmark, excluding tasks with identified issues in grading or "
        "task specification — such as incorrect answer keys (e.g., type errors in expected values, "
        "wrong passenger details), answer keys that contradict the policy instructions (e.g., "
        "cancelling flights that policy forbids cancelling, issuing certificates without required "
        "preconditions), ambiguous or underspecified task descriptions, and tasks referencing past "
        "dates that make the requested actions impossible. Each task presents a simulated user with "
        "a specific request and the agent must converse while calling backend API tools to resolve it. "
        "Tasks vary in complexity from simple single-action lookups to multi-turn dialogues requiring "
        "policy adherence and disambiguation. All metrics are computed from scratch on this curated "
        "subset, providing a more reliable estimate of agent performance and reliability. "
        "This clean version is used in the main results and aggregate scores."
    ),
    "taubench_airline_original": (
        "τ-bench (airline, original) is the full 50-task version of the τ-bench airline benchmark. "
        "It evaluates tool-augmented conversational agents on realistic customer-service scenarios "
        "in an airline domain. Each task presents a simulated user with a specific request — such as "
        "rebooking a flight, changing a seat, or processing a refund — and the agent must converse "
        "with the user while calling backend API tools (e.g., flight search, booking modification) "
        "to resolve the request. Note that 24 of the 50 tasks have known issues in grading or task "
        "specification (e.g., incorrect answer keys, policy contradictions, ambiguous descriptions). "
        "For a cleaner evaluation, see τ-bench (airline) which excludes these problematic tasks."
    ),
}

_BENCH_REFS = {
    "gaia": {
        "paper": ("GAIA: a benchmark for General AI Assistants", "https://arxiv.org/abs/2311.12983"),
        "website": ("HuggingFace Dataset", "https://huggingface.co/datasets/gaia-benchmark/GAIA"),
    },
    "taubench_airline": {
        "paper": ("τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", "https://arxiv.org/abs/2406.12045"),
        "issues": ("SABER: Small Actions, Big Errors -- Safeguarding Mutating Steps in LLM Agents", "https://arxiv.org/abs/2512.07850"),
        "website": ("GitHub Repository", "https://github.com/sierra-research/tau-bench"),
    },
    "taubench_airline_original": {
        "paper": ("τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", "https://arxiv.org/abs/2406.12045"),
        "issues": ("SABER: Small Actions, Big Errors -- Safeguarding Mutating Steps in LLM Agents", "https://arxiv.org/abs/2512.07850"),
        "website": ("GitHub Repository", "https://github.com/sierra-research/tau-bench"),
    },
}


def bench_display(raw: str) -> str:
    """Return a display-friendly benchmark name."""
    return _BENCH_DISPLAY.get(raw, raw)


_DIM_TOOLTIPS = {
    "Consistency": "How reproducible are the agent's answers and trajectories across repeated runs on the same task?",
    "Predictability": "How well does the agent's expressed confidence predict whether it will answer correctly?",
    "Robustness": "How well does the agent maintain accuracy when inputs are perturbed (faults, structural changes, prompt rewording)?",
    "Safety": "How often does the agent violate safety constraints (harmful content, policy non-compliance)?",
    "Abstention": "How well does the agent know when to abstain from answering rather than guessing incorrectly?",
}

# Mapping dimension name → methodology page anchor
_DIM_ANCHORS = {
    "Consistency": "consistency",
    "Predictability": "predictability",
    "Robustness": "robustness",
    "Safety": "safety",
    "Abstention": "abstention",
}

_DIM_ICONS = {
    "Consistency": "fa-solid fa-arrows-rotate",
    "Predictability": "fa-solid fa-circle-question",
    "Robustness": "fa-solid fa-shield-halved",
    "Safety": "fa-solid fa-triangle-exclamation",
    "Abstention": "fa-solid fa-hand",
}

# Sub-metrics shown in leaderboard tables (Safety excludes S_safety since Agg = S_safety)
_LEADERBOARD_DIM_METRICS = {d: (cols if d != "Safety" else ["S_harm", "S_comp"]) for d, cols in DIMENSION_METRICS.items()}
_LEADERBOARD_DIM_METRICS["Abstention"] = ABSTENTION_METRICS

# Load data once at startup
BENCHMARKS = load_benchmarks()
ALL_AGENTS = get_all_agents(BENCHMARKS)
AGENT_SUMMARY = agent_summary(ALL_AGENTS)

# Landing page excludes taubench_airline_original from aggregates
_LANDING_EXCLUDE = {"taubench_airline_original"}
LANDING_AGENTS = ALL_AGENTS[~ALL_AGENTS["benchmark"].isin(_LANDING_EXCLUDE)]
LANDING_SUMMARY = agent_summary(LANDING_AGENTS)

# Averaged version for "All Benchmarks" trend charts (one dot per model)
LANDING_AVERAGED = LANDING_SUMMARY.copy()
_date_map = LANDING_AGENTS.drop_duplicates("display_name").set_index("display_name")["release_date"]
LANDING_AVERAGED["release_date"] = LANDING_AVERAGED["display_name"].map(_date_map)

# Slug → display_name lookup (populated from all agents)
SLUG_TO_NAME = dict(zip(ALL_AGENTS["slug"], ALL_AGENTS["display_name"]))


@reliability_bp.context_processor
def inject_globals():
    return {
        'metric_color': metric_color,
        'metric_latex': metric_latex,
        'provider_logos': _PROVIDER_LOGOS,
        'provider_display': provider_display,
        'bench_display': bench_display,
        'dim_tooltips': _DIM_TOOLTIPS,
        'dim_anchors': _DIM_ANCHORS,
        'dim_icons': _DIM_ICONS,
        'benchmarks': list(BENCHMARKS.keys()),
    }


def _safe_json(obj):
    """Convert to JSON-safe values (handle NaN)."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _df_to_json(df):
    records = df.to_dict(orient="records")
    return json.loads(json.dumps(records, default=_safe_json))


def _build_trend_charts(df):
    """Build trend data for overall + each dimension, both vs date and vs accuracy."""
    dimensions = list(DIMENSION_METRICS.keys())
    metrics = [("Overall Reliability", "overall_reliability")] + [
        (dim, f"dim_{dim}") for dim in dimensions
    ]
    # Accuracy over time (shared across all dimensions)
    acc_over_time = compute_trend_data(df, "accuracy")
    charts = {}
    for label, col in metrics:
        t = compute_trend_data(df, col)
        a = compute_accuracy_scatter(df, col)
        if t or a:
            charts[label] = {"vs_date": t, "vs_accuracy": a, "acc_over_time": acc_over_time}
    return charts


@reliability_bp.route("/")
def index():
    stats = {
        "num_benchmarks": len(BENCHMARKS) - len(_LANDING_EXCLUDE),
        "num_agents": LANDING_AGENTS["display_name"].nunique(),
        "avg_accuracy": float(LANDING_AGENTS["accuracy"].mean()),
    }
    leaderboard = _df_to_json(LANDING_SUMMARY.head(20))
    dimensions = list(DIMENSION_METRICS.keys())

    # Per-benchmark leaderboards for the selector
    leaderboard_by_bench = {"All Benchmarks": leaderboard}
    for bname, bdf in BENCHMARKS.items():
        if bname in _LANDING_EXCLUDE:
            continue
        sorted_bdf = bdf.sort_values("overall_reliability", ascending=False)
        leaderboard_by_bench[bench_display(bname)] = _df_to_json(sorted_bdf)
    trend_charts = _build_trend_charts(LANDING_AVERAGED)
    trend_by_bench = {"All Benchmarks": trend_charts}
    for bname, bdf in BENCHMARKS.items():
        if bname in _LANDING_EXCLUDE:
            continue
        trend_by_bench[bench_display(bname)] = _build_trend_charts(bdf)
    landing_benchmarks = [b for b in BENCHMARKS.keys() if b not in _LANDING_EXCLUDE]
    bench_display_to_raw = {bench_display(b): b for b in BENCHMARKS if b not in _LANDING_EXCLUDE}
    return render_template("reliability/index.html",
        stats=stats,
        leaderboard=leaderboard,
        benchmarks=list(BENCHMARKS.keys()),
        landing_benchmarks=landing_benchmarks,
        dimensions=dimensions,
        dim_metrics=DIMENSION_METRICS,
        leaderboard_dim_metrics=_LEADERBOARD_DIM_METRICS,
        findings=_FINDINGS,
        recommendations=_RECOMMENDATIONS,
        trend_charts=json.dumps(trend_charts, default=_safe_json),
        trend_by_bench=json.dumps(trend_by_bench, default=_safe_json),
        bench_display_to_raw=json.dumps(bench_display_to_raw),
        provider_colors=json.dumps(PROVIDER_COLORS),
        provider_shapes=json.dumps(PROVIDER_SHAPES),
        leaderboard_by_bench=json.dumps(leaderboard_by_bench, default=_safe_json),
        bench_short_desc=_BENCH_SHORT_DESC,
    )


@reliability_bp.route("/benchmark/<name>/")
def benchmark(name):
    if name not in BENCHMARKS:
        abort(404)
    df = BENCHMARKS[name]
    agents = _df_to_json(df.sort_values("overall_reliability", ascending=False))
    dimensions = list(DIMENSION_METRICS.keys())
    # Bar chart data per dimension
    bar_data = {}
    for dim in dimensions:
        col = f"dim_{dim}"
        sorted_agents = sorted(agents, key=lambda a: a.get(col) if a.get(col) is not None else -1, reverse=True)
        bar_data[dim] = {
            "labels": [a["display_name"] for a in sorted_agents],
            "values": [_safe_json(a.get(col, 0)) for a in sorted_agents],
            "colors": [PROVIDER_COLORS.get(a.get("provider", "unknown"), "#888") for a in sorted_agents],
            "providers": [a.get("provider", "") for a in sorted_agents],
        }
    trend_charts = _build_trend_charts(df)
    return render_template("reliability/benchmark.html",
        name=name,
        bench_desc=_BENCH_DESC.get(name, ""),
        bench_refs=_BENCH_REFS.get(name, {}),
        agents=agents,
        dimensions=dimensions,
        leaderboard_dim_metrics=_LEADERBOARD_DIM_METRICS,
        bar_data=json.dumps(bar_data, default=_safe_json),
        dim_metrics=DIMENSION_METRICS,
        benchmarks=list(BENCHMARKS.keys()),
        trend_charts=json.dumps(trend_charts, default=_safe_json),
        provider_colors=json.dumps(PROVIDER_COLORS),
        provider_shapes=json.dumps(PROVIDER_SHAPES),
    )


@reliability_bp.route("/agent/<slug>/")
def agent(slug):
    name = SLUG_TO_NAME.get(slug)
    if name is None:
        abort(404)
    rows = ALL_AGENTS[ALL_AGENTS["display_name"] == name]
    if rows.empty:
        abort(404)
    agent_data = _df_to_json(rows)
    dimensions = list(DIMENSION_METRICS.keys())
    # Radar for this agent across benchmarks
    radar_data = {
        "labels": dimensions,
        "datasets": [],
    }
    colors = ["#0ea5e9", "#f43f5e", "#10b981", "#f59e0b"]
    for i, rec in enumerate(agent_data):
        radar_data["datasets"].append({
            "label": bench_display(rec["benchmark"]),
            "data": [_safe_json(rec.get(f"dim_{d}", 0)) for d in dimensions],
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "20",
        })
    # Detailed metrics
    detail_metrics = {}
    for dim, cols in DIMENSION_METRICS.items():
        detail_metrics[dim] = cols
    return render_template("reliability/agent.html",
        name=name,
        agent_slug=slug,
        agent_data=agent_data,
        radar_data=json.dumps(radar_data, default=_safe_json),
        dimensions=dimensions,
        detail_metrics=detail_metrics,
        abstention_metrics=ABSTENTION_METRICS,
        resource_metrics=RESOURCE_CV_METRICS,
        benchmarks=list(BENCHMARKS.keys()),
    )


@reliability_bp.route("/provider/<name>/")
def provider(name):
    rows = ALL_AGENTS[ALL_AGENTS["provider"] == name]
    if rows.empty:
        abort(404)
    agents = _df_to_json(rows.sort_values("overall_reliability", ascending=False))
    dimensions = list(DIMENSION_METRICS.keys())
    bench_names = sorted(rows["benchmark"].unique())
    # Radar: one dataset per agent across dimensions (averaged over benchmarks)
    summary = rows.groupby("display_name")[
        [f"dim_{d}" for d in dimensions] + ["overall_reliability", "accuracy"]
    ].mean().sort_values("overall_reliability", ascending=False)
    colors = ["#0ea5e9", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6", "#06b6d4", "#ec4899", "#14b8a6"]
    radar_data = {"labels": dimensions, "datasets": []}
    for i, (dname, row) in enumerate(summary.iterrows()):
        radar_data["datasets"].append({
            "label": dname,
            "data": [_safe_json(row.get(f"dim_{d}", 0)) for d in dimensions],
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "20",
        })
    # Trend charts use per-agent-per-benchmark rows (not summary)
    trend_charts = _build_trend_charts(rows)
    return render_template("reliability/provider.html",
        name=name,
        agents=agents,
        bench_names=bench_names,
        dimensions=dimensions,
        dim_metrics=DIMENSION_METRICS,
        leaderboard_dim_metrics=_LEADERBOARD_DIM_METRICS,
        radar_data=json.dumps(radar_data, default=_safe_json),
        provider_color=PROVIDER_COLORS.get(name, "#888"),
        benchmarks=list(BENCHMARKS.keys()),
        trend_charts=json.dumps(trend_charts, default=_safe_json),
        provider_colors=json.dumps(PROVIDER_COLORS),
        provider_shapes=json.dumps(PROVIDER_SHAPES),
    )


_FINDINGS = [
    {
        "icon": "fa-solid fa-scale-balanced",
        "color": "#0ea5e9",
        "title": "Reliability Lags Behind Accuracy Improvements",
        "body": (
            "Despite 24 months of model development, overall reliability shows only small improvements "
            "over time while accuracy steadily climbs. Improving raw task performance is insufficient "
            "for building dependable AI agents — reliability requires targeted attention beyond "
            "capability scaling alone."
        ),
        "detail": (
            "Reliability improvements are also disproportionate across evaluation scenarios: "
            "highly structured environments show moderate gains, while open-ended tasks show "
            "barely any improvement, even among the latest models."
        ),
    },
    {
        "icon": "fa-solid fa-arrows-rotate",
        "color": "#ef4444",
        "title": "Outcome and Resource Consistency Remain Low",
        "body": (
            "Agents that <em>can</em> solve a task often fail to do so consistently. The gap between "
            "capability (pass@k) and reliability (pass^k) is substantial across all models. "
            "Resource consistency is similarly low, with high variance in token and compute usage "
            "across runs — agents allocate effort unpredictably."
        ),
        "detail": (
            "A 'what but not when' pattern emerges: agents achieve substantially higher distribution "
            "consistency than sequence consistency, indicating they reliably select similar action types "
            "across runs but vary in execution order. Improving reliability requires not just better "
            "action selection but more stable planning and execution."
        ),
    },
    {
        "icon": "fa-solid fa-chart-line",
        "color": "#f59e0b",
        "title": "Calibration Improves, but Discrimination Stagnates",
        "body": (
            "Calibration — the alignment between predicted confidence and actual accuracy — has "
            "improved noticeably in recent frontier models. However, discrimination — the ability "
            "to distinguish tasks the agent will solve from those it won't — shows divergent trends "
            "across benchmarks and has in some cases worsened."
        ),
        "detail": (
            "Improvements in calibration alone do not guarantee reliable failure identification. "
            "An agent may express well-calibrated confidence yet still fail to distinguish "
            "correct from incorrect predictions. Both sub-metrics must be measured independently."
        ),
    },
    {
        "icon": "fa-solid fa-shield-halved",
        "color": "#06b6d4",
        "title": "Robustness Saturates, but Prompt Sensitivity Distinguishes Models",
        "body": (
            "Fault robustness and structural robustness show ceiling effects across most models — "
            "agents handle genuine technical failures gracefully. In contrast, prompt robustness "
            "remains a key differentiator: sensitivity to superficial instruction paraphrasing varies "
            "substantially across models."
        ),
        "detail": (
            "This pattern is counterintuitive: models tolerate real infrastructure faults but remain "
            "vulnerable to surface-level variations in how tasks are specified — a critical concern "
            "for real-world deployment where user instructions naturally vary."
        ),
    },
    {
        "icon": "fa-solid fa-arrows-up-down",
        "color": "#3b82f6",
        "title": "Reliability Does Not Scale Uniformly with Capability",
        "body": (
            "While calibration, robustness, and safety generally improve with model size, consistency "
            "often exhibits an inverse pattern: smaller models frequently achieve equal or higher "
            "consistency than their larger counterparts. Reasoning models are generally more reliable, "
            "but their reliability does not improve as quickly as their accuracy."
        ),
        "detail": (
            "Larger models have more solution paths available, which increases run-to-run variability. "
            "This suggests that scaling alone will not solve the reliability problem — targeted "
            "architectural and training interventions are needed."
        ),
    },
    {
        "icon": "fa-solid fa-triangle-exclamation",
        "color": "#ec4899",
        "title": "Safety Improves, but High-Severity Violations Persist",
        "body": (
            "The most recent frontier models exhibit significantly lower overall violation rates. "
            "However, financial accuracy violations — incorrect charges and refunds — remain the "
            "most prevalent failure mode. Even infrequent high-severity failures can carry significant "
            "costs and represent critical blockers for deployment."
        ),
        "detail": (
            "Benchmark quality also matters: safety and predictability improve almost universally when "
            "evaluated on a verified task subset with grading errors removed, underscoring the "
            "importance of clean evaluation data."
        ),
    },
    {
        "icon": "fa-solid fa-chart-column",
        "color": "#14b8a6",
        "title": "Reliability Gains Are Disproportionate Across Benchmarks",
        "body": (
            "Reliability profiles are highly task-type dependent. An agent that is reliable on "
            "open-ended multi-step reasoning may struggle on structured customer-service tasks, "
            "and vice versa. Dimension-level scores vary substantially across benchmarks for "
            "the same agent."
        ),
        "detail": (
            "This highlights the need for multi-benchmark evaluation. Single-benchmark reliability "
            "scores can be misleading — agents must be tested across diverse task structures "
            "to build a complete picture of their reliability."
        ),
    },
]


# Recommendations derived from the paper "Towards a Science of AI Agent Reliability"
_RECOMMENDATIONS = [
    {
        "icon": "fa-solid fa-flask",
        "color": "#0ea5e9",
        "title": "Evaluate with Dynamic, Multi-Run Protocols",
        "body": (
            "Single-run accuracy on fixed benchmarks provides a misleadingly narrow view of capability. "
            "Use <strong>multi-run protocols</strong> to assess variance across identical tasks, "
            "<strong>multi-condition protocols</strong> to systematically perturb user inputs, "
            "and <strong>temporal re-evaluation</strong> at regular intervals to detect silent degradation."
        ),
        "detail": (
            "Current benchmarks are too static. Generative benchmarks with parameterized test sets "
            "(renaming fields, reordering responses, injecting faults) would provide more realistic "
            "and robust evaluations."
        ),
    },
    {
        "icon": "fa-solid fa-compass-drafting",
        "color": "#10b981",
        "title": "Design Agents Explicitly for Reliability",
        "body": (
            "Calibration and safety have improved noticeably — evidence that intentional optimization works. "
            "In contrast, <strong>consistency and discrimination show little progress</strong>, suggesting they "
            "are not yet explicit optimization targets. Make reliability dimensions measurable and "
            "actionable in agent development."
        ),
        "detail": (
            "Capability-oriented evaluation alone misses actionable optimization targets. "
            "Use reliability metrics to identify which dimensions lack progress and need targeted attention."
        ),
    },
    {
        "icon": "fa-solid fa-certificate",
        "color": "#f59e0b",
        "title": "Use Reliability Metrics for Deployment Governance",
        "body": (
            "Treat reliability as a deployment prerequisite, similar to aviation safety standards. "
            "Set <strong>minimum thresholds</strong> for consistency and safety before production deployment, "
            "implement incident reporting, and use multi-dimensional reliability metrics to guide "
            "change management decisions."
        ),
        "detail": (
            "Organizations should require reliability certification before deployment, not just "
            "capability assessment. Diverse contributions through dimension-specific optimization "
            "become possible with clear measurement."
        ),
    },
    {
        "icon": "fa-solid fa-people-arrows",
        "color": "#ec4899",
        "title": "Distinguish Automation vs. Augmentation Use Cases",
        "body": (
            "Reliability requirements differ fundamentally by use case. For <strong>augmentation</strong> "
            "(coding assistants, copilots), moderate reliability may suffice since humans review output. "
            "For <strong>automation</strong> (customer service, database management), reliability is a "
            "hard prerequisite — 90% success with unpredictable 10% failures is unacceptable."
        ),
        "detail": (
            "As the field pushes toward greater agent autonomy, the reliability bar rises significantly. "
            "Deployment standards should be context-aware and scale with the level of autonomous action."
        ),
    },
]


@reliability_bp.route("/findings/")
def findings():
    return render_template("reliability/findings.html",
        findings=_FINDINGS,
        recommendations=_RECOMMENDATIONS,
        benchmarks=list(BENCHMARKS.keys()),
    )


@reliability_bp.route("/methodology/")
def methodology():
    return render_template("reliability/methodology.html",
        benchmarks=list(BENCHMARKS.keys()),
    )


@reliability_bp.route("/benchmark/<bench>/dimension/<dim>/")
def dimension_detail(bench, dim):
    if bench not in BENCHMARKS:
        abort(404)
    # Accept lowercase URLs, capitalize for internal lookup
    dim = dim.capitalize()
    if dim not in DIMENSION_JSON_FILES and dim != "Abstention":
        abort(404)
    detail = load_dimension_detail(bench, dim)
    if detail is None:
        abort(404)
    # Convert to JSON-safe list sorted by display_name
    agents = []
    for agent_id, data in detail.items():
        safe = json.loads(json.dumps(data, default=_safe_json))
        safe["_id"] = agent_id
        safe["provider"] = _guess_provider(agent_id)
        agents.append(safe)
    agents.sort(key=lambda a: a.get("display_name", ""))
    # Leaderboard data from the benchmark CSV, sorted by this dimension's aggregate
    df = BENCHMARKS[bench]
    dim_col = f"dim_{dim}"
    sort_col = dim_col if dim_col in df.columns else "overall_reliability"
    leaderboard_agents = _df_to_json(df.sort_values(sort_col, ascending=False))
    return render_template("reliability/dimension_detail.html",
        benchmark=bench,
        dimension=dim,
        agents=agents,
        agents_json=json.dumps(agents, default=_safe_json),
        provider_colors=json.dumps(PROVIDER_COLORS),
        benchmarks=list(BENCHMARKS.keys()),
        leaderboard_agents=leaderboard_agents,
        leaderboard_dim_metrics=_LEADERBOARD_DIM_METRICS,
        dim_metrics=DIMENSION_METRICS,
    )


@reliability_bp.route("/agent/<slug>/benchmark/<bench>/")
def agent_benchmark(slug, bench):
    agent_name = SLUG_TO_NAME.get(slug)
    if agent_name is None:
        abort(404)
    if bench not in BENCHMARKS:
        abort(404)
    df = BENCHMARKS[bench]
    row = df[df["display_name"] == agent_name]
    if row.empty:
        abort(404)
    row = row.iloc[0]
    agent_key = row["agent"]
    detail = load_agent_benchmark_detail(bench, agent_key)
    if detail is None:
        abort(404)
    # Aggregate metrics from the CSV row
    dimensions = list(DIMENSION_METRICS.keys())
    metrics = {}
    for dim, cols in DIMENSION_METRICS.items():
        metrics[dim] = {c: _safe_json(row.get(c)) for c in cols}
        metrics[dim]["_dim"] = _safe_json(row.get(f"dim_{dim}"))
    metrics["Abstention"] = {c: _safe_json(row.get(c)) for c in ABSTENTION_METRICS}
    metrics["accuracy"] = _safe_json(row.get("accuracy"))
    metrics["overall_reliability"] = _safe_json(row.get("overall_reliability"))

    # Rank among agents on this benchmark
    sorted_df = df.sort_values("overall_reliability", ascending=False).reset_index(drop=True)
    rank = int(sorted_df[sorted_df["display_name"] == agent_name].index[0]) + 1
    total_agents = len(sorted_df)

    safe_detail = json.loads(json.dumps(detail, default=_safe_json))
    return render_template("reliability/agent_benchmark.html",
        agent_name=agent_name,
        agent_slug=slug,
        benchmark=bench,
        provider=row["provider"],
        metrics=metrics,
        dimensions=dimensions,
        dim_metrics=DIMENSION_METRICS,
        abstention_metrics=ABSTENTION_METRICS,
        detail=safe_detail,
        detail_json=json.dumps(safe_detail, default=_safe_json),
        rank=rank,
        total_agents=total_agents,
        benchmarks=list(BENCHMARKS.keys()),
        provider_color=PROVIDER_COLORS.get(row["provider"], "#0ea5e9"),
    )


@reliability_bp.route("/benchmark/<bench>/analysis/")
def benchmark_analysis(bench):
    if bench not in BENCHMARKS:
        abort(404)
    template = f"reliability/analysis_{bench}.html"
    return render_template(template, benchmark=bench, benchmarks=list(BENCHMARKS.keys()))


@reliability_bp.route("/compare/taubench/")
def compare_taubench():
    full_name = "taubench_airline_original"
    clean_name = "taubench_airline"
    if full_name not in BENCHMARKS or clean_name not in BENCHMARKS:
        abort(404)

    df_full = BENCHMARKS[full_name]
    df_clean = BENCHMARKS[clean_name]

    # Merge on display_name, keep only agents in both
    merged = df_full.merge(df_clean, on="display_name", suffixes=("_full", "_clean"), how="inner")
    merged = merged.sort_values("overall_reliability_full", ascending=False)

    dimensions = list(DIMENSION_METRICS.keys())

    # Build comparison bar data: for each dimension + accuracy + overall, paired bars
    # Each chart sorted by the full-benchmark value (descending)
    compare_data = {}
    for key, col_base in [("Accuracy", "accuracy"), ("Overall Reliability", "overall_reliability")] + [(d, f"dim_{d}") for d in dimensions]:
        sorted_m = merged.sort_values(f"{col_base}_full", ascending=False)
        labels = sorted_m["display_name"].tolist()
        vals_full = [_safe_json(v) for v in sorted_m[f"{col_base}_full"].tolist()]
        vals_clean = [_safe_json(v) for v in sorted_m[f"{col_base}_clean"].tolist()]
        compare_data[key] = {
            "labels": labels,
            "full": vals_full,
            "clean": vals_clean,
        }

    # Build detailed metrics table rows
    agents = []
    for _, row in merged.iterrows():
        rec = {"display_name": row["display_name"], "provider": row.get("provider_full", "")}
        for dim, cols in DIMENSION_METRICS.items():
            for c in cols:
                rec[f"{c}_full"] = _safe_json(row.get(f"{c}_full"))
                rec[f"{c}_clean"] = _safe_json(row.get(f"{c}_clean"))
        rec["accuracy_full"] = _safe_json(row.get("accuracy_full"))
        rec["accuracy_clean"] = _safe_json(row.get("accuracy_clean"))
        rec["overall_full"] = _safe_json(row.get("overall_reliability_full"))
        rec["overall_clean"] = _safe_json(row.get("overall_reliability_clean"))
        agents.append(rec)

    return render_template("reliability/compare_taubench.html",
        agents=agents,
        dimensions=dimensions,
        compare_data=json.dumps(compare_data, default=_safe_json),
        dim_metrics=DIMENSION_METRICS,
        benchmarks=list(BENCHMARKS.keys()),
        provider_colors=json.dumps(PROVIDER_COLORS),
    )
