"""
dashboard.py — Streamlit visualization dashboard for RAG evaluation results.

PURPOSE
-------
Provides an interactive view of the evaluation CSV files produced by the
eval_tests/ragas_eval/ test suite. After running the pytest tests, open this
dashboard to explore scores, pass/fail rates, and per-question breakdowns
without writing any code.

HOW TO RUN
----------
    streamlit run dashboard.py

WHAT IT DISPLAYS
----------------
  Sidebar          — switch between the four result CSV files
  Metric cards     — mean score + pass count at a glance
  Bar chart        — scores per question for selected metric(s)
  Donut charts     — pass/fail ratio per metric
  Results table    — colour-coded score table (green = pass, red = fail)
  Question detail  — drill into a single question to see its answer, context,
                     ground truth, and individual scores

DATA SOURCE
-----------
All CSV files are read from eval_results/ (written by the pytest tests).
Streamlit caches the CSV reads so the dashboard doesn't re-read files on
every user interaction — only on a hard refresh.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🍛",
    layout="wide",
)

# Absolute path to the results folder so the dashboard works regardless of
# which directory `streamlit run` is invoked from.
EVAL_DIR  = os.path.join(os.path.dirname(__file__), "eval_results")
THRESHOLD = 0.5   # same threshold used during scoring — anything below is a fail

# Human-readable labels for column names used throughout all charts and tables.
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    """Load a CSV from disk, returning None if the file doesn't exist yet.
    Cached so Streamlit doesn't re-read on every widget interaction."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# Load all four possible result files upfront.
# Each file is produced by a different pytest test in eval_tests/ragas_eval/.
ragas_df         = load_csv(os.path.join(EVAL_DIR, "ragas_results.csv"))           # all 3 metrics
faithfulness_df  = load_csv(os.path.join(EVAL_DIR, "faithfulness_results.csv"))    # faithfulness only
relevancy_df     = load_csv(os.path.join(EVAL_DIR, "answer_relevancy_results.csv"))# answer relevancy only
ctx_precision_df = load_csv(os.path.join(EVAL_DIR, "context_precision_results.csv"))# context precision only

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🍛 Indian Recipes RAG — Evaluation Dashboard")
st.caption("Judge: Claude Sonnet 4.6  |  Generator: Phi-4 Mini (local)  |  Pass threshold: 0.5")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — file selector
# Lets the user switch between single-metric runs and the combined run
# without needing to reload the page or edit code.
# ---------------------------------------------------------------------------
st.sidebar.header("Data Source")
view = st.sidebar.radio(
    "Select results file",
    [
        "All Metrics (ragas_results)",
        "Faithfulness only",
        "Answer Relevancy only",
        "Context Precision only",
    ],
)

# Map the sidebar selection to the correct DataFrame and metric column list.
if view == "All Metrics (ragas_results)":
    df      = ragas_df
    metrics = ["faithfulness", "context_precision", "answer_relevancy"]
elif view == "Faithfulness only":
    df      = faithfulness_df
    metrics = ["faithfulness"]
elif view == "Answer Relevancy only":
    df      = relevancy_df
    metrics = ["answer_relevancy"]
else:
    df      = ctx_precision_df
    metrics = ["context_precision"]

# Guard: show instructions and stop if the selected CSV doesn't exist yet.
if df is None or df.empty:
    st.warning(
        "No results found. Run the eval tests first:\n\n"
        "```\nEVAL_SAMPLE_SIZE=9 pytest eval_tests/ragas_eval/test_ragas_evaluation.py -v -s\n```"
    )
    st.stop()

# Truncate long questions for chart axis labels — full text shown in the detail viewer.
df = df.copy()
df["q_short"] = df["question"].str[:60] + "…"

# ---------------------------------------------------------------------------
# Summary metric cards — mean score + pass count
# One card per metric so quality can be assessed at a glance.
# ---------------------------------------------------------------------------
available_metrics = [m for m in metrics if m in df.columns]
cols = st.columns(len(available_metrics))

for col, metric in zip(cols, available_metrics):
    valid       = df[metric].dropna()
    mean_score  = valid.mean() if not valid.empty else 0
    pass_count  = (valid >= THRESHOLD).sum()
    total       = len(valid)
    # "inverse" turns the delta red when mean is below threshold — signals failure
    delta_color = "normal" if mean_score >= THRESHOLD else "inverse"

    col.metric(
        label=METRIC_LABELS.get(metric, metric),
        value=f"{mean_score:.2f}",
        delta=f"{pass_count}/{total} passing",
        delta_color=delta_color,
    )

st.divider()

# ---------------------------------------------------------------------------
# Bar chart — scores per question
# Single-metric view uses a colour gradient (red→green) so low scores stand out.
# Multi-metric view uses grouped bars so metrics can be compared per question.
# ---------------------------------------------------------------------------
st.subheader("Scores per Question")

if len(available_metrics) == 1:
    metric = available_metrics[0]
    fig = px.bar(
        df,
        x="q_short",
        y=metric,
        color=metric,
        color_continuous_scale=["#d9534f", "#f0ad4e", "#5cb85c"],  # red → amber → green
        range_color=[0, 1],
        labels={"q_short": "Question", metric: METRIC_LABELS.get(metric, metric)},
        text=df[metric].round(2),
    )
else:
    # Melt to long format so Plotly can group bars by metric per question.
    melted = df.melt(
        id_vars=["q_short"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )
    melted["Metric"] = melted["Metric"].map(METRIC_LABELS)
    fig = px.bar(
        melted,
        x="q_short",
        y="Score",
        color="Metric",
        barmode="group",
        labels={"q_short": "Question", "Score": "Score (0–1)"},
        text=melted["Score"].round(2),
    )

# Red dashed threshold line so it's immediately obvious which bars fail.
fig.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Threshold ({THRESHOLD})",
    annotation_position="top left",
)
fig.update_traces(textposition="outside")
fig.update_layout(
    xaxis_tickangle=-30,
    yaxis_range=[0, 1.15],
    legend_title_text="Metric",
    height=420,
    margin=dict(t=30, b=120),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Donut charts — pass/fail breakdown
# Gives a quick ratio view alongside the bar chart so percentages are clear.
# ---------------------------------------------------------------------------
st.subheader("Pass / Fail Breakdown")
pass_cols = st.columns(len(available_metrics))

for col, metric in zip(pass_cols, available_metrics):
    valid      = df[metric].dropna()
    pass_count = int((valid >= THRESHOLD).sum())
    fail_count = int((valid < THRESHOLD).sum())

    donut = go.Figure(go.Pie(
        labels=["Pass", "Fail"],
        values=[pass_count, fail_count],
        hole=0.55,                                          # hollow centre = donut style
        marker_colors=["#5cb85c", "#d9534f"],              # green pass, red fail
        textinfo="value+percent",
    ))
    donut.update_layout(
        title_text=METRIC_LABELS.get(metric, metric),
        showlegend=True,
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    col.plotly_chart(donut, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Results table — colour-coded scores and pass/fail booleans
# Green = at or above threshold, red = below threshold.
# ---------------------------------------------------------------------------
st.subheader("Results Table")

display_cols = ["question"] + available_metrics
# Show per-metric pass columns (ragas_results) or single pass_fail column (individual runs)
if "pass_fail" in df.columns:
    display_cols += ["pass_fail"]
elif "faithfulness_pass" in df.columns:
    display_cols += ["faithfulness_pass", "context_precision_pass", "answer_relevancy_pass"]

table_df = df[display_cols].copy()


def color_score(val):
    """Apply background colour to score and boolean cells for the styled table."""
    if isinstance(val, float):
        if val >= THRESHOLD:
            return "background-color: #d4edda; color: #155724"  # green
        else:
            return "background-color: #f8d7da; color: #721c24"  # red
    if isinstance(val, bool) or str(val).lower() in ("true", "false"):
        v = str(val).lower() == "true"
        return (
            "background-color: #d4edda; color: #155724"
            if v
            else "background-color: #f8d7da; color: #721c24"
        )
    return ""


score_cols_ = [c for c in table_df.columns if c in available_metrics]
pass_cols_  = [c for c in table_df.columns if "pass" in c]

styled = table_df.style.applymap(color_score, subset=score_cols_ + pass_cols_).format(
    {m: "{:.3f}" for m in score_cols_}
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Question detail viewer — drill into a single row
# Lets you read the full generated answer, context, and ground truth side-by-side
# without having to open the CSV manually.
# ---------------------------------------------------------------------------
st.subheader("Question Detail")

question_options = df["question"].tolist()
selected_q = st.selectbox("Select a question to inspect", question_options)
row = df[df["question"] == selected_q].iloc[0]

detail_left, detail_right = st.columns(2)

with detail_left:
    st.markdown("**Question**")
    st.info(row["question"])

    st.markdown("**Generated Answer** *(Phi-4 Mini)*")
    st.write(row.get("answer", "—"))

with detail_right:
    st.markdown("**Ground Truth**")
    st.success(row.get("ground_truth", "—"))

    st.markdown("**Context Preview**")
    st.caption(row.get("contexts_preview", "—"))

st.markdown("**Scores for this question**")
score_display = {
    METRIC_LABELS.get(m, m): f"{row[m]:.3f}" if pd.notna(row.get(m)) else "N/A"
    for m in available_metrics
    if m in row
}
score_cols_disp = st.columns(len(score_display))
for col, (label, val) in zip(score_cols_disp, score_display.items()):
    col.metric(label=label, value=val)

# ---------------------------------------------------------------------------
# Footer — show when the last eval ran and which judge model was used
# ---------------------------------------------------------------------------
st.divider()
if "run_timestamp" in df.columns:
    st.caption(f"Last evaluation run: {df['run_timestamp'].iloc[0]}")
if "judge_model" in df.columns:
    st.caption(f"Judge model: {df['judge_model'].iloc[0]}")
