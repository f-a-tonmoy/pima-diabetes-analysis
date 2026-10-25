"""Plotly figures for the report.

Every figure goes through :func:`base_layout`, which carries the rules that
apply to all of them: dragging off so a swipe on a touch screen scrolls the page
instead of drawing a zoom box, a base font large enough to read, no mode bar,
and a transparent paper so the figure sits on the page rather than in a card.

Colour is semantic and fixed across the report:

===========  ==========================================================
green        the correct or standardized choice, the structure that holds
clay         the artifact, the trap, the thing that looks good and is not
violet/blue  additional model series, assigned in fixed slot order
gold         context and reference marks
===========  ==========================================================
"""

from __future__ import annotations

import plotly.graph_objects as go

GREEN = "#2e8b57"
VIOLET = "#8250c4"
CLAY = "#d2492a"
BLUE = "#2f6fd0"
GOLD = "#b8860b"
SERIES = (GREEN, VIOLET, CLAY, BLUE, GOLD)

INK = "#1a1a17"
INK_SOFT = "#5d5c55"
GRID = "#e2ded4"
SURFACE = "rgba(0,0,0,0)"

MODEL_COLOR = {
    "logistic_regression": GREEN,
    "random_forest": VIOLET,
    "svm_rbf": BLUE,
    "gradient_boosting": GOLD,
}
MODEL_LABEL = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svm_rbf": "SVM (RBF)",
    "gradient_boosting": "Gradient boosting",
    "majority_class": "Always predict no",
}


def base_layout(height: int = 420, **kwargs) -> dict:
    """Shared layout. Dragging is off: these charts reveal nothing when zoomed."""
    layout = dict(
        height=height,
        dragmode=False,
        font=dict(family="Public Sans, system-ui, sans-serif", size=14, color=INK),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title=dict(font=dict(size=13, color=INK_SOFT))),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title=dict(font=dict(size=13, color=INK_SOFT))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=13), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family="Public Sans, system-ui, sans-serif", size=13)),
    )
    layout.update(kwargs)
    return layout


# ---------------------------------------------------------------- section 01
def missingness(facts: dict) -> go.Figure:
    rows = [r for r in facts["missingness_table"] if r["n_missing"] > 0]
    rows.sort(key=lambda r: r["pct_missing"])
    fig = go.Figure(go.Bar(
        x=[r["pct_missing"] for r in rows],
        y=[r["feature"] for r in rows],
        orientation="h",
        marker_color=[CLAY if r["pct_missing"] > 20 else GREEN for r in rows],
        text=[f"{r['pct_missing']}%  ({r['n_missing']})" for r in rows],
        textposition="outside",
        textfont=dict(size=13),
        cliponaxis=False,
        name="Share missing",
        hovertemplate="%{y}<br>%{x}% of patients<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        height=330, showlegend=False,
        xaxis=dict(title="Share of patients with no measurement (%)",
                   range=[0, 62], gridcolor=GRID, zerolinecolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(tickfont=dict(size=13.5, color=INK), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=90, t=8, b=44),
    ))
    return fig


def comissingness(facts: dict) -> go.Figure:
    labels = facts["comissingness"]["labels"]
    matrix = facts["comissingness"]["matrix"]
    peak = max(max(r) for r in matrix)
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale=[[0, "#f4f1e8"], [1, GREEN]],
        showscale=False,
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="%{y} and %{x}<br>%{z} patients missing both<extra></extra>",
        name="Patients missing both",
    ))
    fig.update_layout(**base_layout(
        height=400,
        xaxis=dict(tickangle=-35, tickfont=dict(size=12, color=INK_SOFT), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color=INK_SOFT), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=8, t=8, b=80),
    ))
    return fig


# ---------------------------------------------------------------- section 02
def effect_sizes(facts: dict) -> go.Figure:
    rows = sorted(facts["group_comparison"], key=lambda r: r["cohens_d"])
    fig = go.Figure(go.Bar(
        x=[r["cohens_d"] for r in rows],
        y=[r["feature"] for r in rows],
        orientation="h",
        marker_color=[GREEN if r["cohens_d"] >= 0.8 else INK_SOFT for r in rows],
        text=[f"{r['cohens_d']:.2f}" for r in rows],
        textposition="outside",
        textfont=dict(size=13),
        cliponaxis=False,
        name="Cohen's d",
        hovertemplate="%{y}<br>d = %{x:.2f}<extra></extra>",
    ))
    # Every bar starts at zero, so a band label placed anywhere inside the plot
    # sits on top of one. They go above the plot area instead. Only the "large"
    # threshold is drawn: it is the one the text refers to, and three labels
    # cannot be spaced apart at 375px however they are stacked.
    for edge in (0.2, 0.5):
        fig.add_vline(x=edge, line=dict(color=GRID, width=1, dash="dot"))
    fig.add_vline(x=0.8, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.add_annotation(x=0.8, y=1.0, yref="paper", text="Large effect",
                       showarrow=False, xanchor="left", xshift=4, yanchor="bottom",
                       font=dict(size=12, color=INK_SOFT))
    fig.update_layout(**base_layout(
        height=400, showlegend=False,
        xaxis=dict(title="Standardized difference between the two groups (Cohen's d)",
                   range=[0, 1.42], gridcolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(tickfont=dict(size=13, color=INK), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=56, t=30, b=48),
    ))
    return fig


def prevalence(facts: dict) -> go.Figure:
    rows = facts["prevalence"]["Glucose"]
    base = facts["overall_diabetes_rate"]
    fig = go.Figure(go.Scatter(
        x=[r["midpoint"] for r in rows],
        y=[100 * r["diabetes_rate"] for r in rows],
        mode="lines+markers+text",
        line=dict(color=GREEN, width=3),
        marker=dict(size=11, color=GREEN, line=dict(color="#fdfcf8", width=2)),
        text=[f"{100 * r['diabetes_rate']:.0f}%" for r in rows],
        textposition="top center",
        textfont=dict(size=13, color=INK),
        cliponaxis=False,
        name="Diabetes rate",
        hovertemplate="Glucose near %{x:.0f}<br>%{y:.1f}% have diabetes<extra></extra>",
    ))
    fig.add_hline(y=100 * base, line=dict(color=GOLD, width=1.5, dash="dash"))
    fig.add_annotation(x=0.02, xref="paper", y=100 * base, yanchor="bottom",
                       text=f"Cohort average, {100 * base:.0f}%", showarrow=False,
                       xanchor="left", font=dict(size=12, color=GOLD))
    fig.update_layout(**base_layout(
        height=400, showlegend=False,
        xaxis=dict(title="Blood glucose (mg/dL), grouped into six equal-sized bins",
                   gridcolor=GRID, tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(title="Patients with diabetes (%)", range=[0, 95], gridcolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        margin=dict(l=8, r=16, t=26, b=48),
    ))
    return fig


# ---------------------------------------------------------------- section 04
def leaderboard(facts: dict) -> go.Figure:
    rows = [r for r in facts["test"] if r["model"] != "majority_class"]
    rows.sort(key=lambda r: r["roc_auc"])
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(
            x=[r["roc_auc"]], y=[MODEL_LABEL[r["model"]]],
            mode="markers",
            marker=dict(size=15, color=MODEL_COLOR[r["model"]],
                        line=dict(color="#fdfcf8", width=2)),
            error_x=dict(type="data", symmetric=False,
                         array=[r["ci_high"] - r["roc_auc"]],
                         arrayminus=[r["roc_auc"] - r["ci_low"]],
                         color=MODEL_COLOR[r["model"]], thickness=2.5, width=7),
            name=MODEL_LABEL[r["model"]],
            showlegend=False,
            hovertemplate=(f"{MODEL_LABEL[r['model']]}<br>Score %{{x:.3f}}"
                           f"<br>95% range {r['ci_low']:.2f} to {r['ci_high']:.2f}<extra></extra>"),
        ))
        fig.add_annotation(x=r["ci_high"], y=MODEL_LABEL[r["model"]],
                           text=f"{r['roc_auc']:.3f}", showarrow=False,
                           xanchor="left", xshift=10,
                           font=dict(size=13, color=INK))
    fig.add_vline(x=0.5, line=dict(color=INK_SOFT, width=1, dash="dot"))
    fig.add_annotation(x=0.5, y=-0.7, text="Coin flip", showarrow=False,
                       xanchor="left", xshift=4, yanchor="bottom",
                       font=dict(size=12, color=INK_SOFT))
    fig.update_layout(**base_layout(
        height=360, showlegend=False,
        xaxis=dict(title="Score on the 154 held-out patients (ROC AUC), with its 95% range",
                   range=[0.44, 1.0], gridcolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(tickfont=dict(size=13.5, color=INK), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=56, t=8, b=52),
    ))
    return fig


# ---------------------------------------------------------------- section 05
def threshold(facts: dict) -> go.Figure:
    s = facts["threshold_sweep"]
    fig = go.Figure()
    for key, label, color in (("precision", "Precision", BLUE),
                              ("recall", "Recall", CLAY),
                              ("f1", "Balance of the two (F1)", GREEN)):
        fig.add_trace(go.Scatter(
            x=s["threshold"], y=s[key], mode="lines", name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f"{label} %{{y:.2f}} at cut-off %{{x:.2f}}<extra></extra>",
        ))
    # The four rules land within about six hundredths of each other, so one
    # label each collides at every width. The band carries the same information
    # and the exact cut-offs are in the table beside this figure.
    values = [round(v, 4) for v in facts["threshold_rules"].values()]
    low, high = min(values), max(values)
    # stop the band short of the headroom so it does not run through its own label
    fig.add_vrect(x0=low, x1=high, y1=0.85, fillcolor=GREEN, opacity=0.13,
                  line_width=0, layer="below")
    for value in sorted(set(values)):
        fig.add_vline(x=value, y1=0.85, line=dict(color=GREEN, width=1, dash="dot"))
    # Headroom above the curves keeps this label inside the plot; above the plot
    # it sits in the legend row and lands on top of an entry.
    fig.add_annotation(x=(low + high) / 2, y=1.16, yanchor="top",
                       text=f"All four rules land here<br>{low:.2f} to {high:.2f}",
                       showarrow=False, xanchor="center", align="center",
                       font=dict(size=12, color=GREEN))
    fig.add_vline(x=0.5, y1=0.85, line=dict(color=GOLD, width=1.5))
    # low and to the right of the line: the three curves all sit above 0.5 here
    fig.add_annotation(x=0.5, y=0.32, yanchor="top", text="Library<br>default<br>0.50",
                       showarrow=False, xanchor="left", xshift=7, align="left",
                       font=dict(size=12, color=GOLD))
    fig.update_layout(**base_layout(
        height=430,
        # only one axis labels the origin: two zeros a few pixels apart read as
        # a rendering fault rather than as two scales
        xaxis=dict(title="Cut-off: the predicted risk above which a patient is flagged",
                   range=[0, 1], gridcolor=GRID,
                   tickvals=[0.2, 0.4, 0.6, 0.8, 1],
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(range=[0, 1.2], gridcolor=GRID,
                   tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                   tickfont=dict(size=12.5, color=INK_SOFT)),
        margin=dict(l=8, r=16, t=54, b=48),
    ))
    return fig


# ---------------------------------------------------------------- section 06
def pca_panels(facts: dict) -> go.Figure:
    from plotly.subplots import make_subplots

    # A shared legend cannot work here: green means "group 1" on the left and
    # "no diabetes" on the right, so one row of four keys leaves the reader
    # guessing which two belong to which panel. Each panel carries its own key.
    def key(*pairs: tuple[str, str]) -> str:
        return "&nbsp;&nbsp;&nbsp;".join(
            f'<span style="color:{c}">&#9679; {label}</span>' for label, c in pairs)

    p = facts["pca_points"]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.07,
        subplot_titles=(
            "Colored by the group the algorithm found<br>"
            + key(("Group 1", GREEN), ("Group 2", VIOLET)),
            "Colored by the true class<br>"
            + key(("No diabetes", GREEN), ("Has diabetes", CLAY)),
        ))
    for value, name, color in ((0, "Group 1", GREEN), (1, "Group 2", VIOLET)):
        idx = [i for i, c in enumerate(p["cluster"]) if c == value]
        fig.add_trace(go.Scatter(
            x=[p["x"][i] for i in idx], y=[p["y"][i] for i in idx],
            mode="markers", name=name, legendgroup=name,
            marker=dict(size=6, color=color, opacity=0.75,
                        line=dict(color="#fdfcf8", width=0.5)),
            hovertemplate=f"{name}<extra></extra>"), row=1, col=1)
    for value, name, color in ((0, "No diabetes", GREEN), (1, "Has diabetes", CLAY)):
        idx = [i for i, c in enumerate(p["outcome"]) if c == value]
        fig.add_trace(go.Scatter(
            x=[p["x"][i] for i in idx], y=[p["y"][i] for i in idx],
            mode="markers", name=name, legendgroup=name,
            marker=dict(size=6, color=color, opacity=0.75,
                        line=dict(color="#fdfcf8", width=0.5)),
            hovertemplate=f"{name}<extra></extra>"), row=1, col=2)
    fig.update_layout(**base_layout(height=420, showlegend=False,
                                    margin=dict(l=8, r=8, t=68, b=16)))
    # showline off on both: base_layout only reaches the first subplot's axes,
    # so without this the left panel gets a border and the right one does not
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False)
    for a in fig.layout.annotations:
        a.font = dict(size=13.5, color=INK)
    return fig


# ---------------------------------------------------------------- section 07
def scaling_trap(facts: dict) -> go.Figure:
    groups = ["How tidy the groups look<br>(silhouette score)",
              "How much they match diabetes<br>(agreement score)"]
    raw = [facts["silhouette_raw"], facts["ari_raw"]]
    std = [facts["silhouette_std"], facts["ari_std"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups, y=raw, name="Raw measurements", marker_color=CLAY,
        text=[f"{v:.3f}" for v in raw], textposition="outside",
        textfont=dict(size=14), cliponaxis=False,
        hovertemplate="Raw measurements<br>%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(
        x=groups, y=std, name="Put on a common scale", marker_color=GREEN,
        text=[f"{v:.3f}" for v in std], textposition="outside",
        textfont=dict(size=14), cliponaxis=False,
        hovertemplate="Common scale<br>%{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(
        height=430, bargap=0.42, bargroupgap=0.08,
        xaxis=dict(tickfont=dict(size=13, color=INK), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Score", range=[0, 0.85], gridcolor=GRID,
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        margin=dict(l=8, r=8, t=38, b=58),
    ))
    return fig


# ---------------------------------------------------------------- section 08
def importances(facts: dict) -> go.Figure:
    rows = sorted(facts["importances"], key=lambda r: r["logistic_regression"])
    fig = go.Figure()
    for key, color in (("logistic_regression", GREEN), ("random_forest", VIOLET)):
        fig.add_trace(go.Bar(
            x=[r[key] for r in rows], y=[r["feature"] for r in rows],
            orientation="h", name=MODEL_LABEL[key], marker_color=color,
            hovertemplate="%{y}<br>" + MODEL_LABEL[key] + " %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK_SOFT, width=1))
    fig.update_layout(**base_layout(
        height=430, barmode="group", bargap=0.3, bargroupgap=0.08,
        xaxis=dict(title="How much the score drops when this measurement is scrambled",
                   gridcolor=GRID, tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(tickfont=dict(size=13, color=INK), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=16, t=38, b=48),
    ))
    return fig


def coefficients(facts: dict) -> go.Figure:
    rows = [r for r in facts["coefficients"] if not r["feature"].startswith("missingindicator")]
    rows.sort(key=lambda r: r["coefficient"])
    fig = go.Figure()
    for r in rows:
        solid = r["excludes_zero"]
        color = GREEN if solid else INK_SOFT
        fig.add_trace(go.Scatter(
            x=[r["coefficient"]], y=[r["feature"]], mode="markers",
            marker=dict(size=13, color=color, line=dict(color="#fdfcf8", width=2)),
            error_x=dict(type="data", symmetric=False,
                         array=[r["ci_high"] - r["coefficient"]],
                         arrayminus=[r["coefficient"] - r["ci_low"]],
                         color=color, thickness=2.2, width=6),
            name="Clear of zero" if solid else "Could be zero",
            legendgroup="solid" if solid else "unclear",
            showlegend=False,
            hovertemplate=(f"{r['feature']}<br>Effect %{{x:.2f}}"
                           f"<br>95% range {r['ci_low']:.2f} to {r['ci_high']:.2f}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(color=CLAY, width=1.5))
    fig.add_annotation(x=0, y=-0.72, text="No effect", showarrow=False,
                       xanchor="left", xshift=5, yanchor="bottom",
                       font=dict(size=12, color=CLAY))
    for label, color, y in (("Clear of zero", GREEN, 1), ("Range crosses zero", INK_SOFT, 0)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=color),
                                 name=label, showlegend=True))
    fig.update_layout(**base_layout(
        height=400,
        xaxis=dict(title="Effect on the odds of diabetes, per standard deviation",
                   gridcolor=GRID, tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(tickfont=dict(size=13, color=INK), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=24, t=38, b=52),
    ))
    return fig


def patient(facts: dict) -> go.Figure:
    rows = [r for r in facts["patient_confident"]["contributions"]
            if abs(r["contribution"]) > 0.02 and not r["feature"].startswith("missingindicator")]
    rows.sort(key=lambda r: r["contribution"])
    fig = go.Figure(go.Bar(
        x=[r["contribution"] for r in rows],
        y=[r["feature"] for r in rows],
        orientation="h",
        marker_color=[CLAY if r["contribution"] > 0 else BLUE for r in rows],
        text=[f"{r['contribution']:+.2f}" for r in rows],
        textposition="outside", textfont=dict(size=13), cliponaxis=False,
        name="Contribution",
        # per-bar colors give this trace no single swatch, so Plotly would key it
        # with the last bar's color; the two entries below carry the legend
        showlegend=False,
        hovertemplate="%{y}<br>%{x:+.2f} toward the prediction<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=INK_SOFT, width=1))
    for label, color in (("Pushes toward diabetes", CLAY), ("Pushes away", BLUE)):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=label))

    # Value labels sit outside the bar end, so the axis needs asymmetric
    # padding. Without it a short negative bar pushes its label to the left,
    # into the feature names on the axis.
    values = [r["contribution"] for r in rows] + [0.0]
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)

    fig.update_layout(**base_layout(
        height=360, showlegend=True,
        xaxis=dict(title="Push toward the prediction (log-odds)", gridcolor=GRID,
                   range=[low - 0.55 * span, high + 0.16 * span],
                   tickfont=dict(size=12.5, color=INK_SOFT),
                   title_font=dict(size=13, color=INK_SOFT)),
        yaxis=dict(tickfont=dict(size=13, color=INK), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=60, t=38, b=48),
    ))
    return fig


FIGURES = {
    "missingness": missingness,
    "comissingness": comissingness,
    "effect_sizes": effect_sizes,
    "prevalence": prevalence,
    "leaderboard": leaderboard,
    "threshold": threshold,
    "pca": pca_panels,
    "scaling_trap": scaling_trap,
    "importances": importances,
    "coefficients": coefficients,
    "patient": patient,
}
