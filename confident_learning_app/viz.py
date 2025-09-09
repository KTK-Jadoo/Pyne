from typing import List, Optional

import numpy as np
import plotly.graph_objects as go


def confident_joint_heatmap(cj: np.ndarray, class_names: List[str]) -> go.Figure:
    """Plot heatmap of confident joint: rows=given labels, cols=inferred.

    If `cj` is None, returns an empty figure with a message.
    """
    if cj is None:
        fig = go.Figure()
        fig.add_annotation(text="Confident joint unavailable", showarrow=False)
        fig.update_layout(height=400)
        return fig

    z = cj.astype(int)
    fig = go.Figure(
        data=
        go.Heatmap(
            z=z,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            hovertemplate="Given: %{y}<br>Inferred: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confident Joint (Given vs Inferred)",
        xaxis_title="Inferred Label",
        yaxis_title="Given Label",
        yaxis_autorange="reversed",
        height=500,
    )
    return fig


def uncertainty_scatter(
    self_confidence: np.ndarray, alt_prob: np.ndarray, is_issue: Optional[np.ndarray] = None
) -> go.Figure:
    """Scatter of self-confidence vs best alternative probability."""
    if is_issue is None:
        is_issue = np.zeros_like(self_confidence, dtype=bool)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=self_confidence[~is_issue],
            y=alt_prob[~is_issue],
            mode="markers",
            name="Clean",
            marker=dict(color="rgba(38, 173, 129, 0.7)", size=6),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=self_confidence[is_issue],
            y=alt_prob[is_issue],
            mode="markers",
            name="Suspicious",
            marker=dict(color="rgba(220, 64, 64, 0.8)", size=7),
        )
    )
    fig.update_layout(
        title="Uncertainty Scatter (Self-confidence vs. Best Alternative)",
        xaxis_title="Self-confidence (p[given label])",
        yaxis_title="Best alternative probability",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="gray", dash="dot"),
    )
    return fig

