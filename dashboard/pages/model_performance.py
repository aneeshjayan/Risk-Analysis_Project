"""
Model performance page: ROC curves, metric comparison table, threshold tuning.
Owner: Kalaivani Ravichandran
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.graph_objects as go


_METRICS_PLACEHOLDER = [
    {"Model": "XGBoost", "AUC-ROC": "—", "F1": "—", "Precision": "—", "Recall": "—", "Brier": "—"},
    {"Model": "LightGBM", "AUC-ROC": "—", "F1": "—", "Precision": "—", "Recall": "—", "Brier": "—"},
    {"Model": "Random Forest", "AUC-ROC": "—", "F1": "—", "Precision": "—", "Recall": "—", "Brier": "—"},
]


def layout() -> dbc.Container:
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Model Performance"), width=12)),
        dbc.Row(dbc.Col(
            dash_table.DataTable(
                data=_METRICS_PLACEHOLDER,
                columns=[{"name": c, "id": c} for c in _METRICS_PLACEHOLDER[0]],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
                style_cell={"textAlign": "center"},
            ), width=12
        ), className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="roc-curve", figure=_placeholder_roc()), md=6),
            dbc.Col(dcc.Graph(id="precision-recall-curve", figure=_placeholder_fig("Precision-Recall Curve")), md=6),
        ]),
    ])


def _placeholder_roc() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="grey"))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                      template="plotly_white")
    return fig


def _placeholder_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white")
    return fig
