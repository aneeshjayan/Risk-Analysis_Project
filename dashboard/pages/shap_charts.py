"""
SHAP explainability page: global feature importance bar chart + local waterfall.
Owner: Kalaivani Ravichandran / Subramanian Raj Narayanan
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go


def layout() -> dbc.Container:
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("SHAP Explainability"), width=12)),
        dbc.Row(dbc.Col(html.P(
            "Select a model and a borrower index to see what drove the default prediction."
        ), width=12)),
        dbc.Row([
            dbc.Col([
                html.Label("Model"),
                dcc.Dropdown(
                    id="shap-model-select",
                    options=[
                        {"label": "XGBoost", "value": "xgboost"},
                        {"label": "LightGBM", "value": "lightgbm"},
                        {"label": "Random Forest", "value": "random_forest"},
                    ],
                    value="xgboost",
                    clearable=False,
                ),
            ], md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="shap-global-bar", figure=_placeholder_fig("Global Feature Importance")), md=6),
            dbc.Col(dcc.Graph(id="shap-waterfall", figure=_placeholder_fig("Local Explanation (Waterfall)")), md=6),
        ]),
        dbc.Row(dbc.Col(html.Div(id="llm-explanation", className="mt-3 p-3 border rounded bg-light"))),
    ])


def _placeholder_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white",
                      annotations=[dict(text="Run model first", showarrow=False,
                                        font=dict(size=14), xref="paper", yref="paper",
                                        x=0.5, y=0.5)])
    return fig
