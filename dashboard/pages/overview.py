"""
Overview page: dataset summary, class distribution, key statistics.
Owner: Kalaivani Ravichandran
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go


def layout() -> dbc.Container:
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Dataset Overview"), width=12)),
        dbc.Row([
            dbc.Col(_stat_card("Total Loans", "—"), md=3),
            dbc.Col(_stat_card("Default Rate", "—"), md=3),
            dbc.Col(_stat_card("Features", "—"), md=3),
            dbc.Col(_stat_card("Train / Test Split", "80 / 20"), md=3),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="class-distribution-chart", figure=_placeholder_fig("Class Distribution")), md=6),
            dbc.Col(dcc.Graph(id="fico-distribution-chart", figure=_placeholder_fig("FICO Score Distribution")), md=6),
        ]),
    ])


def _stat_card(title: str, value: str) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H5(title, className="card-title text-muted"),
        html.H3(value, className="card-text"),
    ]), className="shadow-sm")


def _placeholder_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white",
                      annotations=[dict(text="Data not loaded", showarrow=False,
                                        font=dict(size=16), xref="paper", yref="paper",
                                        x=0.5, y=0.5)])
    return fig
