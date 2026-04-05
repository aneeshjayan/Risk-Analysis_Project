"""
Main Dash application entry point.
Wires together the Overview, Model Performance, and SHAP Charts pages.
Owner: Kalaivani Ravichandran
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from pages import overview, model_performance, shap_charts

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "Credit Risk Decision Intelligence"

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Location(id="url", refresh=False),
        dbc.NavbarSimple(
            brand="Credit Risk Decision Intelligence | West Virginia Team",
            brand_href="/",
            color="primary",
            dark=True,
            children=[
                dbc.NavItem(dbc.NavLink("Overview", href="/")),
                dbc.NavItem(dbc.NavLink("Model Performance", href="/performance")),
                dbc.NavItem(dbc.NavLink("SHAP Explainability", href="/shap")),
            ],
        ),
        html.Div(id="page-content", className="mt-4"),
    ],
)


# --------------------------------------------------------------------------- #
# Router callback
# --------------------------------------------------------------------------- #
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname: str):
    if pathname == "/performance":
        return model_performance.layout()
    if pathname == "/shap":
        return shap_charts.layout()
    return overview.layout()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
