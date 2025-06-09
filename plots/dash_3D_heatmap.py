import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import json
import numpy as np

# Load data
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

resource_data_2layer = load_data("../resource_calculated_2layer_data.json")
timing_data_2layer = load_data("../timing_calculated_2layer_data.json")
resource_data_conv2d = load_data("../resource_calculated_conv2d_data.json")
timing_data_conv2d = load_data("../timing_calculated_conv2d_data.json")

# Create Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Layout
app.layout = dbc.Container([
    html.H1("3D Scatter Plot Viewer", className="text-center text-light mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Dataset:", className="text-light"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {"label": "2layer", "value": "2layer"},
                    {"label": "conv2d", "value": "conv2d"}
                ],
                value="2layer",
                className="mb-3"
            ),
        ], width=6),
        dbc.Col([
            html.Label("Select Data Type:", className="text-light"),
            dcc.Dropdown(
                id="data-type-dropdown",
                options=[
                    {"label": "Resource Data", "value": "resource"},
                    {"label": "Timing Data", "value": "timing"}
                ],
                value="resource",
                className="mb-3"
            ),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Resource/Timing Type:", className="text-light"),
            dcc.Dropdown(
                id="value-axis-dropdown",
                options=[],  # Options will be populated dynamically
                value=None,
                className="mb-3"
            ),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="3d-scatter-plot")
        ], width=12),
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output("value-axis-dropdown", "options"),
    Output("value-axis-dropdown", "value"),
    Input("dataset-dropdown", "value"),
    Input("data-type-dropdown", "value")
)
def update_value_axis_options(dataset, data_type):
    if dataset == "2layer":
        data = resource_data_2layer if data_type == "resource" else timing_data_2layer
    else:
        data = resource_data_conv2d if data_type == "resource" else timing_data_conv2d

    options = [{"label": key, "value": key} for key in data.keys()]
    return options, options[0]["value"] if options else None

@app.callback(
    Output("3d-scatter-plot", "figure"),
    Input("dataset-dropdown", "value"),
    Input("data-type-dropdown", "value"),
    Input("value-axis-dropdown", "value")
)
def update_3d_scatter_plot(dataset, data_type, value_axis):
    if not value_axis:
        return {}

    if dataset == "2layer":
        data = resource_data_2layer if data_type == "resource" else timing_data_2layer
    else:
        data = resource_data_conv2d if data_type == "resource" else timing_data_conv2d

    selected_data = data[value_axis]

    df = pd.DataFrame({
        "ops": np.array(selected_data["ops"]),
        "bops": np.array(selected_data["bops"]),
        "values": np.array(selected_data["values"]),
        "bitwidth": np.array(selected_data["bitwidth"]),
        "rf": np.array(selected_data["rf"])
    })

    fig = px.scatter_3d(
        df, x="bops", y="rf", z="values",
        color="bitwidth",
        title=f"3D Scatter Plot for {value_axis} ({dataset})",
        labels={"bops": "BOPs", "rf": "Reuse Factor", "values": value_axis}
    )
    fig.update_traces(marker_size=4)

    # Apply dark mode theme
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="BOPs"),
            yaxis=dict(title="Reuse Factor"),
            zaxis=dict(title=value_axis)
        )
    )
    return fig

# Run app
if __name__ == "__main__":
    app.run(debug=True)