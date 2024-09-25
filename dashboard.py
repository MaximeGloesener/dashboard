import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_dataframe(data):
    rows = []
    for model_size, model_data in data.items():
        model_info = model_data.get('model_info', {})
        for precision, metrics in model_data.items():
            if precision not in ['model', 'model_info']:
                row = {
                    'Model Size': model_size,
                    'Precision': precision,
                    'Layers': model_info.get('layers', 0),
                    'Parameters (M)': model_info.get('params', 0) / 1e6,
                    'FLOPS (B)': model_info.get('flops', 0),
                    'FPS (GPU)': metrics.get('fps_gpu', 0),
                    'Avg Emissions (gCO2eq)': metrics.get('avg_emissions', 0) * 1e6,
                    'Avg Energy (mWh)': metrics.get('avg_energy', 0) * 1e6,
                    'mAP@0.5': metrics.get('map50', 0),
                    'mAP@0.5:0.95': metrics.get('map5095', 0)
                }
                rows.append(row)
    return pd.DataFrame(rows)

def load_all_data(hardware):
    model_sizes = ['n', 's', 'm', 'l', 'x']  # Add or modify as needed
    data = {}
    for size in model_sizes:
        file_path = f'results/{hardware}/yolo_{size}_results.json'  # Adjust file naming convention if needed
        data[size] = load_data(file_path)
    return prepare_dataframe(data)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'
])
server = app.server

# Define custom styles
colors = {
    'background': '#f4f4f4',
    'text': '#333333',
    'primary': '#007bff',
    'secondary': '#6c757d'
}

# Define metrics for dropdowns
metrics = [
    'FPS (GPU)', 'Avg Emissions (gCO2eq)', 'Avg Energy (mWh)',
    'mAP@0.5', 'mAP@0.5:0.95', 'Layers', 'Parameters (M)', 'FLOPS (B)'
]
model_sizes = ['n', 's', 'm', 'l', 'x']

# Load initial data for default hardware
df = load_all_data('4090')

# Define the layout with improved styling
app.layout = html.Div([
    html.H1("YOLO Model Comparison Dashboard",
            style={'textAlign': 'center', 'color': colors['primary'], 'fontFamily': 'Roboto, sans-serif', 'marginBottom': '30px'}),

    html.Div([
        html.Label("Select Hardware:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
        dcc.Dropdown(
            id='hardware-dropdown',
            options=[
                {'label': 'RTX 4090', 'value': '4090'},
                {'label': 'Jetson Xavier', 'value': 'jetson'}
            ],
            value='4090',
            style={'width': '100%', 'marginBottom': '20px'}
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.Div([
            html.Label("Select Metric:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': metric, 'value': metric} for metric in metrics],
                value='FPS (GPU)',
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),

        html.Div([
            html.Label("Select Model Size:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='model-size-dropdown',
                options=[{'label': size.upper(), 'value': size} for size in model_sizes],
                value=model_sizes,
                multi=True,
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),

        html.Div([
            html.Label("Select Precision:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='precision-dropdown',
                options=[{'label': prec, 'value': prec} for prec in df['Precision'].unique()],
                value=df['Precision'].unique(),
                multi=True,
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}),

    dcc.Graph(id='bar-chart'),

    html.H2("Multi-Metric Comparison", style={'textAlign': 'center', 'color': colors['secondary'], 'fontFamily': 'Roboto, sans-serif', 'marginTop': '40px', 'marginBottom': '20px'}),

    html.Div([
        html.Div([
            html.Label("Select X-axis Metric:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': metric, 'value': metric} for metric in metrics],
                value='FPS (GPU)',
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '10%'}),

        html.Div([
            html.Label("Select Y-axis Metric:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': metric, 'value': metric} for metric in metrics],
                value='mAP@0.5:0.95',
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}),

    dcc.Graph(id='scatter-plot'),

    html.H2("Metrics Table", style={'textAlign': 'center', 'color': colors['secondary'], 'fontFamily': 'Roboto, sans-serif', 'marginTop': '40px', 'marginBottom': '20px'}),

    dash_table.DataTable(
        id='metrics-table',
        columns=[
            {"name": col, "id": col} for col in df.columns
        ],
        data=df.to_dict('records'),
        sort_action="native",
        sort_mode="multi",
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Roboto, sans-serif',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': colors['background']
            }
        ]
    )
], style={'padding': '20px', 'fontFamily': 'Roboto, sans-serif', 'backgroundColor': 'white'})

@app.callback(
    [Output('bar-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('metrics-table', 'data')],
    [Input('hardware-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('model-size-dropdown', 'value'),
     Input('precision-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_dashboard(hardware, selected_metric, selected_sizes, selected_precisions, x_metric, y_metric):
    global df
    df = load_all_data(hardware)
    # Round the values in the DataFrame
    for column in df.columns:
        if column in ['Avg Energy (mWh)', 'Avg Emissions (gCO2eq)', 'mAP@0.5', 'mAP@0.5:0.95']:
            df[column] = df[column].round(4)
        elif column not in ['Model Size', 'Precision']:
            df[column] = df[column].round(2)

    filtered_df = df[
        (df['Model Size'].isin(selected_sizes)) &
        (df['Precision'].isin(selected_precisions))
    ]

    bar_fig = px.bar(
        filtered_df,
        x='Model Size',
        y=selected_metric,
        color='Precision',
        barmode='group',
        title=f'{selected_metric} Comparison',
        labels={selected_metric: selected_metric.replace('_', ' ').title(),
                'Model Size': 'YOLO Model Version'}
    )

    scatter_fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color='Model Size',
        symbol='Precision',
        hover_name="Model Size",
        hover_data=df.columns,
        labels={col: col for col in df.columns},
        title=f'{y_metric} vs {x_metric}',
    )

    # Increase marker size
    scatter_fig.update_traces(marker=dict(size=12))

    return bar_fig, scatter_fig, df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)