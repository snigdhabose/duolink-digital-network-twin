# visualizer.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import networkx as nx
import random
import numpy as np
from src.predictor import predict_failures, suggest_reroutes
from src.rl_agent import QLearningAgent
import joblib

# Load the trained RL agent
try:
    rl_agent = joblib.load('models/rl_agent.pkl')
    print("RL agent loaded successfully.")
except FileNotFoundError:
    print("Error: RL agent model file not found. Ensure it is trained and saved.")
    rl_agent = None  # Set to None if not found

# Initialize the Dash app with Font Awesome and custom CSS
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
        "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"  # Optional, for better alignment
    ]
)
# Graph variables
device_count = 15
G = nx.erdos_renyi_graph(device_count, 0.3, seed=42)
pos = nx.spring_layout(G, seed=42)
nx.set_node_attributes(G, 'active', 'status')

# Create another graph for the "Real Network Status" view
G_real = G.copy()
nx.set_node_attributes(G_real, 'active', 'status')


def create_network_data(failed_nodes, reroute_paths):
    # Update failed nodes' status, only if they exist in the graph
    for node in G.nodes:
        if node in failed_nodes:
            G.nodes[node]['status'] = 'failed'
        else:
            G.nodes[node]['status'] = 'active'

    edge_trace = []
    reroute_edge_trace = []
    failed_edge_trace = []

    for edge in G.edges():
        if edge in reroute_paths or (edge[1], edge[0]) in reroute_paths:
            reroute_edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='green'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
        elif G.nodes[edge[0]]['status'] == 'failed' or G.nodes[edge[1]]['status'] == 'failed':
            failed_edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='red'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
        else:
            edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[f"Device {node}" for node in G.nodes()],
        marker=dict(
            color=['red' if G.nodes[node]['status'] == 'failed' else 'green' for node in G.nodes()],
            size=20,
            line=dict(width=2, color='black')
        ),
        textposition='top center',
        hoverinfo='text',
        showlegend=False
    )
    return edge_trace, reroute_edge_trace, failed_edge_trace, node_trace

# Function to create visualization data for Graph 2 (Real Network Status)
def create_real_network_data(package_flow, failed_nodes):
    # Update failed nodes' status in G_real
    for node in G_real.nodes:
        if node in failed_nodes:
            G_real.nodes[node]['status'] = 'failed'
        else:
            G_real.nodes[node]['status'] = 'active'

    edge_trace = []
    package_flow_trace = []
    failed_edge_trace = []

    for edge in G_real.edges():
        if edge in package_flow or (edge[1], edge[0]) in package_flow:
            package_flow_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='blue'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
        elif G_real.nodes[edge[0]]['status'] == 'failed' or G_real.nodes[edge[1]]['status'] == 'failed':
            failed_edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='red'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
        else:
            edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G_real.nodes()],
        y=[pos[node][1] for node in G_real.nodes()],
        mode='markers+text',
        text=[f"Device {node}" for node in G_real.nodes()],
        marker=dict(
            color=['red' if G_real.nodes[node]['status'] == 'failed' else 'green' for node in G_real.nodes()],
            size=20,
            line=dict(width=2, color='black')
        ),
        textposition='top center',
        hoverinfo='text',
        showlegend=False
    )
    return edge_trace, package_flow_trace, failed_edge_trace, node_trace

# App layout

app.layout = html.Div([
    html.H2("Digital Twin Network Dashboard", style={
        'font-family': 'Arial, sans-serif',
        'font-size': '28px',
        'font-weight': 'bold',
        'color': '#4A4A4A',
        'text-align': 'center',
        'margin-top': '20px'
    }),
    html.Div(
        html.Button(
            id="pause-button",
            n_clicks=0,
            children=[html.Span(className="fas fa-pause"), "Pause"],
            style={
                'font-size': '18px',
                'margin-top': '10px',
                'padding': '5px 10px',
                'border-radius': '8px',
                'display': 'inline-flex',
                'align-items': 'center',
                'justify-content': 'center',
                'background-color': '#007bff',
                'color': 'white',
                'border': 'none',
                'cursor': 'pointer',
                'margin-right': '90px',
                
            }
        ),
        style={'text-align': 'right'}
    ),
    html.Div([
        html.Div([
            dcc.Graph(
                id='predicted-graph',
                config={'displayModeBar': True, 'scrollZoom': True},
                style={'height': '500px'}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='real-graph',
                config={'displayModeBar': True, 'scrollZoom': True},
                style={'height': '500px'}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
    ]),
    
    # Custom legend with graphical elements
    html.Div([
        html.H4("Legend", style={
            'font-size': '22px',
            'font-weight': 'bold',
            'color': '#4A4A4A',
            
            'text-align': 'center'
        }),
        html.Div([
            html.Div([
                html.Span("—", style={"color": "green", "fontSize": "18px"}), " Rerouted Path (Graph 1)"
            ]),
            html.Div([
                html.Span("—", style={"color": "red", "fontSize": "18px"}), " Failed Connection"
            ]),
            html.Div([
                html.Span("—", style={"color": "gray", "fontSize": "18px"}), " Active Connection"
            ]),
            html.Div([
                html.Span("—", style={"color": "blue", "fontSize": "18px"}), " Package Flow (Graph 2)"
            ]),
            html.Div([
                html.Span("●", style={"color": "green", "fontSize": "18px"}), " Active Device"
            ]),
            html.Div([
                html.Span("●", style={"color": "red", "fontSize": "18px"}), " Failed Device"
            ]),
        ], style={
            'font-size': '16px',
            'line-height': '1.3',
            'text-align': 'center',
            
        })
    ], style={
        'width': '50%',
        'text-align': 'center',
        'margin': '20px auto',
        'border': '1px solid #ddd',
        'padding': '15px',
        'background-color': '#f1f1f1',
        'border-radius': '10px',
        'font-family': 'Arial, sans-serif',
        'margin-top': '100px',
    }),
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
])

# Callback to toggle the interval (pause/play functionality)
@app.callback(
    [Output('interval-component', 'disabled'), Output('pause-button', 'children')],
    [Input('pause-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_pause_play(n_clicks, is_paused):
    if n_clicks % 2 == 0:
        # Resume (show pause icon)
        return False, [html.Span(className="fas fa-pause"), " Pause"]
    else:
        # Pause (show play icon)
        return True, [html.Span(className="fas fa-play"), " Play"]

# Global variables
previous_failed_nodes_graph1 = set()
node_features = {}  # Initialize node_features globally

# Initialize counters outside of the callback function to keep track over time
total_actual_failures = 0
total_correct_predictions = 0
total_missed_predictions = 0
total_predicted_failures = 0

def simulate_features(node, base_failure_probability=0.2):
    """
    Simulate features for a node based on a base failure probability.
    Nodes with higher failure probability have features indicating potential failure.
    """
    failure_risk = random.random()
    if failure_risk < base_failure_probability:
        # High risk of failure; generate features similar to attack instances
        feature_vector = np.random.uniform(0.6, 1.0, size=41)
    else:
        # Low risk of failure; generate features similar to normal instances
        feature_vector = np.random.uniform(0.0, 0.4, size=41)
    return feature_vector

# Visualization functions remain the same

@app.callback(
    Output('predicted-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_network(n_intervals):
    global previous_failed_nodes_graph1, node_features

    # Generate features for each node and store in global node_features
    node_features = {node: simulate_features(node) for node in G.nodes()}

    # Predict failures
    failed_nodes_graph1 = set(predict_failures(node_features, threshold=0.7))

    # Suggest reroute paths
    reroute_paths = suggest_reroutes(G, failed_nodes_graph1, rl_agent)

    edge_trace, reroute_edge_trace, failed_edge_trace, node_trace = create_network_data(failed_nodes_graph1, reroute_paths)

    # Store the current predicted failed nodes for next comparison
    previous_failed_nodes_graph1 = failed_nodes_graph1

    fig = go.Figure(data=edge_trace + reroute_edge_trace + failed_edge_trace + [node_trace],
                    layout=go.Layout(
                        title='Predicted Network Status',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600, width=800
                    ))
    fig.update_layout(showlegend=False)
    return fig


@app.callback(
    Output('real-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_real_network(n_intervals):
    global previous_failed_nodes_graph1, node_features
    global total_actual_failures, total_correct_predictions, total_missed_predictions, total_predicted_failures

    # Simulate dynamic package flow by selecting a random subset of edges
    active_edges = list(G_real.edges())
    package_flow = random.sample(active_edges, k=int(len(active_edges) * 0.3))  # 30% of edges carry data

    failed_nodes_graph2 = set()
    for node in G_real.nodes:
        feature_vector = node_features[node]
        avg_feature_value = np.mean(feature_vector)
        if avg_feature_value > 0.7:
            failed_nodes_graph2.add(node)

    # Update Graph 2 node statuses
    for node in G_real.nodes:
        G_real.nodes[node]['status'] = 'failed' if node in failed_nodes_graph2 else 'active'

    edge_trace, package_flow_trace, failed_edge_trace, node_trace = create_real_network_data(package_flow, failed_nodes_graph2)

    # Compare predicted failures with actual failures and calculate metrics
    early_predictions = previous_failed_nodes_graph1.intersection(failed_nodes_graph2)
    missed_predictions = failed_nodes_graph2 - previous_failed_nodes_graph1
    false_positives = previous_failed_nodes_graph1 - failed_nodes_graph2

    # Update counters
    total_actual_failures += len(failed_nodes_graph2)
    total_correct_predictions += len(early_predictions)
    total_missed_predictions += len(missed_predictions)
    total_predicted_failures += len(previous_failed_nodes_graph1)

    # Calculate metrics
    accuracy = (total_correct_predictions / total_actual_failures) * 100 if total_actual_failures > 0 else 0
    recall = (total_correct_predictions / total_actual_failures) * 100 if total_actual_failures > 0 else 0
    precision = (total_correct_predictions / total_predicted_failures) * 100 if total_predicted_failures > 0 else 0
    false_positive_rate = (len(false_positives) / len(G.nodes)) * 100 if len(G.nodes) > 0 else 0

    # Print comparison results
    print(f"Time Step {n_intervals}:")
    print(f"Actual Failed Nodes (Graph 2): {failed_nodes_graph2}")
    print(f"Predicted Failed Nodes (Graph 1): {previous_failed_nodes_graph1}")
    print(f"Correct Predictions Before Failure: {early_predictions}")
    print(f"Missed Predictions: {missed_predictions}")
    print(f"False Positives: {false_positives}")
    print(f"Accuracy so far: {accuracy:.2f}%")
    print(f"Recall so far: {recall:.2f}%")
    print(f"Precision so far: {precision:.2f}%")
    print(f"False Positive Rate: {false_positive_rate:.2f}%")
    print("------------------------------------------------")

    fig = go.Figure(data=edge_trace + package_flow_trace + failed_edge_trace + [node_trace],
                    layout=go.Layout(
                        title='Real Network Status',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600, width=800
                    ))
    fig.update_layout(showlegend=False)
    return fig


def run_dashboard():
    app.run_server(debug=True)

# If this script is run directly, start the dashboard
if __name__ == '__main__':
    run_dashboard()
