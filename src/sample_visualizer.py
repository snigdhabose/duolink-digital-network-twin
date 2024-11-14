# sample_visualizer.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import networkx as nx
from collections import deque

# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
        "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
    ]
)

# Create the graph
device_count = 15
# Create a fixed graph structure
G = nx.Graph()
# Define fixed edges to ensure the same graph structure every time
G.add_edges_from([
    (0, 3), (3, 10), (10, 7), (7, 4), (5, 3), (11, 13), (13, 5), (5, 12), (12, 7), (7, 9),
    (5, 12), (12, 3), (3, 2), (3, 6), (3, 8), (3, 9), (3, 0),(11,2),(2,10),(0,2)
])

# Use a fixed layout for consistent node positioning
pos = {
    0: (-1, 1), 3: (0, 1), 10: (1, 1), 7: (0.5, 0),
    4: (1, -1), 11: (-1, -1), 13: (-0.5, -0.5), 5: (0, -1),
    12: (0.5, -1), 9: (1, -0.5), 2: (-0.5, 1.5), 6: (0.5, 1.5),
    8: (1.5, 0.5)
}


nx.set_node_attributes(G, 'active', 'status')

# Scenarios as per your instruction
scenarios = deque([
    # Scenario 1
    {
        "type": "normal",
        "packet": {"packet_id": 1, "source": 0, "destination": 10},
        "expected_path": [0, 3, 10],
        "message": "",
    },
    # Scenario 2
    {
        "type": "normal",
        "packet": {"packet_id": 2, "source": 10, "destination": 4},
        "expected_path": [10, 7, 4],
        "message": "",
    },
    # Scenario 3
    {
        "type": "normal",
        "packet": {"packet_id": 3, "source": 11, "destination": 5},
        "expected_path": [11, 13, 5],
        "message": "",
    },
    # Scenario 4 Part 1
    {
        "type": "predict_failure",
        "packet": {"packet_id": 3, "source": 5, "destination": 9},
        "failed_node": 3,
        "message": "Model predicts Device 3 is going to fail.",
        "part": 1,
    },
    # Scenario 4 Part 2
    {
        "type": "confirm_failure",
        "packet": {"packet_id": 3, "source": 5, "destination": 9},
        "failed_node": 3,
        "reroute_path": [5, 12, 7, 9],
        "message": "Prediction confirmed: Device 3 has failed. Packets rerouted successfully.",
        "part": 2,
    },
        # Scenario 3
    {
        "type": "normal",
        "packet": {"packet_id": 4, "source": 9, "destination": 11},
        "expected_path": [9, 7, 12, 5, 13, 11],
        "message": "",
    },
        {
        "type": "normal",
        "packet": {"packet_id": 5, "source": 11, "destination": 2},
        "expected_path": [11,2],
        "message": "",
    },
            {
        "type": "normal",
        "packet": {"packet_id": 6, "source": 2, "destination": 10},
        "expected_path": [3,10],
        "message": "",
    },
                {
        "type": "normal",
        "packet": {"packet_id": 6, "source": 10, "destination": 0},
        "expected_path": [10,0],
        "message": "",
    },
    # Scenario 4 Part 1
    {
        "type": "predict_failure",
        "packet": {"packet_id": 7, "source": 0, "destination": 13},
        "failed_node": 11,
        "message": "Model predicts Device 11 is going to fail.",
        "part": 1,
    },
    # Scenario 4 Part 2
    {
        "type": "confirm_failure",
        "packet": {"packet_id": 7, "source": 0, "destination": 13},
        "failed_node": 11,
        "reroute_path": [0, 3, 5, 13],
        "message": "Prediction confirmed: Device 11 has failed. Packets rerouted successfully.",
        "part": 2,
    },
                   {
        "type": "normal",
        "packet": {"packet_id": 8, "source": 13, "destination": 3},
        "expected_path": [13,5,3],
        "message": "",
    },
    
                   {
        "type": "normal",
        "packet": {"packet_id": 9, "source": 3, "destination": 4},
        "expected_path": [3,9,7,4],
        "message": "",
    },
])

def create_network_data(G, failed_nodes, failed_nodes_with_cross, packet_edges, reroute_edges, failed_edges):
    edge_trace = []
    reroute_edge_trace = []
    failed_edge_trace = []
    packet_flow_trace = []

    # Update node statuses
    for node in G.nodes():
        G.nodes[node]['status'] = 'failed' if node in failed_nodes else 'active'

    for edge in G.edges():
        if edge in reroute_edges or (edge[1], edge[0]) in reroute_edges:
            reroute_edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='green'),
                mode='lines',
                showlegend=False
            ))
        elif edge in failed_edges or (edge[1], edge[0]) in failed_edges:
            failed_edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='red'),
                mode='lines',
                showlegend=False
            ))
        elif edge in packet_edges or (edge[1], edge[0]) in packet_edges:
            packet_flow_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=2, color='blue'),
                mode='lines',
                showlegend=False
            ))
        else:
            edge_trace.append(go.Scatter(
                x=[pos[edge[0]][0], pos[edge[1]][0]],
                y=[pos[edge[0]][1], pos[edge[1]][1]],
                line=dict(width=1, color='gray'),
                mode='lines',
                showlegend=False
            ))

    node_status = [G.nodes[node]['status'] for node in G.nodes()]
    node_colors = ['red' if status == 'failed' else 'green' for status in node_status]
    node_symbols = ['x' if node in failed_nodes_with_cross else 'circle' for node in G.nodes()]
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[f"Device {node}" for node in G.nodes()],
        marker=dict(
            color=node_colors,
            symbol=node_symbols,
            size=20,
            line=dict(width=2, color='black')
        ),
        textposition='top center',
        showlegend=False
    )
    return edge_trace, reroute_edge_trace, failed_edge_trace, packet_flow_trace, node_trace

# Layout
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
    html.Div(id="alert-box", style={
        'text-align': 'center',
        'font-size': '18px',
        'margin-top': '10px',
        'padding': '10px',
        'border-radius': '8px',
        'color': '#ffffff',
        'display': 'none'
    }),
    html.Div(id="packet-info", style={
        'text-align': 'center',
        'font-size': '16px',
        'margin-top': '10px',
        'color': '#4A4A4A'
    }),
    html.Div([
        dcc.Graph(id='predicted-graph', config={'displayModeBar': True, 'scrollZoom': True}, style={'height': '500px'}),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='real-graph', config={'displayModeBar': True, 'scrollZoom': True}, style={'height': '500px'}),
    ], style={'width': '48%', 'display': 'inline-block'}),
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
        'padding': '10px',
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


# Callback to update graphs and display alerts
@app.callback(
    [Output('predicted-graph', 'figure'),
     Output('real-graph', 'figure'),
     Output('alert-box', 'children'),
     Output('alert-box', 'style'),
     Output('packet-info', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_network(n_intervals):
    # Set default values
    alert_message = ""
    alert_style = {'display': 'none'}
    packet_info = ""
    failed_nodes_graph1 = set()
    failed_nodes_graph2 = set()
    failed_nodes_with_cross_graph2 = set()
    reroute_edges = []
    failed_edges_graph1 = []
    failed_edges_graph2 = []
    packet_edges_graph1 = []
    packet_edges_graph2 = []
    G_predicted = G.copy()
    G_real = G.copy()

    # Retrieve the scenario for the current interval
    if scenarios:
        scenario = scenarios.popleft()
    else:
        # No more scenarios, stop updating
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Packet information
    packet = scenario.get('packet')
    if packet:
        packet_info = f"Packet {packet['packet_id']}: Source={packet['source']}, Destination={packet['destination']}"
    else:
        packet_info = "No packet in this interval."

    # Handle scenarios
    if scenario["type"] == "normal":
        # Normal operation
        # Calculate packet path
        if packet:
            path = scenario['expected_path']
            packet_edges_graph1 = list(zip(path[:-1], path[1:]))
            packet_edges_graph2 = packet_edges_graph1.copy()
    elif scenario["type"] == "predict_failure":
        # Predicted failure in Graph 1
        failed_node = scenario['failed_node']
        failed_nodes_graph1.add(failed_node)
        alert_message = scenario['message']
        alert_style = {'display': 'block', 'background-color': '#ffcc00'}
        # Mark all outgoing edges from failed node as failed in Graph 1
        failed_edges_graph1 = list(G_predicted.edges(failed_node))
        # Packet path in Graph 1 avoids the failed node
        if packet:
            # No blue lines in Graph 1 during predicted failure
            packet_edges_graph1 = []
            # Show alternate route in green
            reroute_path = [5, 12, 7, 9]
            reroute_edges = list(zip(reroute_path[:-1], reroute_path[1:]))
        # In Graph 2, all nodes are green, no blue lines
        packet_edges_graph2 = []
        failed_nodes_graph2 = set()
    elif scenario["type"] == "confirm_failure":
        # Confirmed failure in Graph 2
        failed_node = scenario['failed_node']
        failed_nodes_graph1.add(failed_node)  # Node remains failed in Graph 1
        failed_nodes_graph2.add(failed_node)
        failed_nodes_with_cross_graph2.add(failed_node)
        alert_message = scenario['message']
        alert_style = {'display': 'block', 'background-color': '#d9534f'}
        # Mark all outgoing edges from failed node as failed in Graph 2
        failed_edges_graph2 = list(G_real.edges(failed_node))
        # In Graph 1, packet uses alternate route (now with blue lines)
        if packet:
            reroute_path = scenario['reroute_path']
            packet_edges_graph1 = list(zip(reroute_path[:-1], reroute_path[1:]))
            reroute_edges = list(zip(reroute_path[:-1], reroute_path[1:]))
        # In Graph 2, packet attempts to go through the failed node
        path_real = [packet['source'], 12, failed_node]
        packet_edges_graph2 = list(zip(path_real[:-1], path_real[1:]))
    else:
        # Default to normal operation
        if packet:
            path = scenario['expected_path']
            packet_edges_graph1 = list(zip(path[:-1], path[1:]))
            packet_edges_graph2 = packet_edges_graph1.copy()

    # Create figures
    edge_trace1, reroute_edge_trace1, failed_edge_trace1, packet_flow_trace1, node_trace1 = create_network_data(
        G_predicted, failed_nodes_graph1, set(), packet_edges_graph1, reroute_edges, failed_edges_graph1
    )
    fig_predicted = go.Figure(
        data=edge_trace1 + reroute_edge_trace1 + failed_edge_trace1 + packet_flow_trace1 + [node_trace1],
        layout=go.Layout(
            title='Predicted Network Status',
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            height=600, width=800
        )
    )
    fig_predicted.update_layout(showlegend=False)

    edge_trace2, reroute_edge_trace2, failed_edge_trace2, packet_flow_trace2, node_trace2 = create_network_data(
        G_real, failed_nodes_graph2, failed_nodes_with_cross_graph2, packet_edges_graph2, [], failed_edges_graph2
    )
    fig_real = go.Figure(
        data=edge_trace2 + reroute_edge_trace2 + failed_edge_trace2 + packet_flow_trace2 + [node_trace2],
        layout=go.Layout(
            title='Real Network Status',
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
            height=600, width=800,
            annotations=[
                # Add a cross at the failed node in Graph 2
                go.layout.Annotation(
                    x=pos[failed_node][0],
                    y=pos[failed_node][1],
                    text='X',
                    showarrow=False,
                    font=dict(color='black', size=24)
                )
            ] if failed_nodes_with_cross_graph2 else []
        )
    )
    fig_real.update_layout(showlegend=False)

    return fig_predicted, fig_real, alert_message, alert_style, packet_info

def run_sample_dashboard():
    app.run_server(debug=True)

if __name__ == '__main__':
    run_sample_dashboard()
