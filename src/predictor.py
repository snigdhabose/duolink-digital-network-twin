import joblib
import pickle
from src.rl_agent import QLearningAgent
from src.data_loader import load_data
import networkx as nx
import numpy as np
from xgboost import DMatrix

# src/predictor.py

import numpy as np

# Load the trained XGBoost model for failure prediction
try:
    model = joblib.load('models/xgboost_failure_model.pkl')
    print("Failure prediction model loaded successfully.")
except FileNotFoundError:
    print("Error: Failure prediction model file not found. Ensure it is trained and saved.")
    model = None  # Set to None if not found

def predict_failures(node_features, threshold=0.5):
    global model
    if model is None:
        print("Failure prediction model is not loaded.")
        return []

    # node_features should be a dictionary {node_id: feature_vector}
    node_ids = list(node_features.keys())
    features = np.array(list(node_features.values()))

    # Get the probabilities for each node
    probabilities = model.predict_proba(features)[:, 1]

    # Map probabilities to node IDs
    node_probabilities = dict(zip(node_ids, probabilities))

    # Apply the threshold to determine failed nodes
    failed_nodes = [node_id for node_id, prob in node_probabilities.items() if prob >= threshold]

    # Debug: Print probability statistics
    print(f"Node Probabilities: {node_probabilities}")

    print(f"Predicted failed nodes: {failed_nodes}")
    return failed_nodes


# Load the trained RL agent
try:
    rl_agent = joblib.load('models/rl_agent.pkl')
    print("RL agent loaded successfully.")
except FileNotFoundError:
    print("Error: RL agent model file not found. Ensure it is trained and saved.")
    rl_agent = None  # Set to None if not found

def suggest_reroutes(graph, failed_nodes, rl_agent):
    if rl_agent is None:
        print("RL agent is not loaded.")
        return []

    reroute_paths = []
    for node in failed_nodes:
        if node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                action = rl_agent.select_action(node, neighbors)
                if action in graph.nodes:
                    try:
                        alternative_path = nx.shortest_path(graph, source=neighbor, target=action)
                        reroute_paths.extend([(alternative_path[i], alternative_path[i + 1]) for i in range(len(alternative_path) - 1)])
                    except nx.NetworkXNoPath:
                        continue
    return reroute_paths
