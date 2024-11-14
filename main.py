import os
import argparse
from src.visualizer import run_dashboard
from src.sample_visualizer import run_sample_dashboard
from src.trainer import train_failure_predictor, train_rl_agent
import networkx as nx

def main(retrain=False):
    # Check if models exist or retraining is requested
    if retrain or not os.path.exists('models/xgboost_failure_model.pkl'):
        print("Failure prediction model not found or retraining requested. Training model...")
        train_failure_predictor()
        print("Failure prediction model training completed.")
    else:
        print("Failure prediction model found. Skipping training.")

    if retrain or not os.path.exists('models/rl_agent.pkl'):
        print("RL agent model not found or retraining requested. Training model...")
        graph = nx.erdos_renyi_graph(15, 0.3)  # Example graph
        train_rl_agent(graph, source=0, target=14, episodes=10000)
        print("RL agent training completed.")
    else:
        print("RL agent model found. Skipping training.")

    # print("Starting dashboard...")
    # run_dashboard()
    print("Running the sample network visualization...")
    run_sample_dashboard()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Digital Twin Network")
    parser.add_argument('--retrain', action='store_true', help="Retrain models before running the dashboard")
    args = parser.parse_args()

    # Run main function with retrain flag
    main(retrain=args.retrain)
