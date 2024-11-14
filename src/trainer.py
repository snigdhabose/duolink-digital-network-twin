import joblib
import xgboost as xgb
from src.data_loader import load_data
from src.environment import NetworkRoutingEnv
from src.rl_agent import QLearningAgent
import networkx as nx
import pickle
import numpy as np
# src/trainer.py
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from xgboost import DMatrix, train as xgb_train
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from src.data_loader import load_data

def train_failure_predictor():
    print("Loading and preparing data for failure prediction...")
    X_train, X_test, y_train, y_test = load_data('data/KDDTrain+.txt')
    print("Data loaded successfully.")
    
    # Calculate scale_pos_weight based on class imbalance
    counts = np.bincount(y_train)
    scale_pos_weight = counts[0] / counts[1]
    print(f"Scale_pos_weight set to: {scale_pos_weight}")
    
    # Define the XGBoost model
    print("Training XGBoost model with regularization and class balancing...")
    model = XGBClassifier(
        max_depth=4,
        reg_alpha=10,
        reg_lambda=10,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    # Train the model with early stopping
    model.fit(
        X_train, y_train,
        early_stopping_rounds=10,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    print("Model training completed.")
    
    # Save model
    joblib.dump(model, 'models/xgboost_failure_model.pkl')
    print("Failure prediction model saved as 'xgboost_failure_model.pkl'.")
    
    print("Model training completed.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {roc_auc}")

    # Save model
    joblib.dump(model, 'models/xgboost_failure_model.pkl')
    print("Failure prediction model saved as 'xgboost_failure_model.pkl'.")

    return model



def train_rl_agent(graph, source, target, episodes=10000):
    print(f"Initializing the network routing environment (source: {source}, target: {target})...")
    env = NetworkRoutingEnv(graph, source, target)
    rl_agent = QLearningAgent(n_states=len(graph.nodes))

    print(f"Starting training for {episodes} episode(s).")
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            neighbors = env.get_neighbors()
            if not neighbors:
                break

            action = rl_agent.select_action(state, neighbors)
            next_state, reward, done, _ = env.step(action)
            rl_agent.update_q_value(state, action, reward, next_state, env.get_neighbors())
            episode_reward += reward
            state = next_state

        print(f"Episode {episode + 1} finished with Total Reward: {episode_reward}")

    # Save the trained RL agent
    joblib.dump(rl_agent, 'models/rl_agent.pkl')
    print("RL agent training completed and saved as 'rl_agent.pkl'.")

    return rl_agent
