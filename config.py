# config.py
DATA_PATH = 'data/'
TRAIN_FILE = DATA_PATH + 'KDDTrain+.txt'
TEST_FILE = DATA_PATH + 'KDDTest+.txt'
MODEL_PATH = 'models/saved_model.pth'
RESULTS_PATH = 'results/visualization.png'

# RL Hyperparameters
LEARNING_RATE = 0.001
EPISODES = 100
STATE_DIM = 37  # Adjusted to match the actual number of features in your dataset

ACTION_DIM = 2  # Actions: reroute or no action
