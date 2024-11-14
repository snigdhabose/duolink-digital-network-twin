# src/rl_agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions=4, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_states))  # Q-table initialized to zeros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def select_action(self, state, valid_actions):
        """
        Selects an action based on epsilon-greedy policy.
        If a random number is less than the exploration rate, choose a random action.
        Otherwise, choose the action with the highest Q-value from the Q-table.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(valid_actions)
            # print(f"Exploration: Random action {action} chosen.")
        else:
            q_values = [self.q_table[state, a] for a in valid_actions]
            action = valid_actions[np.argmax(q_values)]
            # print(f"Exploitation: Action {action} with max Q-value chosen.")

        return action

    def update_q_value(self, state, action, reward, next_state, valid_actions):
        """
        Updates the Q-value for the given state-action pair using the Q-learning formula.
        """
        old_value = self.q_table[state, action]
        next_max = max([self.q_table[next_state, a] for a in valid_actions]) if valid_actions else 0
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value
        # print(f"Updated Q-value for state {state}, action {action} from {old_value} to {new_value}")

    def decay_exploration(self):
        """
        Decays the exploration rate after each episode to gradually shift from exploration to exploitation.
        """
        self.exploration_rate *= self.exploration_decay
        # print(f"Exploration rate decayed to {self.exploration_rate}")
