# src/environment.py
import gym
from gym import spaces
import networkx as nx
import numpy as np

class NetworkRoutingEnv(gym.Env):
    def __init__(self, graph, source, target):
        super(NetworkRoutingEnv, self).__init__()
        self.graph = graph
        self.source = source
        self.target = target
        self.current_node = source

        # Define observation space based on the number of nodes
        self.observation_space = spaces.Discrete(len(graph.nodes))
        self.action_space = spaces.Discrete(len(graph.nodes))

    def reset(self):
        self.current_node = self.source
        return self.current_node

    def step(self, action):
        action = int(action)
        if self.graph.has_edge(self.current_node, action):
            reward = -self.graph[self.current_node][action].get('weight', 1)
            self.current_node = action
            done = self.current_node == self.target
        else:
            reward = -10  # High penalty for invalid action
            done = False
        return self.current_node, reward, done, {}

    def get_neighbors(self):
        return list(self.graph.neighbors(self.current_node))
