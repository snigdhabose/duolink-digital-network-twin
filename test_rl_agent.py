import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.rl_agent import DQNAgent

def test_minimal_rl_agent():
    device = torch.device("cpu")
    agent = DQNAgent(state_size=37, action_size=2).to(device)
    optimizer = optim.Adam(agent.parameters())
    criterion = nn.MSELoss()

    state = torch.rand(1, 37)
    target = torch.rand(1, 2)

    output = agent(state)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), 'models/saved_model.pth')
    print("Minimal RL agent training completed and model saved successfully.")

if __name__ == "__main__":
    test_minimal_rl_agent()
