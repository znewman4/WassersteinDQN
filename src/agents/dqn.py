# src/agents/dqn.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from src.utils.logging import get_logger
from src.core.registry import AGENTS


class QNetwork(nn.Module):
    """Simple feedforward network for Q-value approximation."""
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


@AGENTS.register("dqn")
class DQNAgent:
    """
    Deep Q-Network Agent.
    Compatible with TradingEnv (expects obs as np.ndarray).
    """
    def __init__(self, obs_dim, action_dim, cfg):
        self.logger = get_logger("DQNAgent")
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Core hyperparameters
        self.gamma = cfg.get("gamma", 0.99)
        self.epsilon = cfg.get("epsilon_start", 1.0)
        self.epsilon_end = cfg.get("epsilon_end", 0.05)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        self.batch_size = cfg.get("batch_size", 64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_sizes = cfg.get("network", {}).get("hidden_sizes", [128, 128])
        self.q_net = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        lr_value = float(cfg.get("lr", 1e-4))
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_value)
        self.replay = ReplayBuffer(cfg.get("replay_size", 50_000))
        self.target_update_interval = cfg.get("target_update_interval", 1000)
        self.step_count = 0

    def act(self, state):
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def push(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_values = self.q_net(s).gather(1, a)
        with torch.no_grad():
            max_next_q = self.target_net(s2).max(1)[0].unsqueeze(1)
            target_q = r + (1 - d) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "q_net" in state:
            self.q_net.load_state_dict(state["q_net"])
            self.target_net.load_state_dict(state["target_net"])
            self.epsilon = state.get("epsilon", 0.0)
        else:
            self.q_net.load_state_dict(state)
            self.target_net.load_state_dict(self.q_net.state_dict())
