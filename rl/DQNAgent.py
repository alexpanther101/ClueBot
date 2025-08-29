#contains the PyTorch NN, forward pass, action selection + epsilon schedule.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super(DQN, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # Q-values for all actions
    
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-4, gamma=0.99, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.target_net.eval()  # Target net in eval mode

    def select_action(self, obs, valid_mask, epsilon):
        """
        obs: np.array (input_dim,)
        valid_mask: np.array (output_dim,) of bools
        epsilon: float, exploration rate
        """
        if np.random.rand() < epsilon:
            # Random valid action
            valid_indices = np.nonzero(valid_mask)[0]
            return np.random.choice(valid_indices)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t).cpu().numpy().squeeze()

        # Mask invalid actions by setting them to -inf
        q_values[~valid_mask] = -np.inf
        return int(np.argmax(q_values))

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self, batch):
        """
        batch: dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        Each value is a tensor already on the correct device.
        """
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        # Q(s,a)
        q_values = self.q_net(obs)
        state_action_values = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            max_next_q_values, _ = next_q_values.max(dim=1)
            target_values = rewards + self.gamma * (1 - dones) * max_next_q_values

        loss = self.loss_fn(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # stability
        self.optimizer.step()

        return loss.item()