#contains the PyTorch NN, forward pass, action selection + epsilon schedule.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from rl.utils import sizes_from_game
from ClueBasics.GameRules import GameRules
import random

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
    def __init__(self, game_rules, input_dim, lr=1e-4, gamma=0.99, device=None):
        self.game_rules = game_rules
        self.input_dim = input_dim
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- FIXED: Calculate the output dimension correctly ---
        S, W, R, C, _ = sizes_from_game(self.game_rules)
        # Only suggestions + accusations for regular play
        # (Reveals are handled separately and not part of regular action selection)
        self.output_dim = (S * W * R) * 2  # suggestions + accusations only
        print(f"DQNAgent initialized with output_dim: {self.output_dim}")
        print(f"Game dimensions - S:{S}, W:{W}, R:{R}, C:{C}")
        
        self.q_net = DQN(input_dim, self.output_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval() # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss() # Huber loss
    
    def select_action(self, obs, valid_mask, epsilon):
        """
        Selects an action using an epsilon-greedy policy.
        A valid mask is applied to prevent selecting illegal moves.
        """
        print(f"Agent received epsilon: {epsilon}")
        print(f"Q-network output size: {self.output_dim}")
        print(f"Valid mask size: {len(valid_mask)}")
        
        # Ensure mask size matches network output
        if len(valid_mask) != self.output_dim:
            print(f"WARNING: Mask size {len(valid_mask)} doesn't match output dim {self.output_dim}")
            # Truncate or pad the mask to match
            if len(valid_mask) > self.output_dim:
                valid_mask = valid_mask[:self.output_dim]
            else:
                # Pad with False (invalid actions)
                padded_mask = np.zeros(self.output_dim, dtype=bool)
                padded_mask[:len(valid_mask)] = valid_mask
                valid_mask = padded_mask
            print(f"Adjusted mask size to: {len(valid_mask)}")
        
        if random.random() < epsilon:
            # Exploration: pick a random valid action
            valid_action_indices = np.where(valid_mask)[0]
            if len(valid_action_indices) > 0:
                return np.random.choice(valid_action_indices)
            else:
                # Fallback if no valid actions (should not happen in a well-defined game)
                print("WARNING: No valid actions available!")
                return -1
        else:
            # Exploitation: pick the best action according to Q-network
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(obs_t).cpu().numpy().squeeze()

            # Ensure q_values size matches mask
            if len(q_values) != len(valid_mask):
                print(f"WARNING: Q-values size {len(q_values)} doesn't match mask size {len(valid_mask)}")
                return -1
            
            # Mask invalid actions by setting them to -inf
            q_values[~valid_mask] = -np.inf
            
            # The agent may not have enough information to choose an action
            if np.all(q_values == -np.inf):
                print("WARNING: All Q-values are -inf after masking!")
                return -1 # Fallback
                
            return int(np.argmax(q_values))

    def update_target_network(self):
        """Copies the Q-network weights to the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self, batch):
        """Performs a single learning step on the Q-network."""
        # Convert batch to tensors
        obs = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        masks = torch.tensor(np.array(batch.mask), dtype=torch.bool, device=self.device)
        
        # Ensure masks match network output dimension
        if masks.shape[1] != self.output_dim:
            print(f"WARNING: Mask dimension {masks.shape[1]} doesn't match output dim {self.output_dim}")
            if masks.shape[1] > self.output_dim:
                masks = masks[:, :self.output_dim]
            else:
                # Pad with False
                padded_masks = torch.zeros(masks.shape[0], self.output_dim, dtype=torch.bool, device=self.device)
                padded_masks[:, :masks.shape[1]] = masks
                masks = padded_masks
        
        # Rest of learning logic...
        q_values = self.q_net(obs)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            next_q_values[~masks] = -torch.inf
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + self.gamma * max_next_q_values * (1 - dones.float())

        loss = self.loss_fn(state_action_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()