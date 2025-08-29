#!/usr/bin/env python3
"""
Main training script for Clue RL Agent
"""

import torch
import numpy as np
from ClueBasics.GameRules import GameRules
from ClueBasics.Player import Player
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.replay import ReplayBuffer
from rl.trainer import Trainer
from rl.utils import action_space_size, build_observation
from agents import TriggerHappyBot, EliminationBot, BayesianLearner

def create_game_and_players():
    """Create game instance with RL player and bot opponents"""
    
    # Create players (1 RL agent + 3 simple bots)
    players = []
    
    # Temporary game instance for player creation
    temp_game = GameRules([])
    
    # Create RL player
    rl_player = RLPlayer("RL_Agent", temp_game)
    players.append(rl_player)
    
    # Create bot opponents  
    trigbot = TriggerHappyBot("TrigBot", temp_game, "Bot")
    players.append(trigbot)
    
    elimbot = EliminationBot("ElimBot", temp_game, "Bot")
    players.append(elimbot)
    
    bayesianbot = BayesianLearner("BayesBot", temp_game, "Bot")
    players.append(bayesianbot)
    # Create actual game with all players
    game = GameRules(players)
    
    # Update player game references
    for player in players:
        player.game = game
        
    # Set up opponents for each player
    for player in players:
        opponents = [p for p in players if p != player]
        player.setOpponents(opponents)
        
    return game, rl_player, players

def main():
    """Main training function"""
    
    # Hyperparameters
    EPISODES = 5000
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 100
    BUFFER_SIZE = 100000
    N_STEP = 3
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create game and players
    game, rl_player, players = create_game_and_players()
    
    # Calculate observation and action space sizes
    sample_obs = build_observation(game, rl_player)
    obs_dim = len(sample_obs)
    action_dim = action_space_size(game)
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create DQN agent
    agent = DQNAgent(
        input_dim=obs_dim,
        output_dim=action_dim,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=BUFFER_SIZE,
        n_step=N_STEP,
        gamma=GAMMA
    )
    
    # Create trainer
    trainer = Trainer(
        game_rules=game,
        rl_player=rl_player,
        dqn_agent=agent,
        replay_buffer=replay_buffer,
        n_step=N_STEP,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        max_episodes=EPISODES,
        target_update=TARGET_UPDATE,
        learning_starts=1000,
        train_freq=4,
        checkpoint_path="checkpoints/clue_dqn.pth",
        log_interval=50,
        eval_episodes=10,
        device=device
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")

if __name__ == "__main__":
    main()