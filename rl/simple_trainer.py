#!/usr/bin/env python3
"""
Simple trainer that saves the trained model
"""
import sys
import os
import torch
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClueBasics.GameRules import GameRules
from agents import EliminationBot, TriggerHappyBot
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.replay import ReplayBuffer
from rl.utils import build_observation, build_action_mask, calculate_observation_size

class SimpleTrainer:
    def __init__(self, game_rules, rl_player, dqn_agent, replay_buffer, save_path="models/simple_trained_agent.pth"):
        self.game_rules = game_rules
        self.rl_player = rl_player
        self.agent = dqn_agent
        self.replay_buffer = replay_buffer
        self.epsilon = 0.5
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01  # Lower minimum for more exploitation
        self.total_steps = 0
        self.save_path = save_path
        
        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Connect components
        self.rl_player.agent = self.agent
        self.rl_player.trainer = self
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.total_episodes = 0
        
        # Track revealed cards across episodes
        self.revealed_cards = {}
        
        # Store original game state for consistent dimensions
        self.original_players = self.game_rules.players.copy() if self.game_rules.players else []
        
    def train_episode(self):
        """Train a single episode"""
        # Create fresh opponent for this episode
        opponent = TriggerHappyBot(f"Opponent_{self.total_episodes}", self.game_rules, "Trigger")
        players = [self.rl_player, opponent]
        
        # Reset player states
        self.rl_player.inGame = True
        opponent.inGame = True
        
        # Setup game
        self.game_rules.players = players
        for p in players:
            p.setOpponents([op for op in players if op != p])
            
        self.game_rules.reset_game()
        self.game_rules.dealCards()
        
        # Initialize belief matrices
        for player in players:
            player.initialCrossOff()
        
        episode_reward = 0
        turn_count = 0
        max_turns = 100  # Increased from 50
        winner = None
        
        # Track whose turn it is
        current_player_idx = 0
        
        while turn_count < max_turns and winner is None:
            # Check if both players are still in game
            active_players = [p for p in players if p.inGame]
            if len(active_players) < 2:
                break
                
            # Get current player
            current_player = players[current_player_idx]
            
            # Skip eliminated players
            if not current_player.inGame:
                current_player_idx = (current_player_idx + 1) % len(players)
                continue
                
            try:
                if current_player.type == "RL":
                    self.total_steps += 1
                    
                    # Get observation and action mask
                    obs = build_observation(self.game_rules, current_player)
                    mask = build_action_mask(self.game_rules, current_player)
                    
                    # Store previous observation for transition
                    prev_obs = obs.copy()
                    
                    # Player takes turn (returns winner name or False)
                    result = current_player.playTurn(obs, mask)
                    
                    # Check if game ended
                    if result:
                        winner = result
                        # Get final reward
                        reward = self.game_rules.get_last_reward()
                        episode_reward += reward
                        
                        # Store final transition
                        next_obs = build_observation(self.game_rules, current_player)
                        if current_player.last_obs is not None and current_player.last_action is not None:
                            current_player.store_transition(reward, next_obs, done=True)
                    else:
                        # Get step reward
                        reward = self.game_rules.get_last_reward()
                        episode_reward += reward
                        
                        # Get next observation and store transition
                        next_obs = build_observation(self.game_rules, current_player)
                        if current_player.last_obs is not None and current_player.last_action is not None:
                            current_player.store_transition(reward, next_obs, done=False)
                    
                    # Train if we have enough experience
                    if len(self.replay_buffer) >= 64 and self.total_steps % 4 == 0:
                        batch = self.replay_buffer.sample(32)
                        self.agent.learn(batch)
                        
                else:
                    # Opponent turn
                    # For EliminationBot, just call playTurn with dummy parameters
                    result = current_player.playTurn(None, None)
                    if result:
                        winner = result
                        # Negative reward for RL agent if opponent wins
                        if current_player == opponent:
                            episode_reward -= 100
                            # Store final transition for RL player
                            if self.rl_player.last_obs is not None:
                                next_obs = build_observation(self.game_rules, self.rl_player)
                                self.rl_player.store_transition(-100, next_obs, done=True)
                        
            except Exception as e:
                print(f"Error during turn {turn_count}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Move to next player
            current_player_idx = (current_player_idx + 1) % len(players)
            turn_count += 1
            
            # Increment game turn counter
            self.game_rules.turn += 1
            self.game_rules.gameTurn += 1
            
            # Early termination if RL player is eliminated
            if not self.rl_player.inGame:
                # Store final negative reward
                if self.rl_player.last_obs is not None:
                    next_obs = build_observation(self.game_rules, self.rl_player)
                    self.rl_player.store_transition(-50, next_obs, done=True)
                episode_reward -= 50
                break
        
        # Update target network occasionally
        if self.total_episodes > 0 and self.total_episodes % 10 == 0:
            self.agent.update_target_network()
        
        # Record episode results
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(turn_count)
        
        if winner == self.rl_player.name:
            self.win_count += 1
        
        self.total_episodes += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return episode_reward, winner, turn_count
    def train(self, num_episodes=200, log_interval=10, save_interval=25):
        """Train for specified number of episodes"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Saving model to: {self.save_path}")
        
        for episode in range(num_episodes):
            episode_reward, winner, turn_count = self.train_episode()
            
            # Logging
            if (episode + 1) % log_interval == 0:
                recent_rewards = self.episode_rewards[-log_interval:]
                recent_lengths = self.episode_lengths[-log_interval:]
                win_rate = self.win_count / self.total_episodes
                
                print(f"Episode {episode + 1}/{num_episodes}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(recent_lengths):.1f} turns")
                print(f"  Win Rate: {self.win_count}/{self.total_episodes} ({win_rate:.1%})")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Steps: {self.total_steps}")
                
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                self.save_model(f"checkpoint_ep_{episode + 1}")
        
        # Final save
        self.save_model("final_model")
        
        # Print final results
        final_win_rate = self.win_count / self.total_episodes
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        print(f"\nTraining completed!")
        print(f"Final Results:")
        print(f"  Episodes: {self.total_episodes}")
        print(f"  Win Rate: {self.win_count}/{self.total_episodes} ({final_win_rate:.1%})")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Episode Length: {avg_length:.1f} turns")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Final Epsilon: {self.epsilon:.3f}")
        print(f"  Model saved to: {self.save_path}")
        
        return {
            'win_rate': final_win_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
    
    def save_model(self, checkpoint_name=None):
        """Save the trained model"""
        try:
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if checkpoint_name:
                save_path = self.save_path.replace('.pth', f'_{checkpoint_name}_{timestamp}.pth')
            else:
                save_path = self.save_path.replace('.pth', f'_{timestamp}.pth')
            
            checkpoint = {
                'model_state_dict': self.agent.q_net.state_dict(),
                'target_model_state_dict': self.agent.target_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps,
                'win_count': self.win_count,
                'episode_rewards': self.episode_rewards[-100:],  # Keep last 100
                'episode_lengths': self.episode_lengths[-100:],  # Keep last 100
                'training_config': {
                    'input_dim': self.agent.input_dim,
                    'output_dim': self.agent.output_dim,
                    'lr': 1e-4,  # Default from DQNAgent
                    'gamma': 0.99,
                }
            }
            
            torch.save(checkpoint, save_path)
            
            # Also save to the main path (most recent)
            torch.save(checkpoint, self.save_path)
            
            print(f"Model saved to: {save_path}")
            
        except Exception as e:
            print(f"Failed to save model: {e}")

def main():
    """Example usage of SimpleTrainer"""
    print("=== Simple Trainer with Model Saving ===")
    
    # Setup
    game = GameRules([])
    
    # Create players FIRST to get correct dimensions
    rl_player = RLPlayer("TrainedBot", game, "RL")
    dummy_opponent = TriggerHappyBot("DummyOpp", game, "TriggerHappy")
    players = [rl_player, dummy_opponent]
    
    # Set players so P=2 for dimension calculation
    game.players = players
    for p in players:
        p.setOpponents([op for op in players if op != p])
    
    # NOW calculate input dimension with correct P value
    input_dim = calculate_observation_size(game)
    print(f"Calculated input dimension: {input_dim}")
    
    # Create RL components with correct input dimension
    agent = DQNAgent(game_rules=game, input_dim=input_dim)
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Create trainer
    trainer = SimpleTrainer(
        game_rules=game,
        rl_player=rl_player,
        dqn_agent=agent,
        replay_buffer=replay_buffer,
        save_path="models/simple_trained_clue_agent.pth"
    )
    
    # Train the agent
    results = trainer.train(num_episodes=500, log_interval=5, save_interval=10)
    
    print(f"\nTraining completed with results: {results}")

if __name__ == "__main__":
    main()