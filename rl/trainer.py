import numpy as np
import torch
from collections import deque
import logging
import os
import random
from typing import Dict, List, Optional, Tuple
from rl.utils import build_observation, build_action_mask

class CurriculumTrainer:
    def __init__(
        self,
        game_rules,
        rl_player,
        dqn_agent,
        replay_buffer,
        n_step=3,
        batch_size=64,
        gamma=0.99,
        target_update=100,
        learning_starts=100,
        train_freq=1,
        checkpoint_path=None,
        log_interval=50,
        device=None
    ):
        self.game_rules = game_rules
        self.rl_player = rl_player
        self.agent = dqn_agent
        self.replay_buffer = replay_buffer
        
        # Training parameters (kept from your original)
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        
        # Logging and checkpointing
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Curriculum-specific attributes
        self.current_phase = 0
        self.phase_episode_count = 0
        self.phase_win_rates = deque(maxlen=100)
        self.total_steps = 0
        self.total_episodes = 0
        
        # Phase definitions
        self.phases = [
            {
                'name': 'Learning Basics',
                'episodes': 1000,
                'opponents': ['EliminationBot'],
                'num_players': 3,
                'win_threshold': 0.4,
                'epsilon_start': 1.0,
                'epsilon_end': 0.4
            },
            {
                'name': 'Strategic Play',
                'episodes': 1500,
                'opponents': ['EliminationBot', 'EliminationBot'],
                'num_players': 3,
                'win_threshold': 0.4,
                'epsilon_start': 0.4,
                'epsilon_end': 0.2
            },
            {
                'name': 'Competitive Play',
                'episodes': 2000,
                'opponents': ['EliminationBot', 'TriggerHappyBot'],
                'num_players': 4,
                'win_threshold': 0.6,
                'epsilon_start': 0.2,
                'epsilon_end': 0.1
            },
            {
                'name': 'Expert Challenge',
                'episodes': 2500,
                'opponents': ['EliminationBot', 'BayesianLearner'],
                'num_players': 4,
                'win_threshold': 0.35,
                'epsilon_start': 0.1,
                'epsilon_end': 0.05
            }
        ]
        
        # Initialize first phase
        self.setup_phase(0)
        
        # Connect RL player to this trainer
        self.rl_player.agent = self.agent
        self.rl_player.trainer = self
        
    def setup_phase(self, phase_idx):
        """Configure training for a specific phase"""
        if phase_idx >= len(self.phases):
            self.logger.info("All curriculum phases completed!")
            return False
            
        phase = self.phases[phase_idx]
        self.current_phase = phase_idx
        self.phase_episode_count = 0
        self.phase_win_rates.clear()
        
        self.logger.info(f"\n=== Phase {phase_idx + 1}: {phase['name']} ===")
        self.logger.info(f"Episodes: {phase['episodes']}, Players: {phase['num_players']}")
        self.logger.info(f"Opponents: {phase['opponents']}")
        
        # Set epsilon for this phase
        self.epsilon = phase['epsilon_start']
        self.epsilon_start = phase['epsilon_start']
        self.epsilon_end = phase['epsilon_end']
        self.epsilon_decay_rate = (phase['epsilon_start'] / phase['epsilon_end']) ** (1.0 / phase['episodes'])
        
        self.current_opponents = phase['opponents']
        self.num_players = phase['num_players']
        return True
        
    def create_opponents(self):
        """Create opponent bots for current phase"""
        # Import your bot classes
        from agents import EliminationBot, TriggerHappyBot, BayesianLearner
        
        bot_classes = {
            'EliminationBot': EliminationBot,
            'TriggerHappyBot': TriggerHappyBot, 
            'BayesianLearner': BayesianLearner
        }
        
        opponents = []
        opponents_needed = self.num_players - 1
        
        for i in range(opponents_needed):
            bot_type = random.choice(self.current_opponents)
            bot_class = bot_classes[bot_type]
            opponent = bot_class(f"{bot_type}_{i}", self.game_rules, bot_type)
            opponents.append(opponent)
            
        return opponents
        
    def should_advance_phase(self):
        """Check if ready to advance to next phase"""
        phase = self.phases[self.current_phase]
        
        # Need minimum episodes
        if self.phase_episode_count < phase['episodes'] // 3:
            return False
            
        # Check win rate if we have enough data
        if len(self.phase_win_rates) >= 50:
            recent_win_rate = np.mean(list(self.phase_win_rates)[-50:])
            if recent_win_rate >= phase['win_threshold']:
                self.logger.info(f"Phase {self.current_phase + 1} completed early! Win rate: {recent_win_rate:.3f}")
                return True
                
        # Force advance after max episodes
        if self.phase_episode_count >= phase['episodes']:
            avg_win_rate = np.mean(self.phase_win_rates) if self.phase_win_rates else 0
            self.logger.info(f"Phase {self.current_phase + 1} completed by episode limit. Win rate: {avg_win_rate:.3f}")
            return True
            
        return False
        
    def train_single_episode(self):
        """Train one episode - adapted from your original train() method"""
        # Create players for this episode
        opponents = self.create_opponents()
        players = [self.rl_player] + opponents
        
        # Setup game
        self.game_rules.players = players
        self.game_rules.reset_game()
        self.game_rules.dealCards()
        
        # Initialize all players
        for player in players:
            player.setOpponents([p for p in players if p != player])
            player.initialCrossOff()
            
        episode_reward = 0
        winner = None
        max_turns = 200
        
        while winner is None and self.game_rules.turn < max_turns:
            current_player_idx = self.game_rules.turn % len(players)
            current_player = players[current_player_idx]
            
            if not current_player.inGame:
                self.game_rules.turn += 1
                continue
                
            if current_player.type == "RL":
                self.total_steps += 1
                
                # Get observation and valid actions
                obs = build_observation(self.game_rules, current_player)
                valid_mask = build_action_mask(self.game_rules, current_player)
                
                # Player takes action
                winner = current_player.playTurn(obs, valid_mask)
                
                # Get reward and next state
                reward = self.game_rules.get_last_reward()
                episode_reward += reward
                next_obs = build_observation(self.game_rules, current_player)
                done = bool(winner)
                
                # Store transition
                current_player.store_transition(reward, next_obs, done)
                
                # Training step (from your original logic)
                if (self.total_steps > self.learning_starts and 
                    self.total_steps % self.train_freq == 0 and
                    len(self.replay_buffer) >= self.batch_size):
                    
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.agent.learn(batch)
                    
            else:
                # Opponent turn
                winner = current_player.playTurn(None, None)
                
            self.game_rules.turn += 1
            
        # Episode finished
        self.phase_episode_count += 1
        self.total_episodes += 1
        
        # Record win/loss
        if winner == self.rl_player.name:
            self.phase_win_rates.append(1)
        else:
            self.phase_win_rates.append(0)
            
        # Update epsilon (phase-specific decay)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)
        
        return episode_reward, winner
        
    def train(self):
        """Main training loop with curriculum progression"""
        while self.current_phase < len(self.phases):
            # Train one episode
            episode_reward, winner = self.train_single_episode()
            
            # Update target network periodically
            if self.total_episodes % self.target_update == 0:
                self.agent.update_target_network()
                
            # Logging
            if self.phase_episode_count % self.log_interval == 0:
                self.log_progress()
                
            # Save checkpoint
            if self.checkpoint_path and self.total_episodes % 200 == 0:
                self.save_checkpoint()
                
            # Check if should advance phase
            if self.should_advance_phase():
                if not self.setup_phase(self.current_phase + 1):
                    break  # All phases complete
                    
        self.logger.info(f"Training completed! Total episodes: {self.total_episodes}")
        
    def log_progress(self):
        """Log current training progress"""
        phase = self.phases[self.current_phase]
        recent_win_rate = (np.mean(list(self.phase_win_rates)[-self.log_interval:]) 
                          if len(self.phase_win_rates) >= self.log_interval 
                          else np.mean(self.phase_win_rates))
        
        self.logger.info(
            f"Phase {self.current_phase + 1} ({phase['name']}): "
            f"Episode {self.phase_episode_count}/{phase['episodes']}, "
            f"Total Steps: {self.total_steps}, "
            f"Win Rate: {recent_win_rate:.3f}, "
            f"Epsilon: {self.epsilon:.3f}"
        )
        
    def save_checkpoint(self):
        """Save model checkpoint - adapted from your original"""
        try:
            checkpoint = {
                'total_episodes': self.total_episodes,
                'current_phase': self.current_phase,
                'phase_episode_count': self.phase_episode_count,
                'model_state_dict': self.agent.q_net.state_dict(),
                'target_model_state_dict': self.agent.target_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'win_rates': list(self.phase_win_rates)
            }
            
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save(checkpoint, self.checkpoint_path)
            self.logger.info(f"Checkpoint saved at episode {self.total_episodes}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.agent.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.current_phase = checkpoint.get('current_phase', 0)
            self.phase_episode_count = checkpoint.get('phase_episode_count', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            
            if 'win_rates' in checkpoint:
                self.phase_win_rates = deque(checkpoint['win_rates'], maxlen=100)
                
            # Re-setup current phase
            self.setup_phase(self.current_phase)
            
            self.logger.info(f"Resumed from checkpoint: Phase {self.current_phase + 1}, Episode {self.total_episodes}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False