import numpy as np
import torch
from collections import deque
import logging
import os
from typing import Dict, List, Optional, Tuple

class Trainer:
    def __init__(
        self,
        game_rules,
        rl_player,
        dqn_agent,
        replay_buffer,
        n_step=3,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        max_episodes=5000,
        target_update=100,
        learning_starts=1000,
        train_freq=4,
        checkpoint_path=None,
        log_interval=50,
        eval_episodes=10,
        device=None
    ):
        self.game_rules = game_rules
        self.rl_player = rl_player
        self.agent = dqn_agent
        self.replay_buffer = replay_buffer
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training parameters
        self.max_episodes = max_episodes
        self.target_update = target_update
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        
        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = deque(maxlen=100)
        self.losses = []
        
        # Checkpointing and logging
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self.eval_episodes = eval_episodes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Connect agent to RL player
        self.rl_player.agent = self.agent
        self.rl_player.trainer = self

    def run_episode(self, training=True):
        """Run a single game episode"""
        # Reset the game
        self.game_rules.__init__(self.game_rules.players)  # Reinitialize game
        
        # Deal cards and set up initial state
        self.game_rules.dealCards()
        
        # Initialize episode tracking
        episode_reward = 0
        episode_length = 0
        game_won = False
        
        # Store initial state for the RL player
        initial_obs = self._get_observation()
        self.rl_player.last_obs = initial_obs
        self.rl_player.last_action = None
        
        try:
            # Run the game loop
            while self.game_rules.checkAllPlayers():
                self.game_rules.gameTurn += 1
                episode_length += 1
                
                # Check if game is taking too long (prevent infinite games)
                if episode_length > 500:
                    break
                
                for player in self.game_rules.players:
                    if not player.inGame:
                        continue
                        
                    if player == self.rl_player:
                        # RL player's turn
                        obs = self._get_observation()
                        valid_mask = self._get_valid_actions_mask()
                        
                        # Select action
                        if training:
                            action = self.agent.select_action(obs, valid_mask, self.epsilon)
                        else:
                            action = self.agent.select_action(obs, valid_mask, 0.0)  # No exploration
                        
                        # Execute action and get reward
                        reward, done, info = self._execute_action(action, player)
                        
                        # Store transition if training
                        if training and self.rl_player.last_obs is not None:
                            self.replay_buffer.push(
                                self.rl_player.last_obs,
                                self.rl_player.last_action,
                                reward,
                                obs if not done else None,
                                done,
                                valid_mask
                            )
                        
                        # Update state
                        self.rl_player.last_obs = obs
                        self.rl_player.last_action = action
                        episode_reward += reward
                        
                        if done:
                            game_won = info.get('won', False)
                            break
                            
                        # Train the agent periodically
                        if training and self.steps_done % self.train_freq == 0:
                            loss = self.optimize_model()
                            if loss is not None:
                                self.losses.append(loss)
                        
                        self.steps_done += 1
                        
                    else:
                        # Other players' turns (use their existing AI)
                        obs = self.game_rules.getObservation(player)
                        valid_mask = self.game_rules.getValidActionsMask(player)
                        winner = player.playTurn(obs, valid_mask)
                        if winner:
                            done = True
                            break
                
                if not self.game_rules.checkAllPlayers():
                    break
                    
        except Exception as e:
            self.logger.error(f"Episode failed: {e}")
            episode_reward = -10  # Penalty for crashed episodes
        
        # Record episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.win_rates.append(1.0 if game_won else 0.0)
        self.episodes_done += 1
        
        # Update target network
        if training and self.steps_done % self.target_update == 0:
            self.agent.update_target_network()
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return episode_reward, episode_length, game_won

    def _get_observation(self):
        """Get current observation for RL player"""
        from rl.utils import build_observation
        return build_observation(self.game_rules, self.rl_player)

    def _get_valid_actions_mask(self):
        """Get valid actions mask for RL player"""
        from rl.utils import build_action_mask
        return build_action_mask(self.game_rules, self.rl_player)

    def _execute_action(self, action_idx: int, player) -> Tuple[float, bool, Dict]:
        """Execute the selected action and return reward, done, info"""
        from rl.utils import decode_action, global_index_to_card
        
        try:
            action_type, payload = decode_action(self.game_rules, action_idx)
            
            if action_type == "suggest":
                s_idx, w_idx, r_idx = payload
                suspect = self.game_rules.SUSPECTS[s_idx]
                weapon = self.game_rules.WEAPONS[w_idx]
                room = self.game_rules.ROOMS[r_idx]
                
                # Execute suggestion
                responder, card_shown = self.game_rules.makeSuggestion(
                    player, suspect, weapon, room
                )
                
                # Reward based on information gained
                if card_shown:
                    reward = 0.1  # Small positive reward for gaining information
                else:
                    reward = 0.2  # Larger reward if no one could refute
                
                return reward, False, {'action': 'suggest', 'responder': responder, 'card': card_shown}
                
            elif action_type == "accuse":
                s_idx, w_idx, r_idx = payload
                suspect_card = self.game_rules.suspectCards[self.game_rules.SUSPECTS[s_idx]]
                weapon_card = self.game_rules.weaponCards[self.game_rules.WEAPONS[w_idx]]
                room_card = self.game_rules.roomCards[self.game_rules.ROOMS[r_idx]]
                
                # Execute accusation
                won = self.game_rules.makeAccusation(player, suspect_card, weapon_card, room_card)
                
                if won:
                    reward = 10.0  # Large positive reward for winning
                    return reward, True, {'action': 'accuse', 'won': True}
                else:
                    reward = -5.0  # Large negative reward for wrong accusation
                    return reward, True, {'action': 'accuse', 'won': False}
                    
            elif action_type == "reveal":
                # This shouldn't happen during regular turns
                # Reveal actions are handled in the revealCard method
                return -0.1, False, {'action': 'invalid_reveal'}
                
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return -1.0, False, {'action': 'error', 'error': str(e)}
        
        return 0.0, False, {'action': 'unknown'}

    def optimize_model(self):
        """Sample a batch and train DQN"""
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        try:
            # Sample batch from replay buffer
            transitions = self.replay_buffer.sample(self.batch_size)
            
            # Convert to tensors
            states = torch.tensor(np.array(transitions.state), dtype=torch.float32, device=self.device)
            actions = torch.tensor(transitions.action, dtype=torch.long, device=self.device)
            rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device)
            
            # Handle next_states (some might be None for terminal states)
            next_states = []
            dones = []
            for next_state, done in zip(transitions.next_state, transitions.done):
                if done or next_state is None:
                    next_states.append(np.zeros_like(states[0].cpu().numpy()))
                    dones.append(1.0)
                else:
                    next_states.append(next_state)
                    dones.append(0.0)
            
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
            
            # Create batch dictionary
            batch = {
                'obs': states,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_states,
                'dones': dones
            }
            
            # Train the agent
            loss = self.agent.learn(batch)
            return loss
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return None

    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.max_episodes} episodes")
        
        for episode in range(self.max_episodes):
            # Run training episode
            episode_reward, episode_length, won = self.run_episode(training=True)
            
            # Log progress
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1)
            
            # Save checkpoint
            if self.checkpoint_path and (episode + 1) % (self.log_interval * 2) == 0:
                self._save_checkpoint(episode + 1)
        
        self.logger.info("Training completed!")
        
        # Final evaluation
        if self.eval_episodes > 0:
            self._evaluate()

    def _log_progress(self, episode):
        """Log training progress"""
        recent_rewards = self.episode_rewards[-self.log_interval:]
        recent_lengths = self.episode_lengths[-self.log_interval:]
        recent_wins = list(self.win_rates)[-self.log_interval:] if len(self.win_rates) >= self.log_interval else list(self.win_rates)
        recent_losses = self.losses[-100:] if len(self.losses) >= 100 else self.losses
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        win_rate = np.mean(recent_wins) if recent_wins else 0
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        
        self.logger.info(
            f"Episode {episode}/{self.max_episodes} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Length: {avg_length:.1f} | "
            f"Win Rate: {win_rate:.3f} | "
            f"Epsilon: {self.epsilon:.3f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Buffer Size: {len(self.replay_buffer)}"
        )

    def _evaluate(self):
        """Run evaluation episodes without training"""
        self.logger.info(f"Running {self.eval_episodes} evaluation episodes...")
        
        eval_rewards = []
        eval_wins = []
        
        for _ in range(self.eval_episodes):
            reward, length, won = self.run_episode(training=False)
            eval_rewards.append(reward)
            eval_wins.append(won)
        
        avg_eval_reward = np.mean(eval_rewards)
        eval_win_rate = np.mean(eval_wins)
        
        self.logger.info(
            f"Evaluation Results: "
            f"Avg Reward: {avg_eval_reward:.2f} | "
            f"Win Rate: {eval_win_rate:.3f}"
        )

    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        if not self.checkpoint_path:
            return
            
        try:
            checkpoint = {
                'episode': episode,
                'model_state_dict': self.agent.q_net.state_dict(),
                'target_model_state_dict': self.agent.target_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'win_rates': list(self.win_rates)
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            
            torch.save(checkpoint, self.checkpoint_path)
            self.logger.info(f"Checkpoint saved at episode {episode}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.agent.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            
            if 'win_rates' in checkpoint:
                self.win_rates = deque(checkpoint['win_rates'], maxlen=100)
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint.get('episode', 0)
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return 0