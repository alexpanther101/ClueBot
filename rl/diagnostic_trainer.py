#!/usr/bin/env python3
"""
Diagnostic trainer to debug why agent isn't making accusations - FIXED VERSION
Uses the same working pattern as replay_trainer but with diagnostic features
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
from rl.utils import build_observation, build_action_mask, calculate_observation_size, decode_action, sizes_from_game

class DiagnosticTrainer:
    def __init__(self, game_rules, rl_player, dqn_agent, replay_buffer):
        self.game_rules = game_rules
        self.rl_player = rl_player
        self.agent = dqn_agent
        self.replay_buffer = replay_buffer
        
        # More permissive epsilon schedule for debugging
        self.epsilon = 0.8
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        
        self.total_steps = 0
        
        # Connect components
        self.rl_player.agent = self.agent
        self.rl_player.trainer = self
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.total_episodes = 0
        self.accusation_count = 0
        self.suggestion_count = 0
        
        # Debug tracking
        self.action_type_counts = {"suggest": 0, "accuse": 0, "invalid": 0}
        
        print("Diagnostic Trainer initialized")
        
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition with proper mask"""
        S, W, R, C, P = sizes_from_game(self.game_rules)
        action_space_size = (S * W * R) * 2  # suggestions + accusations
        mask = np.ones(action_space_size, dtype=bool)  # All actions valid for storage
        
        try:
            self.replay_buffer.push(state, action, reward, next_state, done, mask)
        except Exception as e:
            print(f"Error storing transition: {e}")
    
    def apply_debug_accusation_masking(self, player, mask):
        """Apply more permissive accusation masking for debugging"""
        S, W, R = len(self.game_rules.SUSPECTS), len(self.game_rules.WEAPONS), len(self.game_rules.ROOMS)
        num_suggestions = S * W * R
        
        # More lenient accusation conditions
        should_allow_accusations = (
            self.game_rules.turn > 15 or  # After 15 turns
            len(player.possibleSuspects) <= 3 or
            len(player.possibleWeapons) <= 3 or
            len(player.possibleRooms) <= 3
        )
        
        if should_allow_accusations:
            # Allow top accusation candidates
            best_accusations = self.get_top_accusation_candidates(player, max_candidates=20)
            
            for s_idx, w_idx, r_idx in best_accusations:
                acc_idx = num_suggestions + s_idx * W * R + w_idx * R + r_idx
                if acc_idx < len(mask):
                    mask[acc_idx] = True
                    
            if len(best_accusations) > 0:
                print(f"DEBUG: Enabled {len(best_accusations)} accusations at turn {self.game_rules.turn}")
        
        return mask
    
    def get_top_accusation_candidates(self, player, max_candidates=10):
        """Get promising accusation candidates"""
        candidates = []
        
        for suspect in player.possibleSuspects:
            for weapon in player.possibleWeapons:
                for room in player.possibleRooms:
                    s_idx = self.game_rules.SUSPECTS.index(suspect.name)
                    w_idx = self.game_rules.WEAPONS.index(weapon.name)
                    r_idx = self.game_rules.ROOMS.index(room.name)
                    
                    joint_prob = (player.getProbability("Solution", suspect) *
                                 player.getProbability("Solution", weapon) *
                                 player.getProbability("Solution", room))
                    
                    candidates.append((joint_prob, s_idx, w_idx, r_idx))
        
        candidates.sort(reverse=True, key=lambda x: x[0])
        return [(s, w, r) for _, s, w, r in candidates[:max_candidates]]
    
    def print_debug_info(self, player, mask, turn):
        """Print debugging information"""
        S, W, R = len(self.game_rules.SUSPECTS), len(self.game_rules.WEAPONS), len(self.game_rules.ROOMS)
        num_suggestions = S * W * R
        num_accusations = S * W * R
        
        suggestion_mask = mask[0:num_suggestions]
        accusation_mask = mask[num_suggestions:num_suggestions + num_accusations]
        
        print(f"\n--- TURN {turn} DEBUG ---")
        print(f"Valid suggestions: {np.sum(suggestion_mask)}/{num_suggestions}")
        print(f"Valid accusations: {np.sum(accusation_mask)}/{num_accusations}")
        
        if player.possibleSuspects and player.possibleWeapons and player.possibleRooms:
            suspect_conf = max(player.getProbability("Solution", card) for card in player.possibleSuspects)
            weapon_conf = max(player.getProbability("Solution", card) for card in player.possibleWeapons)
            room_conf = max(player.getProbability("Solution", card) for card in player.possibleRooms)
            joint_conf = suspect_conf * weapon_conf * room_conf
            
            print(f"Confidences - S:{suspect_conf:.3f}, W:{weapon_conf:.3f}, R:{room_conf:.3f}")
            print(f"Joint confidence: {joint_conf:.3f}")
        
        print(f"Remaining cards - S:{len(player.possibleSuspects)}, W:{len(player.possibleWeapons)}, R:{len(player.possibleRooms)}")
        print("------------------------\n")
    
    def train_episode(self):
        """Train a single episode using the working replay_trainer pattern"""
        print(f"\n--- Episode {self.total_episodes + 1} ---")
        
        # Create opponent
        opponent = TriggerHappyBot(f"Opponent_{self.total_episodes}", self.game_rules, "Trigger")
        players = [self.rl_player, opponent]
        
        # Reset player states
        self.rl_player.inGame = True
        self.rl_player.cards = []
        self.rl_player.numCards = 0
        self.rl_player.cumulative_episode_reward = 0
        opponent.inGame = True
        opponent.cards = []
        opponent.numCards = 0
        
        # Setup game
        self.game_rules.players = players
        for p in players:
            p.setOpponents([op for op in players if op != p])
            
        self.game_rules.reset_game()
        self.game_rules.dealCards()
        
        # Initialize belief matrices
        for player in players:
            player.initialCrossOff()
        
        print(f"Game setup complete. RL player has {self.rl_player.numCards} cards")
        
        # Game loop using the working pattern from replay_trainer
        max_turns = 50
        current_player_idx = 0
        episode_reward = 0
        episode_accusations = 0
        episode_suggestions = 0
        
        for turn in range(max_turns):
            current_player = players[current_player_idx]
            
            if not current_player.inGame:
                break
                
            print(f"Turn {turn}: {current_player.name}'s turn (type: {current_player.type})")
                
            if current_player.type == "RL":
                # RL Player's turn - use the working pattern
                self.total_steps += 1
                
                # Get observation
                obs = build_observation(self.game_rules, current_player)
                print(f"Observation shape: {obs.shape}")
                
                # Get action mask
                mask = build_action_mask(self.game_rules, current_player)
                
                # Apply debug masking
                mask = self.apply_debug_accusation_masking(current_player, mask)
                
                print(f"Mask allows {np.sum(mask)} actions")
                
                # Debug info every few turns
                if turn % 5 == 0:
                    self.print_debug_info(current_player, mask, turn)
                
                # Select action
                action_idx = self.agent.select_action(obs, mask, self.epsilon)
                print(f"Selected action: {action_idx}")
                print(f"Selected action: {action_idx}")
                if action_idx >= 324:
                    print(f"This is an ACCUSATION (index {action_idx - 324})")
                else:
                    print(f"This is a suggestion (index {action_idx})")
                if action_idx < 0:
                    print("Invalid action selected, skipping turn")
                    break
                
                # Decode action
                try:
                    action_type, payload = decode_action(self.game_rules, action_idx)
                    print(f"Action type: {action_type}")
                    
                    # Track action type
                    if action_type in self.action_type_counts:
                        self.action_type_counts[action_type] += 1
                    
                    if action_type == "suggest":
                        episode_suggestions += 1
                        self.suggestion_count += 1
                        
                        s_idx, w_idx, r_idx = payload
                        suspect = self.game_rules.suspectCards[self.game_rules.SUSPECTS[s_idx]]
                        weapon = self.game_rules.weaponCards[self.game_rules.WEAPONS[w_idx]]
                        room = self.game_rules.roomCards[self.game_rules.ROOMS[r_idx]]
                        
                        print(f"Making suggestion: {suspect.name}, {weapon.name}, {room.name}")
                        
                        # Execute suggestion
                        responder, card_shown = self.game_rules.makeSuggestion(current_player, suspect, weapon, room)
                        
                        # Calculate reward
                        reward = 10.0 if responder else 1.0
                        print(f"Suggestion reward: {reward}")
                        episode_reward += reward
                        
                        # Get next observation and store transition
                        next_obs = build_observation(self.game_rules, current_player)
                        self.store_transition(obs, action_idx, reward, next_obs, False)
                        
                    elif action_type == "accuse":
                        episode_accusations += 1
                        self.accusation_count += 1
                        
                        print(f"ACCUSATION MADE! Episode {self.total_episodes}, Turn {turn}")
                        
                        s_idx, w_idx, r_idx = payload
                        suspect = self.game_rules.suspectCards[self.game_rules.SUSPECTS[s_idx]]
                        weapon = self.game_rules.weaponCards[self.game_rules.WEAPONS[w_idx]]
                        room = self.game_rules.roomCards[self.game_rules.ROOMS[r_idx]]
                        
                        print(f"Accusing: {suspect.name}, {weapon.name}, {room.name}")
                        
                        # Execute accusation
                        is_correct = self.game_rules.makeAccusation(current_player, suspect, weapon, room)
                        
                        reward = 200.0 if is_correct else -200.0
                        episode_reward += reward
                        print(f"Accusation reward: {reward}")
                        
                        # Store final transition
                        next_obs = build_observation(self.game_rules, current_player)
                        self.store_transition(obs, action_idx, reward, next_obs, True)
                        
                        # End episode
                        if is_correct:
                            print(f"RL Player won with accusation!")
                            self.win_count += 1
                            break
                        else:
                            print(f"RL Player eliminated with wrong accusation!")
                            break
                    
                except Exception as e:
                    print(f"Error processing action: {e}")
                    break
                    
            else:
                # Opponent turn - simplified
                print(f"Opponent turn")
                try:
                    result = current_player.playTurn(None, None)
                    if result:
                        print(f"Opponent won!")
                        # Give penalty to RL player
                        if len(self.replay_buffer) > 0:
                            # Store losing transition if we have recent state
                            obs = build_observation(self.game_rules, self.rl_player)
                            self.store_transition(obs, 0, -100.0, obs, True)
                        episode_reward -= 100
                        break
                except Exception as e:
                    print(f"Error in opponent turn: {e}")
                    break
            
            current_player_idx = (current_player_idx + 1) % len(players)
            self.game_rules.turn += 1
            
            # Check if RL player is still in game
            if not self.rl_player.inGame:
                break
        
        # Train if we have enough data
        if len(self.replay_buffer) >= 16 and self.total_steps % 4 == 0:
            try:
                batch = self.replay_buffer.sample(8)
                self.agent.learn(batch)
            except Exception as e:
                print(f"Error during learning: {e}")
        
        # Record episode results
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(turn + 1)
        self.total_episodes += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Episode summary
        print(f"Episode {self.total_episodes} complete:")
        print(f"  Suggestions: {episode_suggestions}, Accusations: {episode_accusations}")
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Buffer size: {len(self.replay_buffer)}")
        
        return episode_reward, None, turn + 1
    
    def train(self, num_episodes=100):
        """Train with detailed debugging output"""
        print(f"Starting diagnostic training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_reward, winner, turn_count = self.train_episode()
            
            # Detailed logging every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"\n=== Episode {episode + 1} Summary ===")
                print(f"Win Rate: {self.win_count}/{self.total_episodes} ({self.win_count/self.total_episodes*100:.1f}%)")
                print(f"Total Accusations Made: {self.accusation_count}")
                print(f"Total Suggestions Made: {self.suggestion_count}")
                print(f"Action Type Distribution: {self.action_type_counts}")
                print(f"Epsilon: {self.epsilon:.3f}")
                if len(self.episode_lengths) >= 10:
                    print(f"Average Episode Length (last 10): {np.mean(self.episode_lengths[-10:]):.1f}")
                print(f"Buffer Size: {len(self.replay_buffer)}")
                print("===============================\n")
        
        print(f"\nDiagnostic Training Complete!")
        print(f"Final Stats:")
        print(f"  Total Episodes: {self.total_episodes}")
        print(f"  Total Accusations: {self.accusation_count}")
        print(f"  Total Suggestions: {self.suggestion_count}")
        print(f"  Win Rate: {self.win_count}/{self.total_episodes} ({self.win_count/self.total_episodes*100:.1f}%)")
        print(f"  Action Distribution: {self.action_type_counts}")

def main():
    """Diagnostic training to debug action selection"""
    print("=== Diagnostic Trainer (Fixed) ===")
    
    # Setup
    game = GameRules([])
    rl_player = RLPlayer("DiagnosticBot", game, "RL")
    dummy_opponent = TriggerHappyBot("DummyOpp", game, "Trigger")
    players = [rl_player, dummy_opponent]
    
    game.players = players
    for p in players:
        p.setOpponents([op for op in players if op != p])
    
    input_dim = calculate_observation_size(game)
    print(f"Input dimension: {input_dim}")
    
    agent = DQNAgent(game_rules=game, input_dim=input_dim)
    replay_buffer = ReplayBuffer(capacity=5000)
    
    trainer = DiagnosticTrainer(
        game_rules=game,
        rl_player=rl_player,
        dqn_agent=agent,
        replay_buffer=replay_buffer
    )
    
    # Run diagnostic training
    trainer.train(num_episodes=50)

if __name__ == "__main__":
    main()