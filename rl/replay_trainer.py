#!/usr/bin/env python3
"""
Simple trainer focused on fixing the replay buffer storage issue
"""
import sys
import os
import torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClueBasics.GameRules import GameRules
from agents import EliminationBot
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.replay import ReplayBuffer
from rl.utils import build_observation, build_action_mask, calculate_observation_size, decode_action

class ReplayFixTrainer:
    def __init__(self, game_rules, rl_player, dqn_agent, replay_buffer):
        self.game_rules = game_rules
        self.rl_player = rl_player
        self.agent = dqn_agent
        self.replay_buffer = replay_buffer
        
        self.epsilon = 0.8
        self.total_steps = 0
        self.total_episodes = 0
        
        # Connect components
        self.rl_player.agent = self.agent
        self.rl_player.trainer = self
        
        print("Trainer initialized")
        print(f"Replay buffer capacity: {self.replay_buffer.capacity}")
        
    def manual_store_transition(self, state, action, reward, next_state, done):
        """Manually store transition to debug replay buffer"""
        print(f"Storing transition: action={action}, reward={reward}, done={done}")
        
        # FIXED: Create a mask with the correct action space size, not observation size
        from rl.utils import sizes_from_game
        S, W, R, C, P = sizes_from_game(self.game_rules)
        action_space_size = (S * W * R) * 2  # suggestions + accusations
        mask = np.ones(action_space_size, dtype=bool)  # All actions valid for storage
        
        print(f"Created mask with size: {len(mask)} for action space size: {action_space_size}")
        
        try:
            self.replay_buffer.push(state, action, reward, next_state, done, mask)
            print(f"Transition stored successfully. Buffer size now: {len(self.replay_buffer)}")
        except Exception as e:
            print(f"Error storing transition: {e}")
            import traceback
            traceback.print_exc()
    
    def train_episode(self):
        """Simple episode to test replay buffer storage"""
        print(f"\n--- Episode {self.total_episodes + 1} ---")
        
        # Create opponent
        opponent = EliminationBot(f"Opponent_{self.total_episodes}", self.game_rules, "Elimination")
        players = [self.rl_player, opponent]
        
        # Reset player states
        self.rl_player.inGame = True
        self.rl_player.cards = []
        self.rl_player.numCards = 0
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
        
        # Take a few turns to test replay buffer
        max_turns = 10
        current_player_idx = 0
        
        for turn in range(max_turns):
            current_player = players[current_player_idx]
            
            if not current_player.inGame:
                break
                
            if current_player.type == "RL":
                print(f"Turn {turn}: RL Player's turn")
                
                # Get observation
                obs = build_observation(self.game_rules, current_player)
                print(f"Observation shape: {obs.shape}")
                
                # Get proper action mask
                mask = build_action_mask(self.game_rules, current_player)
                print(f"Mask allows {np.sum(mask)} actions")
                
                # Select action
                action_idx = self.agent.select_action(obs, mask, self.epsilon)
                print(f"Selected action: {action_idx}")
                
                # Decode action
                action_type, payload = decode_action(self.game_rules, action_idx)
                print(f"Action type: {action_type}")
                
                if action_type == "suggest":
                    s_idx, w_idx, r_idx = payload
                    suspect = self.game_rules.suspectCards[self.game_rules.SUSPECTS[s_idx]]
                    weapon = self.game_rules.weaponCards[self.game_rules.WEAPONS[w_idx]]
                    room = self.game_rules.roomCards[self.game_rules.ROOMS[r_idx]]
                    
                    print(f"Making suggestion: {suspect.name}, {weapon.name}, {room.name}")
                    
                    # Execute suggestion
                    responder, card_shown = self.game_rules.makeSuggestion(current_player, suspect, weapon, room)
                    
                    # Simple reward
                    reward = 10.0 if responder else 1.0
                    print(f"Suggestion reward: {reward}")
                    
                    # Get next observation
                    next_obs = build_observation(self.game_rules, current_player)
                    
                    # Store transition manually with correct mask
                    self.manual_store_transition(obs, action_idx, reward, next_obs, False)
                    
                elif action_type == "accuse":
                    print("Accusation selected (not implemented in this test)")
                    break
                    
            else:
                # Opponent turn - simplified
                print(f"Turn {turn}: Opponent's turn")
                try:
                    result = current_player.playTurn(None, None)
                    if result:
                        print(f"Opponent won!")
                        # Store final transition for RL player with correct mask
                        obs = build_observation(self.game_rules, self.rl_player)
                        next_obs = obs  # Same observation
                        self.manual_store_transition(obs, 0, -100.0, next_obs, True)
                        break
                except Exception as e:
                    print(f"Error in opponent turn: {e}")
                    break
            
            current_player_idx = (current_player_idx + 1) % len(players)
            
            # Stop if we have enough data to test
            if len(self.replay_buffer) >= 5:
                print(f"Collected {len(self.replay_buffer)} transitions, stopping episode")
                break
        
        self.total_episodes += 1
        
        # Test learning if we have enough data
        if len(self.replay_buffer) >= 4:
            print("Testing learning...")
            try:
                batch = self.replay_buffer.sample(2)  # Small batch
                print(f"Sampled batch with {len(batch.state)} transitions")
                
                # Verify mask dimensions before learning
                print(f"Batch mask shape: {np.array(batch.mask).shape}")
                print(f"Expected mask size per sample: {self.agent.output_dim}")
                
                # Try to learn
                self.agent.learn(batch)
                print("Learning step completed successfully")
                
            except Exception as e:
                print(f"Error during learning: {e}")
                import traceback
                traceback.print_exc()
        
        return 0.0, None, turn + 1
    
    def test_replay_buffer(self):
        """Direct test of replay buffer functionality"""
        print("\n=== Testing Replay Buffer Directly ===")
        
        # Create dummy data with correct dimensions
        dummy_state = np.random.random(100).astype(np.float32)
        dummy_next_state = np.random.random(100).astype(np.float32)
        
        # FIXED: Create mask with correct action space size
        from rl.utils import sizes_from_game
        S, W, R, C, P = sizes_from_game(self.game_rules)
        action_space_size = (S * W * R) * 2  # suggestions + accusations
        dummy_mask = np.ones(action_space_size, dtype=bool)
        
        print("Creating dummy transition...")
        print(f"State size: {len(dummy_state)}")
        print(f"Mask size: {len(dummy_mask)}")
        print(f"Expected action space size: {action_space_size}")
        
        try:
            self.replay_buffer.push(
                state=dummy_state,
                action=5,
                reward=1.0,
                next_state=dummy_next_state,
                done=False,
                mask=dummy_mask
            )
            print(f"Success! Buffer size: {len(self.replay_buffer)}")
            
            # Test sampling
            if len(self.replay_buffer) >= 1:
                batch = self.replay_buffer.sample(1)
                print(f"Sampling success! Batch size: {len(batch.state)}")
                print(f"Batch mask shape: {np.array(batch.mask).shape}")
                
        except Exception as e:
            print(f"Direct test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def train(self, num_episodes=5):
        """Train for a few episodes to test replay functionality"""
        print("Starting replay buffer diagnostic training...")
        
        # First, test replay buffer directly
        self.test_replay_buffer()
        
        # Then test with actual episodes
        for episode in range(num_episodes):
            episode_reward, winner, turn_count = self.train_episode()
            
            print(f"Episode {episode + 1} complete:")
            print(f"  Buffer size: {len(self.replay_buffer)}")
            print(f"  Turn count: {turn_count}")
            
            if len(self.replay_buffer) >= 10:
                print("Buffer working correctly - stopping test")
                break
        
        print(f"\nTest complete. Final buffer size: {len(self.replay_buffer)}")


def main():
    """Test replay buffer functionality"""
    print("=== Replay Buffer Fix Test ===")
    
    # Setup
    game = GameRules([])
    rl_player = RLPlayer("TestBot", game, "RL")
    dummy_opponent = EliminationBot("DummyOpp", game, "Elimination")
    players = [rl_player, dummy_opponent]
    
    game.players = players
    for p in players:
        p.setOpponents([op for op in players if op != p])
    
    input_dim = calculate_observation_size(game)
    print(f"Input dimension: {input_dim}")
    
    agent = DQNAgent(game_rules=game, input_dim=input_dim)
    replay_buffer = ReplayBuffer(capacity=1000, n_step=1)  # Simple 1-step for debugging
    
    trainer = ReplayFixTrainer(
        game_rules=game,
        rl_player=rl_player,
        dqn_agent=agent,
        replay_buffer=replay_buffer
    )
    
    # Run test
    trainer.train(num_episodes=3)

if __name__ == "__main__":
    main()