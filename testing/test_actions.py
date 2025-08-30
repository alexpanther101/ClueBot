import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClueBasics.GameRules import GameRules
from agents import EliminationBot
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.replay import ReplayBuffer
from rl.utils import build_observation, build_action_mask, decode_action, sizes_from_game

def test_action_system():
    print("=== Testing RL Action System ===")
    
    try:
        # 1. Create game and players
        print("1. Creating game and players...")
        game = GameRules([])
        
        # Create RL agent
        S, W, R, C, P = sizes_from_game(game)
        print(f"Game dimensions: S={S}, W={W}, R={R}, C={C}, P={P}")
        
        input_dim = (P + 1) * C + 1  # Same as in your main.py
        print(f"Input dimension: {input_dim}")
        
        agent = DQNAgent(game_rules=game, input_dim=input_dim)
        rl_player = RLPlayer("TestBot", game, "RL")
        rl_player.agent = agent
        
        # Create opponent
        opponent = EliminationBot("Opponent", game, "Elimination")
        
        # Set up players
        players = [rl_player, opponent]
        game.players = players
        
        for p in players:
            p.setOpponents([op for op in players if op != p])
        
        print("✓ Players created successfully")
        
    except Exception as e:
        print(f"✗ Failed to create players: {e}")
        return
    
    try:
        # 2. Initialize game
        print("\n2. Initializing game...")
        game.reset_game()
        game.dealCards()
        
        for player in players:
            player.initialCrossOff()
        
        print("✓ Game initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize game: {e}")
        return
    
    try:
        # 3. Test observation building
        print("\n3. Testing observation building...")
        obs = build_observation(game, rl_player)
        input_dim = len(obs)
        agent.input_dim = input_dim
        print(f"✓ Observation created with shape: {len(obs)}")
        print(f"   Observation type: {type(obs)}")
        print(f"   Sample values: {obs[:5]} ... {obs[-5:]}")
        
    except Exception as e:
        print(f"✗ Failed to build observation: {e}")
        return
    
    try:
        # 4. Test action mask
        print("\n4. Testing action mask...")
        mask = build_action_mask(game, rl_player)
        print(f"✓ Action mask created with shape: {len(mask)}")
        print(f"   Valid actions: {sum(mask)}/{len(mask)}")
        print(f"   Mask type: {type(mask)}")
        
    except Exception as e:
        print(f"✗ Failed to build action mask: {e}")
        return
    
    try:
        # 5. Test action decoding
        print("\n5. Testing action decoding...")
        print("First 10 valid actions:")
        
        valid_actions_tested = 0
        for i in range(len(mask)):
            if mask[i] and valid_actions_tested < 10:
                try:
                    action_type, payload = decode_action(game, i)
                    print(f"   Action {i}: {action_type} - {payload}")
                    valid_actions_tested += 1
                except Exception as e:
                    print(f"   Action {i}: ERROR - {e}")
                    
        if valid_actions_tested == 0:
            print("   ✗ No valid actions found!")
        else:
            print(f"✓ Decoded {valid_actions_tested} actions successfully")
            
    except Exception as e:
        print(f"✗ Failed to decode actions: {e}")
        return
    
    try:
        # 6. Test agent action selection
        print("\n6. Testing agent action selection...")
        epsilon = 0.1  # Low epsilon for more deterministic behavior
        
        action_idx = agent.select_action(obs, mask, epsilon)
        print(f"✓ Agent selected action: {action_idx}")
        
        if action_idx == -1:
            print("   ⚠ Warning: Agent returned -1 (no valid action)")
        elif action_idx >= len(mask):
            print(f"   ✗ Error: Action index {action_idx} out of range")
        elif not mask[action_idx]:
            print(f"   ✗ Error: Agent selected invalid action {action_idx}")
        else:
            action_type, payload = decode_action(game, action_idx)
            print(f"   Decoded action: {action_type} - {payload}")
            
    except Exception as e:
        print(f"✗ Failed agent action selection: {e}")
        return
    
    try:
        # 7. Test reward calculation
        print("\n7. Testing reward calculation...")
        
        # Test suggestion reward
        if hasattr(game, 'calculate_suggestion_reward'):
            reward = game.calculate_suggestion_reward(opponent, rl_player, 0)
            print(f"✓ Suggestion reward: {reward}")
        
        # Test accusation rewards
        if hasattr(game, 'calculate_accusation_reward'):
            correct_reward = game.calculate_accusation_reward(True)
            wrong_reward = game.calculate_accusation_reward(False)
            print(f"✓ Correct accusation reward: {correct_reward}")
            print(f"✓ Wrong accusation reward: {wrong_reward}")
            
    except Exception as e:
        print(f"✗ Failed reward calculation test: {e}")
    
    try:
        # 8. Check card indices
        print("\n8. Checking card global indices...")
        cards_with_indices = 0
        for card in game.cards:
            if hasattr(card, 'global_index'):
                cards_with_indices += 1
            else:
                print(f"   ⚠ Card {card.name} missing global_index")
                
        print(f"✓ {cards_with_indices}/{len(game.cards)} cards have global indices")
        
    except Exception as e:
        print(f"✗ Failed card index check: {e}")
    
    print("\n=== Test Complete ===")
    
    # Summary of what to check
    print("\n=== What to verify: ===")
    print("1. All steps should show ✓ (success)")
    print("2. Observation shape should match your input_dim")
    print("3. Should have valid actions > 0")
    print("4. Action decoding should not error")
    print("5. Agent should select valid action indices")
    print("6. All cards should have global_index attribute")

if __name__ == "__main__":
    test_action_system()