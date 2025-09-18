#!/usr/bin/env python3
"""
Simple integration test to verify RL agent can complete basic actions
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClueBasics.GameRules import GameRules
from agents import EliminationBot
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.utils import build_observation, build_action_mask, calculate_observation_size

def test_rl_integration():
    print("=== Simple RL Integration Test ===\n")
    
    try:
        # 1. Setup game
        print("1. Setting up game...")
        game = GameRules([])
        
        
        rl_player = RLPlayer("RLBot", game, "RL")
        
        opponent = EliminationBot("OpponentBot", game, "Elimination")
        players = [rl_player, opponent]
        
        game.players = players
        for p in players:
            p.setOpponents([op for op in players if op != p])
        
        # Create agents with correct input dimension
        input_dim = calculate_observation_size(game)
        print(input_dim)
        agent = DQNAgent(game_rules=game, input_dim=input_dim)
        rl_player.agent = agent

        game.reset_game()
        game.dealCards()
        print("   ✓ Game setup complete")
        
        # 2. Test RL player can take a turn
        print("\n2. Testing RL player turn...")
        
        obs = build_observation(game, rl_player)
        mask = build_action_mask(game, rl_player)
        
        print(f"   Observation size: {len(obs)}")
        print(f"   Valid actions: {sum(mask)}/{len(mask)}")
        
        # Mock trainer for epsilon
        class MockTrainer:
            def __init__(self):
                self.epsilon = 0.5
        
        rl_player.trainer = MockTrainer()
        
        # Try to take a turn
        try:
            print(f"Feeding an epsilon of {rl_player.trainer.epsilon} to the agent")
            result = rl_player.playTurn(obs, mask)
            print(f"   ✓ RL player completed turn, result: {result}")
        except Exception as e:
            print(f"   ✗ RL player turn failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. Test opponent can take a turn
        print("\n3. Testing opponent turn...")
        try:
            result = opponent.playTurn(None, None)
            print(f"   ✓ Opponent completed turn, result: {result}")
        except Exception as e:
            print(f"   ✗ Opponent turn failed: {e}")
            return False
        
        # 4. Test a few more turns to see if game progresses
        print("\n4. Testing game progression...")
        max_turns = 10
        turn_count = 0
        winner = None
        
        while winner is None and turn_count < max_turns:
            current_player_idx = game.turn % len(players)
            current_player = players[current_player_idx]
            
            if not current_player.inGame:
                game.turn += 1
                continue
            
            if current_player.type == "RL":
                obs = build_observation(game, current_player)
                mask = build_action_mask(game, current_player)
                winner = current_player.playTurn(obs, mask)
            else:
                winner = current_player.playTurn(None, None)
            
            turn_count += 1
            game.turn += 1
            
            print(f"   Turn {turn_count}: {current_player.name} played")
            
            if winner:
                print(f"   ✓ Game ended with winner: {winner}")
                break
        
        if not winner and turn_count >= max_turns:
            print(f"   ✓ Game progressed {turn_count} turns without crashing")
        
        # 5. Test action selection consistency
        print("\n5. Testing action selection consistency...")
        obs = build_observation(game, rl_player)
        mask = build_action_mask(game, rl_player)
        
        # Test multiple action selections
        actions = []
        for i in range(5):
            action = agent.select_action(obs, mask, epsilon=0.1)
            actions.append(action)
        
        valid_actions = [a for a in actions if a != -1 and a < len(mask) and mask[a]]
        print(f"   Selected actions: {actions}")
        print(f"   Valid actions: {len(valid_actions)}/5")
        
        if len(valid_actions) >= 3:
            print("   ✓ Agent consistently selects valid actions")
        else:
            print("   ✗ Agent often selects invalid actions")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_integration():
    """Test if reward system integrates properly with RL player"""
    print("\n=== Testing Reward Integration ===")
    
    game = GameRules([])
    rl_player = RLPlayer("RLBot", game, "RL")
    
    # Test reward calculation methods exist and return numbers
    try:
        reward1 = game.calculate_accusation_reward(True)
        reward2 = game.calculate_accusation_reward(False)
        print(f"Accusation rewards: {reward1}, {reward2}")
        
        if isinstance(reward1, (int, float)) and isinstance(reward2, (int, float)):
            print("✓ Accusation rewards return numeric values")
        else:
            print("✗ Accusation rewards don't return numbers")
            
    except Exception as e:
        print(f"✗ Reward calculation failed: {e}")

if __name__ == "__main__":
    success = test_rl_integration()
    test_reward_integration()
    
    print(f"\n=== Integration Test {'PASSED' if success else 'FAILED'} ===")
    if success:
        print("Your RL system basic integration works!")
        print("Next steps:")
        print("1. Run the critical fixes test")
        print("2. Fix any issues found")
        print("3. Try training for a few episodes")
    else:
        print("Fix the integration issues before proceeding to training")