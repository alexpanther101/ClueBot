#!/usr/bin/env python3
"""
Evaluation script that loads a saved trained agent and evaluates its performance
"""
import sys
import os
import torch
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClueBasics.GameRules import GameRules
from agents import EliminationBot, TriggerHappyBot
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.utils import build_observation, build_action_mask, calculate_observation_size, decode_action

class AgentEvaluator:
    def __init__(self, model_path="models/simple_trained_clue_agent.pth"):
        self.model_path = model_path
        self.trained_agent = None
        self.game = None
        self.evaluation_results = {}
        
    def load_trained_agent(self):
        """Load the trained agent from saved checkpoint"""
        print(f"Loading trained agent from: {self.model_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(self.model_path):
                print(f"Error: Model file not found at {self.model_path}")
                print("Make sure you've run training and saved a model first.")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Create game and agent
            self.game = GameRules([])
            
            # Get input dimensions from checkpoint or calculate
            if 'training_config' in checkpoint and 'input_dim' in checkpoint['training_config']:
                input_dim = checkpoint['training_config']['input_dim']
            else:
                input_dim = calculate_observation_size(self.game)
            
            # Create agent with correct dimensions
            self.trained_agent = DQNAgent(game_rules=self.game, input_dim=input_dim)
            
            # Load trained weights
            self.trained_agent.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.trained_agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
            self.trained_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training metadata
            self.training_info = {
                'total_episodes': checkpoint.get('total_episodes', 'Unknown'),
                'total_steps': checkpoint.get('total_steps', 'Unknown'),
                'final_epsilon': checkpoint.get('epsilon', 'Unknown'),
                'win_count': checkpoint.get('win_count', 'Unknown'),
                'episode_rewards': checkpoint.get('episode_rewards', []),
                'episode_lengths': checkpoint.get('episode_lengths', [])
            }
            
            print(f"✓ Successfully loaded trained agent!")
            print(f"  Training episodes: {self.training_info['total_episodes']}")
            print(f"  Training steps: {self.training_info['total_steps']}")
            print(f"  Final epsilon: {self.training_info['final_epsilon']}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load trained agent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_behavior_patterns(self, num_tests=20):
        """Analyze the trained agent's behavior patterns"""
        print(f"\n=== Evaluating Behavior Patterns ({num_tests} tests) ===")
        
        if not self.trained_agent:
            print("Error: No trained agent loaded")
            return None
        
        action_counts = defaultdict(int)
        action_types = defaultdict(int)
        q_value_stats = []
        
        for test in range(num_tests):
            # Setup test game
            rl_player = RLPlayer("TestBot", self.game, "RL")
            rl_player.agent = self.trained_agent
            
            opponent = EliminationBot(f"TestOpp_{test}", self.game, "Elimination")
            players = [rl_player, opponent]
            self.game.players = players
            
            for p in players:
                p.setOpponents([op for op in players if op != p])
            
            self.game.reset_game()
            self.game.dealCards()
            
            # Get observation and test action selection
            obs = build_observation(self.game, rl_player)
            mask = build_action_mask(self.game, rl_player)
            
            # Get Q-values for analysis
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.trained_agent.q_net(obs_tensor).cpu().numpy().squeeze()
            
            valid_q_values = q_values[mask]
            q_value_stats.append({
                'mean': np.mean(valid_q_values),
                'std': np.std(valid_q_values),
                'max': np.max(valid_q_values),
                'min': np.min(valid_q_values),
                'range': np.max(valid_q_values) - np.min(valid_q_values)
            })
            
            # Test action selection with low epsilon (mostly exploitation)
            action = self.trained_agent.select_action(obs, mask, epsilon=0.05)
            action_counts[action] += 1
            
            # Decode action type
            try:
                action_type, payload = decode_action(self.game, action)
                action_types[action_type] += 1
            except:
                action_types['decode_error'] += 1
        
        # Analyze results
        most_common_action = max(action_counts, key=action_counts.get)
        max_count = max(action_counts.values())
        unique_actions = len(action_counts)
        
        # Q-value statistics
        avg_q_stats = {
            'mean': np.mean([stat['mean'] for stat in q_value_stats]),
            'std': np.mean([stat['std'] for stat in q_value_stats]),
            'max': np.mean([stat['max'] for stat in q_value_stats]),
            'min': np.mean([stat['min'] for stat in q_value_stats]),
            'range': np.mean([stat['range'] for stat in q_value_stats])
        }
        
        print(f"Behavior Analysis:")
        print(f"  Most common action: {most_common_action} ({max_count}/{num_tests} times)")
        print(f"  Unique actions used: {unique_actions}")
        print(f"  Action type distribution: {dict(action_types)}")
        
        print(f"\nQ-Value Analysis:")
        print(f"  Average Q-value: {avg_q_stats['mean']:.4f}")
        print(f"  Average std dev: {avg_q_stats['std']:.4f}")
        print(f"  Average range: {avg_q_stats['range']:.4f}")
        
        # Behavior assessment
        determinism_rate = max_count / num_tests
        if determinism_rate > 0.9:
            print(f"  ⚠️  Very deterministic behavior ({determinism_rate:.1%}) - may indicate limited learning")
        elif determinism_rate > 0.7:
            print(f"  ~ Somewhat deterministic ({determinism_rate:.1%}) - reasonable for trained agent")
        else:
            print(f"  ✓ Good action diversity ({determinism_rate:.1%} max frequency)")
        
        if avg_q_stats['std'] < 0.01:
            print(f"  ⚠️  Very similar Q-values - may indicate insufficient training")
        elif avg_q_stats['std'] > 10:
            print(f"  ⚠️  Very large Q-value variance - may indicate training instability")
        else:
            print(f"  ✓ Reasonable Q-value distribution")
        
        self.evaluation_results['behavior'] = {
            'action_counts': dict(action_counts),
            'action_types': dict(action_types),
            'q_value_stats': avg_q_stats,
            'determinism_rate': determinism_rate,
            'unique_actions': unique_actions
        }
        
        return self.evaluation_results['behavior']
    
    def evaluate_game_performance(self, num_games=30):
        """Evaluate actual game performance"""
        print(f"\n=== Game Performance Evaluation ({num_games} games) ===")
        
        if not self.trained_agent:
            print("Error: No trained agent loaded")
            return None
        
        wins = 0
        eliminations = 0
        game_lengths = []
        episode_rewards = []
        
        class MockTrainer:
            def __init__(self):
                self.epsilon = 0.05  # Low exploration for evaluation
                self.replay_buffer = None
        
        for game_num in range(num_games):
            # Setup game
            rl_player = RLPlayer("EvalBot", GameRules([]), "RL")
            rl_player.agent = self.trained_agent
            rl_player.trainer = MockTrainer()
            
            opponent = EliminationBot(f"EvalOpp_{game_num}", GameRules([]), "Elimination")
            players = [rl_player, opponent]
            
            game = GameRules(players)
            for p in players:
                p.setOpponents([op for op in players if op != p])
            
            game.reset_game()
            game.dealCards()
            
            # Initialize belief matrices
            for player in players:
                player.initialCrossOff()
            
            # Play the game
            winner = None
            turn_count = 0
            max_turns = 100
            episode_reward = 0
            
            while winner is None and turn_count < max_turns:
                current_player_idx = turn_count % len(players)
                current_player = players[current_player_idx]
                
                if not current_player.inGame:
                    turn_count += 1
                    continue
                
                try:
                    if current_player.type == "RL":
                        obs = build_observation(game, current_player)
                        mask = build_action_mask(game, current_player)
                        winner = current_player.playTurn(obs, mask)
                        
                        # Track reward
                        reward = game.get_last_reward()
                        episode_reward += reward
                        
                    else:
                        winner = current_player.playTurn(None, None)
                        
                    if winner:
                        break
                        
                except Exception as e:
                    print(f"  Game {game_num+1} error: {e}")
                    break
                
                turn_count += 1
                
                # Check if RL player was eliminated
                if not rl_player.inGame and winner != rl_player.name:
                    eliminations += 1
                    break
            
            # Record results
            game_lengths.append(turn_count)
            episode_rewards.append(episode_reward)
            
            if winner == rl_player.name:
                wins += 1
            
            # Progress indicator
            if (game_num + 1) % 10 == 0:
                current_win_rate = wins / (game_num + 1)
                print(f"  Progress: {game_num + 1}/{num_games} games, current win rate: {current_win_rate:.1%}")
        
        # Calculate statistics
        win_rate = wins / num_games
        elimination_rate = eliminations / num_games
        avg_game_length = np.mean(game_lengths)
        avg_episode_reward = np.mean(episode_rewards)
        
        print(f"\nGame Performance Results:")
        print(f"  Win Rate: {wins}/{num_games} ({win_rate:.1%})")
        print(f"  Elimination Rate: {eliminations}/{num_games} ({elimination_rate:.1%})")
        print(f"  Average Game Length: {avg_game_length:.1f} turns")
        print(f"  Average Episode Reward: {avg_episode_reward:.2f}")
        
        # Performance assessment
        if win_rate >= 0.4:
            print(f"  🏆 Excellent performance - agent learned very well!")
        elif win_rate >= 0.25:
            print(f"  ✓ Good performance - solid learning achieved")
        elif win_rate >= 0.15:
            print(f"  ~ Moderate performance - some learning occurred")
        elif win_rate >= 0.05:
            print(f"  ⚠️  Weak performance - minimal learning")
        else:
            print(f"  ✗ Poor performance - little to no learning")
        
        # Additional insights
        if elimination_rate > 0.5:
            print(f"  ⚠️  High elimination rate - agent makes too many wrong accusations")
        elif elimination_rate < 0.1:
            print(f"  ✓ Low elimination rate - agent plays conservatively")
        
        if avg_game_length < 10:
            print(f"  ⚠️  Very short games - may indicate early eliminations or quick wins")
        elif avg_game_length > 50:
            print(f"  ⚠️  Very long games - may indicate overly conservative play")
        
        self.evaluation_results['performance'] = {
            'win_rate': win_rate,
            'wins': wins,
            'total_games': num_games,
            'elimination_rate': elimination_rate,
            'avg_game_length': avg_game_length,
            'avg_episode_reward': avg_episode_reward,
            'game_lengths': game_lengths,
            'episode_rewards': episode_rewards
        }
        
        return self.evaluation_results['performance']
    
    def compare_with_untrained(self):
        """Compare trained agent with fresh untrained agent"""
        print(f"\n=== Comparing Trained vs Untrained ===")
        
        if not self.trained_agent:
            print("Error: No trained agent loaded")
            return None
        
        # Create untrained agent for comparison
        input_dim = calculate_observation_size(self.game)
        untrained_agent = DQNAgent(game_rules=self.game, input_dim=input_dim)
        
        # Test both on same game states
        print("Testing action selection on 5 identical game states:")
        
        for test in range(5):
            # Setup identical game state
            rl_player = RLPlayer("TestBot", self.game, "RL")
            opponent = EliminationBot("TestOpp", self.game, "Elimination")
            players = [rl_player, opponent]
            self.game.players = players
            
            for p in players:
                p.setOpponents([op for op in players if op != p])
            
            # Use fixed seed for reproducible game state
            np.random.seed(test)
            torch.manual_seed(test)
            
            self.game.reset_game()
            self.game.dealCards()
            
            obs = build_observation(self.game, rl_player)
            mask = build_action_mask(self.game, rl_player)
            
            # Get actions from both agents
            trained_action = self.trained_agent.select_action(obs, mask, epsilon=0.05)
            untrained_action = untrained_agent.select_action(obs, mask, epsilon=0.05)
            
            # Decode actions
            try:
                trained_type, _ = decode_action(self.game, trained_action)
                untrained_type, _ = decode_action(self.game, untrained_action)
            except:
                trained_type, untrained_type = "decode_error", "decode_error"
            
            print(f"  Test {test+1}: Trained={trained_action}({trained_type}), Untrained={untrained_action}({untrained_type})")
        
        # Quick performance comparison (5 games each)
        print(f"\nQuick performance comparison (5 games each):")
        
        def quick_eval(agent, name):
            wins = 0
            for game_num in range(5):
                rl_player = RLPlayer(f"{name}Bot", GameRules([]), "RL")
                rl_player.agent = agent
                rl_player.trainer = MockTrainer()
                
                opponent = EliminationBot("QuickOpp", GameRules([]), "Elimination")
                players = [rl_player, opponent]
                
                game = GameRules(players)
                for p in players:
                    p.setOpponents([op for op in players if op != p])
                
                game.reset_game()
                game.dealCards()
                
                winner = None
                for turn in range(30):  # Quick games
                    current_player = players[turn % 2]
                    if not current_player.inGame:
                        break
                    
                    try:
                        if current_player.type == "RL":
                            obs = build_observation(game, current_player)
                            mask = build_action_mask(game, current_player)
                            winner = current_player.playTurn(obs, mask)
                        else:
                            winner = current_player.playTurn(None, None)
                            
                        if winner:
                            break
                    except:
                        break
                
                if winner == rl_player.name:
                    wins += 1
            
            return wins / 5
        
        class MockTrainer:
            def __init__(self):
                self.epsilon = 0.05
                self.replay_buffer = None
        
        trained_win_rate = quick_eval(self.trained_agent, "Trained")
        untrained_win_rate = quick_eval(untrained_agent, "Untrained")
        
        print(f"  Trained agent: {trained_win_rate:.1%} win rate")
        print(f"  Untrained agent: {untrained_win_rate:.1%} win rate")
        
        improvement = trained_win_rate - untrained_win_rate
        if improvement > 0.2:
            print(f"  ✓ Significant improvement (+{improvement:.1%})")
        elif improvement > 0:
            print(f"  ~ Modest improvement (+{improvement:.1%})")
        else:
            print(f"  ⚠️  No clear improvement ({improvement:+.1%})")
        
        return {
            'trained_win_rate': trained_win_rate,
            'untrained_win_rate': untrained_win_rate,
            'improvement': improvement
        }
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print(f"\n" + "="*50)
        print(f"COMPREHENSIVE AGENT EVALUATION REPORT")
        print(f"="*50)
        
        if not self.trained_agent:
            print("ERROR: No trained agent loaded. Run load_trained_agent() first.")
            return
        
        print(f"Model: {self.model_path}")
        print(f"Training Episodes: {self.training_info['total_episodes']}")
        print(f"Training Steps: {self.training_info['total_steps']}")
        
        # Overall assessment
        if 'performance' in self.evaluation_results:
            win_rate = self.evaluation_results['performance']['win_rate']
            if win_rate >= 0.3:
                overall_grade = "A - Excellent Learning"
            elif win_rate >= 0.2:
                overall_grade = "B - Good Learning"  
            elif win_rate >= 0.1:
                overall_grade = "C - Moderate Learning"
            elif win_rate >= 0.05:
                overall_grade = "D - Minimal Learning"
            else:
                overall_grade = "F - No Clear Learning"
            
            print(f"\nOVERALL GRADE: {overall_grade}")
            print(f"Primary Metric - Win Rate: {win_rate:.1%}")
        
        # Recommendations
        print(f"\nRECOMMENDations:")
        if 'performance' in self.evaluation_results:
            perf = self.evaluation_results['performance']
            if perf['win_rate'] < 0.1:
                print("- Consider longer training or different hyperparameters")
                print("- Check reward system and action masking")
            elif perf['elimination_rate'] > 0.5:
                print("- Agent makes too many early accusations")
                print("- Consider adjusting action masking or reward penalties")
            else:
                print("- Training appears successful")
                print("- Consider curriculum learning for further improvement")

def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained Clue agent')
    parser.add_argument('--model_path', type=str, 
                       default='models/simple_trained_clue_agent.pth',
                       help='Path to trained model file')
    parser.add_argument('--behavior_tests', type=int, default=20,
                       help='Number of behavior pattern tests')
    parser.add_argument('--performance_games', type=int, default=30,
                       help='Number of games for performance evaluation')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip comparison with untrained agent')
    
    args = parser.parse_args()
    
    print("=== Clue Agent Evaluation Tool ===\n")
    
    # Create evaluator
    evaluator = AgentEvaluator(model_path=args.model_path)
    
    # Load trained agent
    if not evaluator.load_trained_agent():
        print("Failed to load trained agent. Exiting.")
        return
    
    # Run evaluations
    print("\nRunning comprehensive evaluation...")
    
    # Behavior analysis
    evaluator.evaluate_behavior_patterns(num_tests=args.behavior_tests)
    
    # Performance evaluation
    evaluator.evaluate_game_performance(num_games=args.performance_games)
    
    # Comparison with untrained (optional)
    if not args.skip_comparison:
        evaluator.compare_with_untrained()
    
    # Generate final report
    evaluator.generate_report()
    
    print(f"\nEvaluation complete!")

if __name__ == "__main__":
    main()