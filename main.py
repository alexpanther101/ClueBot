import sys
import os
import torch
import random
import numpy as np

# Add parent directory to path to find all modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ClueBasics.GameRules import GameRules
from agents import BayesianLearner, TriggerHappyBot, EliminationBot, HumanPlayer
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.utils import build_observation, build_action_mask, sizes_from_game

def load_rl_agent(checkpoint_path, game_rules, device):
    """
    Loads a pre-trained DQNAgent from a checkpoint file.
    """
    # Assuming input_dim is based on the belief matrix + game turn
    S, W, R, C, P = sizes_from_game(game_rules)
    input_dim = (P + 1) * C + 1  # 1 for belief + turn number

    agent = DQNAgent(
        game_rules=game_rules,
        input_dim=input_dim,
        device=device
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.q_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
        print(f"Successfully loaded RL agent from {checkpoint_path}")
        return agent
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def run_game_loop(players_list,game, rl_agent=None):
    """
    Runs a single game of Clue with the given list of players.
    """
    game.players = players_list
    for p in players_list:
            p.setOpponents([op for op in players_list if op != p])
    # Set up the game
    game.reset_game()
    game.dealCards()
    
    # Let each player perform their initial logic
    for player in players_list:
        player.initialCrossOff()

    print("\nGame has started! Solution is hidden.")
    print("------------------------------------------")

    winner = None
    turn_count = 0

    while winner is None and turn_count < 200: # Add a turn limit to prevent infinite loops
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")
        
        current_player_idx = (game.turn) % len(players_list)
        current_player = players_list[current_player_idx]

        if not current_player.inGame:
            print(f"{current_player.name} is out of the game.")
            game.turn += 1
            continue

        print(f"It is {current_player.name}'s turn.")
        
        # --- Handle different player types ---
        if current_player.type == "RL":
            # For the RL bot, we need to provide the full observation and valid mask
            obs = build_observation(game, current_player)
            valid_mask = build_action_mask(game, current_player)
            winner = current_player.playTurn(obs, valid_mask)
        else:
            # For other bots, they can handle their own logic internally.
            # We provide a dummy observation and mask for compatibility.
            dummy_obs = [0]
            dummy_mask = [True]
            winner = current_player.playTurn(dummy_obs, dummy_mask)
            
        game.turn += 1
        
    # Game ends
    print("\n------------------------------------------")
    if winner:
        print(f"Game over! The winner is {winner}.")
        return winner
    else:
        print("Game over! No winner found within turn limit.")
        return None

if __name__ == "__main__":
    # --- Configuration ---
    # Set the path to your trained RL model
    MODEL_CHECKPOINT_PATH = "path/to/your/rl_model_checkpoint.pth" 
    
    # Make sure a checkpoint file exists before running
    # You will need to train your agent first to generate this file
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
       print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
       print("Please train your RL agent first to create the checkpoint.")
       sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Create players for the game ---
    # Instantiate the RL player
    rl_player = RLPlayer("Cluebot_RL", None, "RL")
    
    # Load the trained model into the RL player
    trained_agent = load_rl_agent(MODEL_CHECKPOINT_PATH, None, device) # Pass None for game rules, since the agent doesn't need it for loading
    rl_player.agent = trained_agent
    game = GameRules(players=[])
    # Instantiate other bots to play against the RL agent
    players = [
        #rl_player,
        BayesianLearner("Bayesian_A", game, "Bayesian"),
        BayesianLearner("Bayesian_B", game, "Bayesian"), 
        EliminationBot("ElimA", game, "Elimination")
    ]
    # --- Run the game ---
    run_game_loop(players, game)