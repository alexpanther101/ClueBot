#action mapping, state builder, masking helpers

"""
rl/utils.py — action mapping, state builder, and masking helpers

This module centralizes:
  • Action indexing for Suggest, Accuse, and Reveal
  • Observation/state vector construction (belief + light context)
  • Valid-action masks for the agent

Assumptions about your codebase (kept minimal and backwards-compatible):
  • GameRules exposes SUSPECTS, WEAPONS, ROOMS lists.
  • GameRules has attributes: players (list), suggestionLog (list of dicts), gameTurn (int).
  • Player exposes: numCards (int), inGame (bool), get_flattened_belief() -> 1D np.ndarray of length 105 (5x21) or similar.
  • Cards are identified by their name string matching entries in GameRules lists.

If some of these differ in your code, adjust the small adapter functions at the bottom.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math
from collections import deque

# -------------------------------
# Constants and derived sizes
# -------------------------------

def sizes_from_game(game) -> Tuple[int, int, int, int, int]:
    """Return (S, W, R, C, P) = (#suspects, #weapons, #rooms, #cards, #players)."""
    S = len(game.SUSPECTS)
    W = len(game.WEAPONS)
    R = len(game.ROOMS)
    C = len(game.cards)
    P = len(game.players)
    return S, W, R, C, P

# -------------------------------
# Observation/State vector
# -------------------------------

def build_observation(game, player):
    """
    Builds a comprehensive observation vector for the RL agent.
    
    The observation includes:
    1. Flattened belief matrix (probabilities for all cards across all players).
    2. Suggestion log history.
    3. The current game turn.
    """
    S, W, R, C, P = sizes_from_game(game)
    
    # 1. Player's belief matrix (belief space)
    belief_obs = player.get_flattened_belief()
    
    # 2. Suggestion log history (last 10 suggestions, one-hot encoded)
    # This adds crucial information about what has been asked and who has responded.
    suggestion_log_len = len(game.suggestionLog)
    suggestion_obs = np.zeros(10 * (P + 1 + 3 + C), dtype=np.float32)
    # A simple one-hot encoding for the last 10 suggestions
    # You could also use a more complex representation, e.g., a mini-convnet
    # For now, we'll just one-hot encode the suggester, responder, and the three suggested cards
    # Format: [suggester_idx, responder_idx, card1_idx, card2_idx, card3_idx]
    
    # NOTE: The one-hot encoding logic here would need to be implemented
    # based on your Player and Card objects' properties. For simplicity,
    # let's assume a basic encoding.
    
    # 3. Game turn (turn number)
    game_turn_obs = np.array([game.turn], dtype=np.float32)
    
    # Combine all observation components
    # The suggestion log encoding is a placeholder for now
    # We will simply add the game turn for a basic implementation.
    obs = np.concatenate([belief_obs, game_turn_obs])
    
    return obs

# -------------------------------
# Action Space and Masking
# -------------------------------
# These functions map between a flat integer action space and structured
# game actions (suggestion, accusation, reveal).

# Total actions = (S*W*R suggestions) + (S*W*R accusations) + (C reveals)
# S, W, R = #suspects, weapons, rooms
# C = total #cards

def build_action_mask(game, player):
    """Creates a boolean mask for all possible actions."""
    S, W, R, C, P = sizes_from_game(game)
    num_suggestions = S * W * R
    num_accusations = S * W * R
    num_reveals = C
    total_actions = num_suggestions + num_accusations + num_reveals

    mask = np.zeros(total_actions, dtype=bool)
    
    # Suggestions are always valid
    mask[0:num_suggestions] = True

    # Allow accusations but with probability based on confidence
    # Let the agent learn when to accuse through rewards
    mask[num_suggestions:num_suggestions + num_accusations] = True
    
    return mask

def should_accuse(self):
    """
    A simple heuristic for deciding when to allow an accusation.
    This should be based on the belief matrix certainty.
    """
    # Placeholder: a more complex check would be needed here.
    # For now, let's say a player is "confident" if they have a
    # solution in their belief matrix with a high probability.
    
    # A simple check: if the product of the highest probabilities for each
    # category (suspect, weapon, room) is above a certain threshold, accuse.
    suspect_prob = max(self.getProbability("Solution", card) for card in self.game.suspectCards.values())
    weapon_prob = max(self.getProbability("Solution", card) for card in self.game.weaponCards.values())
    room_prob = max(self.getProbability("Solution", card) for card in self.game.roomCards.values())
    
    # Threshold for accusation: can be tuned
    return suspect_prob * weapon_prob * room_prob > 0.95

# -------------------------------
# Action Indexing
# -------------------------------

def suggest_index_from_swr(game, s_idx: int, w_idx: int, r_idx: int) -> int:
    S, W, R, _, _ = sizes_from_game(game)
    return s_idx * W * R + w_idx * R + r_idx

def accuse_index_from_swr(game, s_idx: int, w_idx: int, r_idx: int) -> int:
    S, W, R, _, _ = sizes_from_game(game)
    num_suggestions = S * W * R
    return num_suggestions + s_idx * W * R + w_idx * R + r_idx

def reveal_index_from_card(game, card) -> int:
    S, W, R, C, _ = sizes_from_game(game)
    num_suggestions = S * W * R
    num_accusations = S * W * R
    
    if hasattr(card, 'global_index'):
        return num_suggestions + num_accusations + card.global_index
    else:
        # Fallback for older card objects. Assign a temporary global index.
        # It's better to ensure Card objects have a global_index attribute.
        return num_suggestions + num_accusations + game.cards.index(card)

def decode_action(game, action_idx: int) -> Tuple[str, Any]:
    S, W, R, C, P = sizes_from_game(game)
    SWR = S * W * R
    
    if action_idx < SWR:
        # It's a suggestion
        payload = decode_swr_index(game, action_idx)
        return "suggest", payload
    elif action_idx < SWR * 2:
        # It's an accusation
        payload = decode_swr_index(game, action_idx - SWR)
        return "accuse", payload
    elif action_idx < SWR * 2 + C:
        # It's a reveal action
        card_idx = action_idx - (SWR * 2)
        return "reveal", card_idx
    else:
        raise ValueError(f"Invalid action index: {action_idx}")
        
def decode_swr_index(game, index: int) -> Tuple[int, int, int]:
    S, W, R, _, _ = sizes_from_game(game)
    r_idx = index % R
    index //= R
    w_idx = index % W
    index //= W
    s_idx = index % S
    return s_idx, w_idx, r_idx

def get_card_from_action_idx(game, action_idx):
    """Decodes a reveal action index back into a Card object."""
    S, W, R, C, P = sizes_from_game(game)
    num_suggestions = S * W * R
    num_accusations = S * W * R
    
    card_idx = action_idx - (num_suggestions + num_accusations)
    return game.cards[card_idx]