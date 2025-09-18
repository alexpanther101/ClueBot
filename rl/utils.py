"""
Improved rl/utils.py with better action mapping and masking
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math
from collections import deque

def sizes_from_game(game) -> Tuple[int, int, int, int, int]:
    """Return (S, W, R, C, P) = (#suspects, #weapons, #rooms, #cards, #players)."""
    S = len(game.SUSPECTS)
    W = len(game.WEAPONS)
    R = len(game.ROOMS)
    C = len(game.cards)
    P = len(game.players)
    return S, W, R, C, P

def build_observation(game, player):
    """
    Builds a comprehensive observation vector for the RL agent.
    """
    S, W, R, C, P = sizes_from_game(game)
    
    # 1. Belief Matrix - Core observation
    belief_obs = player.get_flattened_belief()
    
    # 2. Suggestion History (last 10 suggestions)
    max_history = 10
    suggestion_features = []
    
    recent_suggestions = list(game.suggestionLog)[-max_history:]
    
    for i in range(max_history):
        if i < len(recent_suggestions):
            rec = recent_suggestions[i]
            
            # Encode suggester (one-hot)
            suggester_encoding = np.zeros(P)
            suggester_idx = game.players.index(rec['suggester'])
            suggester_encoding[suggester_idx] = 1
            
            # Encode responder (one-hot + no responder option)
            responder_encoding = np.zeros(P + 1)
            if rec['responder']:
                responder_idx = game.players.index(rec['responder'])
                responder_encoding[responder_idx] = 1
            else:
                responder_encoding[-1] = 1  # No responder
            
            # Encode suggested cards (multi-hot)
            cards_encoding = np.zeros(C)
            for card in rec['suggestion']:
                card_idx = card.global_index
                cards_encoding[card_idx] = 1
            
            # Encode whether card was shown to this player
            card_shown_to_me = 0
            if rec['card_shown'] and (player == rec['suggester'] or player == rec['responder']):
                card_shown_to_me = 1
            
            # Encode skipped players
            skipped_encoding = np.zeros(P)
            for skip_player in rec.get('skipped_players', []):
                skip_idx = game.players.index(skip_player)
                skipped_encoding[skip_idx] = 1
            
            suggestion_features.extend(suggester_encoding)
            suggestion_features.extend(responder_encoding)
            suggestion_features.extend(cards_encoding)
            suggestion_features.append(card_shown_to_me)
            suggestion_features.extend(skipped_encoding)
        else:
            # Pad with zeros for missing history
            feature_size = P + (P + 1) + C + 1 + P
            suggestion_features.extend([0] * feature_size)
    
    suggestion_obs = np.array(suggestion_features, dtype=np.float32)
    
    # 3. Game State Features
    game_features = []
    
    # Normalized turn counter
    game_features.append(game.turn / 200.0)
    
    # Players still in game (binary)
    for p in game.players:
        game_features.append(1.0 if p.inGame else 0.0)
    
    # Number of cards each player has (normalized)
    for p in game.players:
        game_features.append(p.numCards / 7.0)  # Max ~7 cards per player
    
    # Current player indicator
    current_player_encoding = np.zeros(P)
    current_idx = game.players.index(player)
    current_player_encoding[current_idx] = 1
    game_features.extend(current_player_encoding)
    
    game_obs = np.array(game_features, dtype=np.float32)
    
    # 4. Player-specific features
    player_features = []
    
    # Confidence scores for each category
    if len(player.possibleSuspects) > 0:
        suspect_confidence = max(player.getProbability("Solution", card) 
                               for card in player.possibleSuspects)
    else:
        suspect_confidence = 0
    
    if len(player.possibleWeapons) > 0:
        weapon_confidence = max(player.getProbability("Solution", card) 
                              for card in player.possibleWeapons)
    else:
        weapon_confidence = 0
        
    if len(player.possibleRooms) > 0:
        room_confidence = max(player.getProbability("Solution", card) 
                            for card in player.possibleRooms)
    else:
        room_confidence = 0
    
    player_features.extend([suspect_confidence, weapon_confidence, room_confidence])
    
    # Number of possibilities remaining per category (normalized)
    player_features.append(len(player.possibleSuspects) / S)
    player_features.append(len(player.possibleWeapons) / W)
    player_features.append(len(player.possibleRooms) / R)
    
    player_obs = np.array(player_features, dtype=np.float32)
    
    # Combine all observations
    obs = np.concatenate([belief_obs, suggestion_obs, game_obs, player_obs])
    
    return obs

def calculate_observation_size(game):
    """Calculate the total observation vector size"""
    S, W, R, C, P = sizes_from_game(game)
    
    # Belief matrix: (P + 1) * C
    belief_size = (P + 1) * C
    
    # Suggestion history: 10 * (P + P+1 + C + 1 + P)
    suggestion_size = 10 * (P + (P + 1) + C + 1 + P)
    
    # Game state: 1 + P + P + P
    game_state_size = 1 + P + P + P
    
    # Player features: 3 + 3
    player_size = 6
    
    total_size = belief_size + suggestion_size + game_state_size + player_size
    return total_size

def build_action_mask(game, player):
    """
    Creates an intelligent boolean mask for all possible actions.
    FIXED: Only includes suggestions and accusations (no reveal actions for regular play)
    """
    S, W, R, C, P = sizes_from_game(game)
    num_suggestions = S * W * R
    num_accusations = S * W * R
    # FIXED: Remove reveal actions from regular play action space
    total_actions = num_suggestions + num_accusations

    mask = np.zeros(total_actions, dtype=bool)
    print(f"Created action mask with size: {total_actions} (suggestions: {num_suggestions}, accusations: {num_accusations})")
    
    # 1. Suggestion Masking - Be more selective
    mask = apply_suggestion_masking(game, player, mask, num_suggestions)
    
    # 2. Accusation Masking - Be very restrictive
    mask = apply_accusation_masking(game, player, mask, num_suggestions, num_accusations)
    
    # 3. Reveal actions are NOT part of regular player turn actions
    # They are handled separately when responding to suggestions
    
    return mask

def apply_suggestion_masking(game, player, mask, num_suggestions):
    """Apply intelligent suggestion masking"""
    S, W, R = len(game.SUSPECTS), len(game.WEAPONS), len(game.ROOMS)
    
    # Start with all suggestions allowed
    mask[0:num_suggestions] = True
    
    # Block suggestions about cards we know players have
    for i in range(num_suggestions):
        s_idx, w_idx, r_idx = decode_swr_index(game, i)
        
        suspect = game.suspectCards[game.SUSPECTS[s_idx]]
        weapon = game.weaponCards[game.WEAPONS[w_idx]]
        room = game.roomCards[game.ROOMS[r_idx]]
        
        # Count how many cards we already know about
        known_count = 0
        for card in [suspect, weapon, room]:
            for owner in player.owners:
                if owner != "Solution" and player.getProbability(owner, card) >= 0.99:
                    known_count += 1
                    break
        
        # Block suggestions where we know about too many cards
        if known_count >= 2:
            mask[i] = False
            
        # Block suggestions where all cards have very low solution probability
        solution_probs = [player.getProbability("Solution", card) 
                         for card in [suspect, weapon, room]]
        if all(p < 0.05 for p in solution_probs):
            mask[i] = False
            
        # Early game: prefer high-entropy cards
        if game.turn < 15:
            total_entropy = sum(calculate_card_entropy(player, card) 
                              for card in [suspect, weapon, room])
            if total_entropy < 3.0:  # Low information potential
                if np.random.random() < 0.8:  # Probabilistically block
                    mask[i] = False
    
    # Ensure at least some suggestions remain valid
    if not np.any(mask[0:num_suggestions]):
        # Emergency fallback: allow some high-entropy suggestions
        for i in range(min(20, num_suggestions)):
            mask[i] = True
    
    return mask

def apply_accusation_masking(game, player, mask, num_suggestions, num_accusations):
    """Apply very restrictive accusation masking"""
    # Default: block all accusations
    mask[num_suggestions:num_suggestions + num_accusations] = False
    
    # Only allow accusations under specific conditions
    if player.should_accuse():
        # Calculate the best accusation candidates
        best_accusations = get_best_accusation_candidates(game, player)
        
        for s_idx, w_idx, r_idx in best_accusations:
            acc_idx = accuse_index_from_swr(game, s_idx, w_idx, r_idx)
            if acc_idx < len(mask):
                mask[acc_idx] = True
    
    return mask

def get_best_accusation_candidates(game, player, max_candidates=5):
    """Get the most promising accusation combinations"""
    S, W, R = len(game.SUSPECTS), len(game.WEAPONS), len(game.ROOMS)
    
    candidates = []
    
    # Get highest probability cards from each category
    best_suspects = get_top_solution_candidates(player.possibleSuspects, player, 2)
    best_weapons = get_top_solution_candidates(player.possibleWeapons, player, 2)
    best_rooms = get_top_solution_candidates(player.possibleRooms, player, 2)
    
    # Generate combinations of top candidates
    for suspect in best_suspects:
        for weapon in best_weapons:
            for room in best_rooms:
                s_idx = game.SUSPECTS.index(suspect.name)
                w_idx = game.WEAPONS.index(weapon.name)
                r_idx = game.ROOMS.index(room.name)
                
                # Calculate joint probability
                joint_prob = (player.getProbability("Solution", suspect) *
                             player.getProbability("Solution", weapon) *
                             player.getProbability("Solution", room))
                
                candidates.append((joint_prob, s_idx, w_idx, r_idx))
    
    # Sort by probability and return top candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [(s, w, r) for _, s, w, r in candidates[:max_candidates]]

def get_top_solution_candidates(card_list, player, max_count):
    """Get the cards with highest solution probability from a category"""
    if not card_list:
        return []
    
    card_probs = [(player.getProbability("Solution", card), card) for card in card_list]
    card_probs.sort(reverse=True, key=lambda x: x[0])
    
    return [card for _, card in card_probs[:max_count]]

def calculate_card_entropy(player, card):
    """Calculate entropy for a specific card"""
    entropy = 0.0
    for owner in player.owners:
        prob = player.getProbability(owner, card)
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

# Action indexing functions (unchanged from original)
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