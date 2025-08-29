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

# -------------------------------
# Constants and derived sizes
# -------------------------------

def sizes_from_game(game) -> Tuple[int, int, int, int, int]:
    """Return (S, W, R, C, P) = (#suspects, #weapons, #rooms, #cards, #players)."""
    S = len(game.SUSPECTS)
    W = len(game.WEAPONS)
    R = len(game.ROOMS)
    C = S + W + R
    return S, W, R, C, len(game.players)


def action_space_size(game) -> int:
    """Total number of discrete actions = SUGGEST (S*W*R) + ACCUSE (S*W*R) + REVEAL (C)."""
    S, W, R, C, P = sizes_from_game(game)
    return (S * W * R) * 2 + C


# -------------------------------
# Card indexing helpers (global index 0..C-1)
# -------------------------------

def card_name_lists(game) -> Tuple[List[str], List[str], List[str]]:
    return game.SUSPECTS, game.WEAPONS, game.ROOMS


def card_to_global_index(game, card_name: str) -> int:
    SUS, WEP, RMS = card_name_lists(game)
    if card_name in SUS:
        return SUS.index(card_name)
    off = len(SUS)
    if card_name in WEP:
        return off + WEP.index(card_name)
    off += len(WEP)
    if card_name in RMS:
        return off + RMS.index(card_name)
    raise ValueError(f"Unknown card name: {card_name}")


def global_index_to_card(game, idx: int) -> str:
    SUS, WEP, RMS = card_name_lists(game)
    S, W, R, C, P = sizes_from_game(game)
    if idx < 0 or idx >= C:
        raise IndexError(f"Card index out of range: {idx}")
    if idx < S:
        return SUS[idx]
    idx -= S
    if idx < W:
        return WEP[idx]
    idx -= W
    return RMS[idx]


# -------------------------------
# Action indexing for Suggest / Accuse / Reveal
# -------------------------------

def suggest_index_from_swr(game, s_idx: int, w_idx: int, r_idx: int) -> int:
    """Map (suspect, weapon, room) indices to Suggest action index in [0, S*W*R)."""
    S, W, R, C, P = sizes_from_game(game)
    assert 0 <= s_idx < S and 0 <= w_idx < W and 0 <= r_idx < R
    #Think of it as a 3 digit number - you need the ones place digit + (tens place dimensions) * tens place digit + (hundreds place dimensions) * hundreds place digit
    return (s_idx * W + w_idx) * R + r_idx


def accuse_index_from_swr(game, s_idx: int, w_idx: int, r_idx: int) -> int:
    """Map (s,w,r) to Accuse action index in [SWR, 2*SWR)."""
    S, W, R, C, P = sizes_from_game(game)
    SWR = S * W * R
    return SWR + suggest_index_from_swr(game, s_idx, w_idx, r_idx)


def reveal_index_from_card_idx(game, card_idx: int) -> int:
    """Map global card index to Reveal action index in [2*SWR, 2*SWR + C)."""
    S, W, R, C, P = sizes_from_game(game)
    SWR = S * W * R
    assert 0 <= card_idx < C
    return 2 * SWR + card_idx


def decode_action(game, action_idx: int) -> Tuple[str, Tuple[int, int, int] | int]:
    """Decode action index to a tuple.
    Returns:
      ("suggest", (s_idx, w_idx, r_idx))
      ("accuse",  (s_idx, w_idx, r_idx))
      ("reveal",  card_idx)
    """
    S, W, R, C, P = sizes_from_game(game)
    SWR = S * W * R
    if 0 <= action_idx < SWR:
        # Suggest
        x = action_idx
        r_idx = x % R; x //= R
        w_idx = x % W; x //= W
        s_idx = x
        return "suggest", (s_idx, w_idx, r_idx)
    if SWR <= action_idx < 2 * SWR:
        # Accuse
        a = action_idx - SWR
        r_idx = a % R; a //= R
        w_idx = a % W; a //= W
        s_idx = a
        return "accuse", (s_idx, w_idx, r_idx)
    base = 2 * SWR
    if base <= action_idx < base + C:
        return "reveal", action_idx - base
    raise IndexError(f"action_idx out of range: {action_idx}")


# -------------------------------
# Observation / state construction
# -------------------------------

def _safe_flatten_belief(player) -> np.ndarray:
    """Adapter: tries player.get_flattened_belief(), else flattens player.belief."""
    if hasattr(player, "get_flattened_belief"):
        arr = player.get_flattened_belief()
        return np.asarray(arr, dtype=np.float32)
    # Fallbacks
    if hasattr(player, "belief"):
        return np.asarray(player.belief, dtype=np.float32).ravel()
    raise AttributeError("Player must expose get_flattened_belief() or .belief array")


def _solution_entropy_from_belief(flat_belief: np.ndarray, game) -> float:
    """Estimate entropy over the Solution column if belief encodes it as a fifth 'owner'.
    Expected belief layout (owner x card) flattened; if not available, return 0.0.
    """
    S, W, R, C , P = sizes_from_game(game)
    owners = P + 1  # 4 players + Solution; adjust if your game differs
    expected = owners * C
    if flat_belief.size != expected:
        return 0.0
    # Take the last owner row as Solution distribution
    sol = flat_belief[(owners - 1) * C : owners * C]
    p = np.clip(sol.astype(np.float64), 1e-9, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _normalize_scalar(x: float, max_x: float) -> float:
    if max_x <= 0:
        return 0.0
    return float(np.clip(x / max_x, 0.0, 1.0))


def build_observation(game, player, *, include_context: bool = True) -> np.ndarray:
    """Construct the model input vector (np.float32).

    Components:
      • Flattened belief matrix (owners × cards)
      • Optional lightweight context scalars
    """
    belief = _safe_flatten_belief(player)
    if not include_context:
        return belief.astype(np.float32)

    # Lightweight contextual features
    turn_norm = _normalize_scalar(getattr(game, "gameTurn", 0), 200.0)  # heuristic cap

    # Estimate max cards per player after removing solution cards
    S, W, R, C = sizes_from_game(game)
    total_dealt = C - 3  # 3 solution cards removed
    n_players = max(1, len(getattr(game, "players", [])))
    max_cards = math.ceil(total_dealt / n_players)
    cards_norm = _normalize_scalar(getattr(player, "numCards", 0), max_cards)

    entropy_solution = _solution_entropy_from_belief(belief, game)
    entropy_solution = float(entropy_solution) / math.log(max(C, 2))  # normalize by log(|cards|)

    # last refuter id (or -1 if none), normalized to [0,1]
    last_refuter_norm = 0.0
    log = getattr(game, "suggestionLog", [])
    if log:
        last = log[-1]
        responder = last.get("responder")
        if responder is not None and responder in game.players:
            ridx = game.players.index(responder)
            last_refuter_norm = _normalize_scalar(ridx, max(1, n_players - 1))

    ctx = np.array([turn_norm, cards_norm, entropy_solution, last_refuter_norm], dtype=np.float32)
    return np.concatenate([belief.astype(np.float32), ctx], axis=0)


# -------------------------------
# Valid action masks
# -------------------------------

def build_action_mask(game, player, *, phase: Optional[str] = None, matching_cards: Optional[List[str]] = None) -> np.ndarray:
    """Build a boolean mask over the full action space.

    Args:
      phase: one of {None, "turn", "reveal"}.
        • None/"turn": mask Suggest/Accuse; Reveal actions off.
        • "reveal": only Reveal actions for the provided matching_cards are enabled.
      matching_cards: list of card names that this player is allowed to reveal (used in phase="reveal").

    Returns:
      mask: np.ndarray(bool) of shape [ACTION_SPACE]
    """
    S, W, R, C = sizes_from_game(game)
    SWR = S * W * R
    A = 2 * SWR + C
    mask = np.zeros(A, dtype=bool)

    if phase == "reveal":
        # Enable only matching reveal cards
        if not matching_cards:
            return mask  # all false → no valid reveal (should not happen)
        for name in matching_cards:
            cidx = card_to_global_index(game, name)
            mask[reveal_index_from_card_idx(game, cidx)] = True
        return mask

    # Default: player's normal turn (Suggest + Accuse enabled)
    # Suggest: enable all combinations (your game has no movement constraints)
    mask[0:SWR] = True

    # Accuse: decide when accusing is allowed
    allow_accuse_anytime = getattr(game, "ALLOW_ACCUSE_ANYTIME", False)
    can_accuse = allow_accuse_anytime and getattr(player, "inGame", True)
    if can_accuse:
        mask[SWR:2*SWR] = True

    # Reveal actions are only valid during reveal phase, keep them False here
    return mask


# -------------------------------
# Convenience helpers for Player/RLPlayer
# -------------------------------

def encode_suggest_by_names(game, suspect: str, weapon: str, room: str) -> int:
    SUS, WEP, RMS = card_name_lists(game)
    s = SUS.index(suspect); w = WEP.index(weapon); r = RMS.index(room)
    return suggest_index_from_swr(game, s, w, r)


def encode_accuse_by_names(game, suspect: str, weapon: str, room: str) -> int:
    SUS, WEP, RMS = card_name_lists(game)
    s = SUS.index(suspect); w = WEP.index(weapon); r = RMS.index(room)
    return accuse_index_from_swr(game, s, w, r)


def decode_reveal_action(game, action_idx: int) -> str:
    kind, payload = decode_action(game, action_idx)
    if kind != "reveal":
        raise ValueError("decode_reveal_action called on non-reveal action")
    card_idx = int(payload)
    return global_index_to_card(game, card_idx)


# -------------------------------
# Debug / validation utilities
# -------------------------------

def validate_action_indexing(game):
    S, W, R, C = sizes_from_game(game)
    SWR = S * W * R
    # Round-trip checks for a few samples
    triples = [(0,0,0), (S-1, W-1, R-1), (S//2, W//2, R//2)]
    for s,w,r in triples:
        a = suggest_index_from_swr(game, s,w,r)
        kind, (s2,w2,r2) = decode_action(game, a)
        assert kind == "suggest" and (s,w,r) == (s2,w2,r2)
        b = accuse_index_from_swr(game, s,w,r)
        kind2, (s3,w3,r3) = decode_action(game, b)
        assert kind2 == "accuse" and (s,w,r) == (s3,w3,r3)
    for c in [0, C-1, C//2]:
        ridx = reveal_index_from_card_idx(game, c)
        kind3, c2 = decode_action(game, ridx)
        assert kind3 == "reveal" and c == c2
    assert action_space_size(game) == 2*SWR + C


def pretty_print_mask(mask: np.ndarray, game) -> Dict[str, Any]:
    """Summarize how many actions are enabled in each block for quick debugging."""
    S, W, R, C = sizes_from_game(game)
    SWR = S * W * R
    d = {
        "suggest_enabled": int(mask[0:SWR].sum()),
        "accuse_enabled": int(mask[SWR:2*SWR].sum()),
        "reveal_enabled": int(mask[2*SWR:2*SWR+C].sum()),
        "total": int(mask.sum()),
    }
    return d
