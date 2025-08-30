# test_basic_rl.py
from ClueBasics.GameRules import GameRules
from agents import EliminationBot
from rl.RLPlayer import RLPlayer
from rl.DQNAgent import DQNAgent
from rl.replay import ReplayBuffer
from rl.utils import sizes_from_game

# Test with just RL player vs EliminationBot
game = GameRules([])
S, W, R, C, P = sizes_from_game(game)
input_dim = (P + 1) * C + 1

agent = DQNAgent(game_rules=game, input_dim=input_dim)
rl_player = RLPlayer("TestBot", game, "RL")
rl_player.agent = agent

players = [
    rl_player,
    EliminationBot("Opponent", game, "Elimination")
]

# Try to run one game
game.players = players
for p in players:
    p.setOpponents([op for op in players if op != p])
game.reset_game()
game.dealCards()

print("If this prints, basic setup works!")