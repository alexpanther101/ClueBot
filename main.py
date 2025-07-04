# main.py
from ClueBasics import GameRules, Player

if __name__ == "__main__":
    player_names = ["Alice", "Bob", "Charlie"]
    game = GameRules(players=[])
    dummy_players = [Player(name, game, []) for name in player_names]
   
    
    # Now update references
    for p in dummy_players:
            p.setOpponents([op for op in dummy_players if op != p])
    
    game.players = dummy_players
    game.gameLoop()
