# main.py
from ClueBasics import GameRules, Player
from bots import TriggerHappyBot, EliminationBot
#testing commitsß
if __name__ == "__main__":
    trigger_player_names = ["Trigger1", "Trigger2"]
    elim_player_names = ["Elim1" ,"Elim2"]
    
    game = GameRules(players=[])
    dummy_players = [TriggerHappyBot(name, game) for name in trigger_player_names]
    
    for name in elim_player_names:
            dummy_players.append(EliminationBot(name, game))
             
    # Now update references
    for p in dummy_players:
            p.setOpponents([op for op in dummy_players if op != p])
    
    game.players = dummy_players
    game.gameLoop()
