import random
class GameRules:
    ROOMS = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
    SUSPECTS = ["Colonel Mustard", "Reverend Green", "Miss Scarlet", "Mrs. Peacock", "Mrs White", "Professor Plum" ]
    WEAPONS = ["Knife", "Candlestick", "Rope", "Revolver", "Wrench", "Lead Pipe"]
    # gameboard = [[]]
    
    def __init__(self, players):
        self.players = players
        self.solution = {
            "Suspect" : random.choice(self.SUSPECTS),
            "Weapon" : random.choice(self.WEAPONS),
            "Room" : random.choice(self.ROOMS)
        }
        
    def makeAccusation(self, player, perp, weapon, room):
        if(self.solution.get("Suspect") == perp & self.solution.get("Weapon") == weapon & self.solution.get("Room") == room):
            return True
        player.inGame = False
        return False
    
    def makeSuggestion(self, player, perp, weapon, room):
        playerPos = self.players.index(player) 
        i = playerPos +1
        while(i!= playerPos):
            if(i==len(self.players)):
                i=0
            cardShown = self.players[i].hasACard(perp, weapon, room)
            if(cardShown!=None):
                return self.players[i], cardShown 
            else:
                i+=1
                
        return None, None