import time 
import random
from .Card import Card
class GameRules:
    ROOMS = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
    SUSPECTS = ["Colonel Mustard", "Reverend Green", "Miss Scarlet", "Mrs. Peacock", "Mrs White", "Professor Plum" ]
    WEAPONS = ["Knife", "Candlestick", "Rope", "Revolver", "Wrench", "Lead Pipe"]
    # gameboard = [[]]
    
    def __init__(self, players):
        self.players = players
        self.suspectCards = {}
        self.weaponCards = {}
        self.roomCards = {}
        self.deck = {}
        self.suggestionLog = []
        for card in self.SUSPECTS:
            suspect = Card("Suspect", card)
            self.suspectCards[card] = suspect
            self.deck[card] = suspect
            
        for card in self.WEAPONS:
            weapon = Card("Weapon", card) 
            self.weaponCards[card] = (weapon)  
            self.deck[card] = weapon
        for card in self.ROOMS:
            room = Card("Room", card)
            self.roomCards[card] = (room)
            self.deck[card] = room
                    
        self.solution = {
            "Suspect" : random.choice(list(self.suspectCards.values())),
            "Weapon" : random.choice(list(self.weaponCards.values())),
            "Room" : random.choice(list(self.roomCards.values()))
        }
        
        #Take out solution cards from deck
        self.deck.pop((self.solution.get("Suspect")).getName())
        self.deck.pop((self.solution.get("Weapon")).getName())
        self.deck.pop((self.solution.get("Room")).getName())
        
        
    def makeAccusation(self, player, perp, weapon, room):
        print((player.name) + " accuses " + perp.name + " with a " + weapon.name + " in the "+room.name)
        if(self.solution.get("Suspect") == perp and self.solution.get("Weapon") == weapon and self.solution.get("Room") == room):
            return True
        player.inGame = False
        return False
    
    def makeSuggestion(self, player, perp, weapon, room):
        print(f"{player.name} suggests: {perp} with {weapon} in {room}")
        self.suggestionLog.append(player.name + " " + perp.name + " " + weapon.name + " " + room.name)
        
        suggestionCards = []
        suggestionCards.append(perp)
        suggestionCards.append(weapon)
        suggestionCards.append(room)
         
        playerPos = self.players.index(player) 
        i = playerPos +1
        if(i==len(self.players)):
                i =0
        while(i!= playerPos):
            print(self.players[i], end = " ")
            print("is checking their hand")
            
            cardShown = self.players[i].refuteSuggestion(suggestionCards)
            if(cardShown!=None):
                return self.players[i], cardShown 
            else:
                i+=1
            if(i==len(self.players)):
                i=0      
        return None, None
    
    def dealCards(self):
        deckCards = list(self.deck.keys())
        random.shuffle(deckCards)
        playerIter =0
        for card in deckCards:
            self.players[playerIter].isDealt(self.deck.get(card))
            playerIter+=1
            if(playerIter==len(self.players)):
                playerIter = 0
        
        for player in self.players:
            print(player.name + " has " + str(player.numCards)+" cards")
            if player.type == "Human":
                    player.revealCards()
                       
    def checkAllPlayers(self):
        count =0
        for player in self.players:
            if(player.inGame):
                count+=1
            if(count > 1):
                return True
        return False
    
    def findWinner(self):
        for player in self.players:
            if(player.inGame):
                return player
            
    def gameLoop(self):
        self.dealCards()
        while(True):
            if not (self.checkAllPlayers()):
                break
            
            for player in self.players:
                winner = player.playTurn()
                if(winner):
                    return
                
                    
    
    
  