import random
import Card
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
            "Suspect" : random.choice(self.suspectCards),
            "Weapon" : random.choice(self.weaponCards),
            "Room" : random.choice(self.roomCards)
        }
        
        #Take out solution cards from deck
        self.deck.pop((self.solution.get("Suspect")).getName())
        self.deck.pop((self.solution.get("Weapon")).getName())
        self.deck.pop((self.solution.get("Room")).getName())
        
        
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
    
    def dealCards(self):
        deckCards = list(self.deck.keys())
        random.shuffle(deckCards)
        playerIter =0
        for card in deckCards:
            self.players[playerIter].isDealt(card)
                