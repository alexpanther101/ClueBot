from .GameRules import GameRules
import random
from .Card import Card
class Player:
    
    def __init__(self, name, game):
        self.name = name
        self.game = game
        self.possibleSuspects = list(game.suspectCards.values())
        self.possibleWeapons = list(game.weaponCards.values())
        self.possibleRooms = list(game.roomCards.values())
        self.inGame = True
        self.cards = []
        self.numCards = 0
        
    def __str__(self):
        return f"Player: {self.name}"
    
    def __repr__(self):
        return str(self)
        
    def getNumCards(self):
        return self.numCards
    
    def setOpponents(self, opponents):
        self.opponents = opponents
        self.ownersAndCards = dict.fromkeys(opponents)
        cardMatrixLen = len(self.possibleRooms)+len(self.possibleSuspects)+len(self.possibleWeapons)
        
        #Create belief matrix for each opponent - mapping each owner's card to a belief 
        for opponent in self.opponents: 
            self.ownersAndCards[opponent] = {}    
        
        
            for card in self.possibleSuspects:
                self.ownersAndCards[opponent][card] = 1/cardMatrixLen
            
            for card in self.possibleWeapons:
                self.ownersAndCards[opponent][card] = 1/cardMatrixLen
                
            for card in self.possibleRooms:
                self.ownersAndCards[opponent][card] = 1/cardMatrixLen
                #need to update that probability when cards start to fill up
        
        self.ownersAndCards[self] = {}
    
    #Removes cards from possible solutions and maps to owner
    def crossOffMulti(self, owner, cardList):
        for card in cardList:
            if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
            elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
            else:
                self.possibleRooms.remove(card)
        
            self.ownersAndCards[owner][card] = 1
        
            for op in self.opponents:
                self.ownersAndCards[op][card] = 0    
        
        
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
                self.possibleWeapons.remove(card)
                
        else: 
                self.possibleRooms.remove(card)
        
        for op in self.opponents:
            self.ownersAndCards[op][card] = 0
        
        self.ownersAndCards[owner][card] = 1
        
        
        
    #Possible moves and helpers    
    def makeAccusation(self, perp, weapon, room):
        return self.game.makeAccusation(self, perp, weapon, room)
    
    def chooseSuggestion(self):
        suspect = random.choice(self.possibleSuspects)
        weapon = random.choice(self.possibleWeapons)
        room = random.choice(self.possibleRooms)
        return suspect, weapon, room
        
    def makeSuggestion(self, perp, weapon, room):
        
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        
        if(owner!=None):
            if(len(self.ownersAndCards[owner])<owner.getNumCards()):
                self.crossOff(owner, card)

            else:
                print(owner+ "seems to already have "+ len(self.ownersAndCards[owner]) + "cards. There was a mistake in tracking cards")
    
    def isDealt(self, card):
        self.cards.append(card)
        self.crossOff(self, card)
        self.numCards+=1
        
    def hasACard(self, perp, weapon, room):
        for card in self.cards:
            if card == perp or card == weapon or card == room:
                return card
        return None

    #playTurn method
    def playTurn(self):
        if not self.inGame:
            return
    
        # For demo purposes, randomly suggest or accuse
        # You can replace this with smarter logic later
        

        perp, weapon, room = self.chooseSuggestion()
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)

        if owner is None:
            print(f"No one disproved. {self.name} might try an accusation!")
            if self.makeAccusation(perp, weapon, room):
                return self.name
            else:
                print(f"{self.name} made a wrong accusation and is out.")
                self.inGame = False
        else:
            print(f"{owner.name} showed a card.")
            self.crossOff(owner, card)

        if len(self.possibleSuspects) == 1 and len(self.possibleRooms) == 1 and len(self.possibleWeapons) == 1:
            if self.makeAccusation(self.possibleSuspects[0], self.possibleWeapons[0], self.possibleRooms[0]):
                print(f"{self.name} WINS! The solution was correct.")
                exit(0)
            else:
                print(f"{self.name} made a wrong accusation and is out.")