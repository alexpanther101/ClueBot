from .GameRules import GameRules
import random
from .Card import Card
from abc import ABC, abstractmethod

# Player is an abstract class that bots inherit from
class Player(ABC):
    
    #Set up methods
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
        return f"{self.name}"
    
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
                self.ownersAndCards[opponent][card] = 1/len(opponents)
            
            for card in self.possibleWeapons:
                self.ownersAndCards[opponent][card] = 1/len(opponents)
                
            for card in self.possibleRooms:
                self.ownersAndCards[opponent][card] = 1/len(opponents)
                #need to update that probability when cards start to fill up
        
        self.ownersAndCards[self] = {}
        
    #Deals a card to self    
    def isDealt(self, card):
        self.cards.append(card)
        self.crossOff(self, card)
        self.numCards+=1
    
#---------------------------------------------------------------------------------------------
# Helper methods

    #Removes cards from possible solutions and maps to owner
    def crossOffMulti(self, owner, cardList):
        for card in cardList:
            if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
            elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
            else:
                self.possibleRooms.remove(card)
        
        
    #Removes single card from possible solutions and maps to owner    
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
                self.possibleWeapons.remove(card)
                
        else: 
                self.possibleRooms.remove(card)
     
        
    #Checks if a player has a card and returns it
    def hasACard(self, perp, weapon, room):
        for card in self.cards:
            if card == perp or card == weapon or card == room:
                return card
        return None
    
    def getNumCards(self):
        return self.numCards
#--------------------------------------------------------------------------------------------
# Player mechanics        
    
    def updateBeliefs(self):
        """Optional: Override in smarter bots to update probability matrix"""
        pass
    
    def makeAccusation(self, perp, weapon, room):
        return self.game.makeAccusation(self, perp, weapon, room)
    
    @abstractmethod 
    def chooseSuggestion(self):
        """Must be implemented by child class"""
        pass
    
    
    def makeSuggestion(self, perp, weapon, room):
        
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        
        return owner, card
    
    
    #playTurn method
    @abstractmethod
    def playTurn(self):
        """Must be implemented by child class"""
        pass