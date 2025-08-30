from .GameRules import GameRules
import random
from .Card import Card
from abc import ABC, abstractmethod

# Player is an abstract class that bots inherit from
class Player(ABC):
    
    #Set up methods
    def __init__(self, name, game, type):
        self.name = name
        self.game = game
        self.possibleSuspects = list(game.suspectCards.values())
        self.possibleWeapons = list(game.weaponCards.values())
        self.possibleRooms = list(game.roomCards.values())
        self.inGame = True
        self.cards = []
        self.numCards = 0
        self.type = type
        self.privateSuggestionLog = []
        self.owners = []
        
    def __str__(self):
        return f"{self.name}"
    
    def __repr__(self):
        return str(self)
        
    def getNumCards(self):
        return self.numCards
    
    def setOpponents(self, opponents):
        self.opponents = opponents
        self.players = self.opponents + [self]
        self.owners = self.opponents + [self] + ["Solution"]
        self.ownersAndCards = {owner: {} for owner in self.owners}
        
        
    def getProbability(self, owner, card):
        return self.ownersAndCards[owner][card]
        

    def setProbability(self, owner, card, prob):
        self.ownersAndCards[owner][card] = prob
    
    def createBeliefMatrix(self):
        self.players = self.opponents + [self]
        self.owners = self.opponents + [self] + ["Solution"]
        total_cards = self.game.totalCards
        #Create belief matrix for each opponent - mapping each owner's card to a belief 
        for owner in self.owners:
            self.ownersAndCards[owner] = {}
        
        for player in self.players: 
            num_cards = player.getNumCards()
            for card in self.game.cards:
                self.ownersAndCards[player][card] = num_cards/ total_cards

        for card in self.game.suspectCards.values():
            self.ownersAndCards["Solution"][card] = 1/ len(self.game.SUSPECTS)
         
        for card in self.game.weaponCards.values():
            self.ownersAndCards["Solution"][card] = 1/ len(self.game.WEAPONS)
            
        for card in self.game.roomCards.values():
            self.ownersAndCards["Solution"][card] = 1/ len(self.game.ROOMS)
            
            
        
    #Deals a card to self    
    def isDealt(self, card):
        self.cards.append(card)
        self.crossOff(self, card)
        self.numCards+=1
    
#---------------------------------------------------------------------------------------------
# Helper methods
    def getFlattenedBeliefs(self):
        """
        Returns a flat list representing the probability that each card
        belongs to each possible owner (including the solution envelope).
        Order: [card1_owner1, card1_owner2, ..., cardN_solution]
        """
        state = []
        for card in self.game.cards:
            for owner in self.owners:
                prob = self.ownersAndCards[owner].get(card, 0.0)
                state.append(prob)
        return state

    #Removes cards from possible solutions and maps to owner
    def crossOffMulti(self, owner, cardList):
        for card in cardList:
            if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
            elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
            else:
                self.possibleRooms.remove(card)
        
    
    def revealCards(self):
        print(self.name + " has cards: ")
        for i, card in enumerate(self.cards):
            print(f"{i + 1}. {card.name}")
            
    #Removes single card from possible solutions and maps to owner    
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
            if card in self.possibleSuspects:
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
            if card in self.possibleWeapons:
                self.possibleWeapons.remove(card)
                
        else: 
            if card in self.possibleRooms:
                self.possibleRooms.remove(card)
     
        
    #Checks if a player has a card and returns it
    def hasACard(self, perp, weapon, room):
        for card in self.cards:
            if card == perp or card == weapon or card == room:
                return card
        return None
    
    def hasCard(self, card):
        if(card in self.cards):
            return True
        else: return False
        
    def getNumCards(self):
        return self.numCards
    
    def initialCrossOff(self):
        self.createBeliefMatrix()
        
#--------------------------------------------------------------------------------------------
# Player mechanics        
    def chooseCard(self):
        pass
    
    def updateBeliefs(self):
        """Optional: Override in smarter bots to update probability matrix"""
        pass
    
    def makeAccusation(self, perp, weapon, room):
        return self.game.makeAccusation(self, perp, weapon, room)
    
    @abstractmethod 
    def chooseSuggestion(self):
        """Must be implemented by child class"""
        pass
    
    def showCard(self, matching_cards):
        return random.choice(matching_cards)
    
    """deprecated -> use revealCard instead"""    
    def refuteSuggestion(self, suggestionCards):
        matching_cards = [card for card in self.cards if card in suggestionCards]
        if not matching_cards:
            return None
        chosen = self.showCard(matching_cards)
        return chosen
    
    def revealCard(self, matching_cards):
        """
        Default behavior: pick the first card (or random) if multiple can be shown.
        """
        if not matching_cards:
            return None
        return random.choice(matching_cards)
    
    def makeSuggestion(self, perp, weapon, room):
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        return owner, card
    
    
    #playTurn method
    @abstractmethod
    def playTurn(self, obs, valid_mask):
        """Must be implemented by child class"""
        pass