from ClueBasics.Player import Player
import random
import math

class HeuristicsBot(Player):
    
    #------------------------------------------------------
    # Helpers
    
    def mostUncertainCard(self, category):
        #Calculating entropy for each card - 
        def entropy(card):
            prob = (self.ownersAndCards[opp][card] for opp in self.opponents)
            return -sum(p * math.log2(p) for p in prob if p > 0)
        
        return max(category, key=entropy)
    
    def initialCrossOff(self):
        self.createBeliefMatrix()
        for card in self.cards:
            self.crossOff(self, card)
    
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
                self.possibleWeapons.remove(card)
                
        else: 
                self.possibleRooms.remove(card)
        
        for op in self.owners:
            self.ownersAndCards[op][card] = 0
        
        self.ownersAndCards[owner][card] = 1
    
    def chooseSuggestion(self):
        suspect = self.mostUncertainCard(self.possibleSuspects)
        weapon = self.mostUncertainCard(self.possibleWeapons)
        room = self.mostUncertainCard(self.possibleRooms)
        return suspect, weapon, room

    def playTurn(self):
        if not self.inGame:
            return

        perp, weapon, room = self.chooseSuggestion()
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        #Eliminates the new card
        if not owner is None:
            print(f"{owner.name} showed a card.")
            self.crossOff(owner, card)

        if len(self.possibleSuspects) == 1 and len(self.possibleRooms) == 1 and len(self.possibleWeapons) == 1:
            if self.makeAccusation(self.possibleSuspects[0], self.possibleWeapons[0], self.possibleRooms[0]):
                print(f"{self.name} WINS! The solution was correct.")
                return self.name
            else:
                print(f"{self.name} made a wrong accusation and is out.")
