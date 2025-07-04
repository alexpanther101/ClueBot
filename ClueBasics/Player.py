import GameRules
class Player:
    
    def __init__(self, name, cards, game, opponents):
        self.name = name
        self.cards = cards
        self.numCards = len(cards)
        self.game = game
        self.opponents = opponents
        self.ownersAndCards = dict.fromkeys(opponents)
        for opponent in self.opponents: 
            self.ownersAndCards[opponent] = []
        self.ownersAndCards[self] = []
        self.possibleSuspects = list(game.SUSPECTS)
        self.possibleWeapons = list(game.WEAPONS)
        self.possibleRooms = list(game.ROOMS)
        self.crossOffMulti(self, cards)
        self.inGame = True
      
    def __str__(self):
        return f"Player: {self.name}"
        
    def getNumCards(self):
        return self.numCards
    
    
    
    #Removes cards from possible solutions and maps to owner
    def crossOffMulti(self, owner, cardList):
        for card in cardList:
            if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
            elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
            else: self.possibleRooms.remove(card)
        self.ownersAndCards[owner].append(card)
        
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
                self.possibleWeapons.remove(card)
                
        else: self.possibleRooms.remove(card)
        
        self.ownersAndCards[owner].append(card)
        
    #Possible moves and helpers    
        
    def makeAccusation(self, perp, weapon, room):
        return self.game.makeAccusation(self, perp, weapon, room)
    
    def makeSuggestion(self, perp, weapon, room):
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        
        if(owner!=None):
            if(len(self.ownersAndCards[owner])<owner.getNumCards()):
                self.owners[owner].insert(0, card)
                if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
                elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
                else: self.possibleRooms.remove(card)

            else:
                print(owner+ "seems to already have "+ len(self.ownersAndCards[owner]) + "cards. There was a mistake in tracking cards")
    
    def isDealt(self, card):
        self.cards.append(card)
        self.crossOff(self, card)
                             
    #playTurn method
   