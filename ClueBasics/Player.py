import GameRules
class Player:
    
    def __init__(self, name, cards, game, opponents):
        self.name = name
        self.cards = cards
        self.numCards = len(cards)
        self.game = game
        self.opponents = opponents
        self.owners = dict.fromkeys(opponents)
        for opponent in self.opponents: 
            self.owners[opponent] = []
        self.possibleSuspects = list(game.SUSPECTS)
        self.possibleWeapons = list(game.WEAPONS)
        self.possibleRooms = list(game.ROOMS)
        self.crossOff(self, cards)
      
    def __str__(self):
        return f"Player: {self.name}"
        
    def getNumCards(self):
        return self.numCards
    
    def crossOff(self, owner, cardList):
        for card in cardList:
            if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
            elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
            else: self.possibleRooms.remove(card)
     
    def makeAccusation(self, perp, weapon, room):
        return self.game.makeAccusation(self, perp, weapon, room)
    
    def makeSuggestion(self, perp, weapon, room):
        owner, card = self.game.makeSuggestion(self, perp, weapon, room)
        
        if(owner!=None):
            if(len(self.owners[owner])<owner.getNumCards()):
                self.owners[owner].insert(0, card)
                if(card.getType() == 'Suspect'):
                    self.possibleSuspects.remove(card)
                
                elif (card.getType() == 'Weapon'):
                    self.possibleWeapons.remove(card)
                
                else: self.possibleRooms.remove(card)

            else:
                print(owner+ "seems to already have "+ len(self.owners[owner]) + "cards. Replacing last one with "+ card )
                
    #playTurn method
   