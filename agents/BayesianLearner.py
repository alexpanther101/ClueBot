from ClueBasics.Player import Player
import random
import math
from fractions import Fraction
import logging

class BayesianLearner(Player):
    
    def __init__(self, name, game, type):
        super().__init__(name, game, type)
        self.lastProcessedTurn = 0 
    
    #------------------------------------------------------
    # Helpers
    
    def entropy(self, card):
            probs = (self.getProbability(owner, card) for owner in self.owners)
            return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def highestSolutionChance(self, category):
        max_prob = max(self.getProbability("Solution", card) for card in category)
        
        tied_cards = [card for card in category if self.getProbability("Solution", card) == max_prob]
        max_entropy = max(self.entropy(card) for card in tied_cards)
        tied_entropy = [card for card in category if self.entropy(card) == max_entropy]
        return random.choice(tied_entropy)
        
    
    #Initial cross off and belief matrix set up
    def initialCrossOff(self):
        self.createBeliefMatrix()
    
    #Cross off a card from possibilities
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
            if(card in self.possibleSuspects):    
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
            if(card in self.possibleWeapons):
                self.possibleWeapons.remove(card)
                
        else:
            if(card in self.possibleRooms): 
                self.possibleRooms.remove(card)
        
        for op in self.owners:
            self.setProbability(op, card, 0)
        
        self.setProbability(owner, card, 1)
        
        for other_card in self.ownersAndCards[owner]:
            if other_card == card:
                continue
            prob = self.ownersAndCards[owner][other_card]
            if prob>0.04:
                self.setProbability(owner, other_card, prob-0.04)
            else:
                self.setProbability(owner, other_card, prob/2)
    
    def chooseSuggestion(self):
        suspect = self.highestSolutionChance(self.possibleSuspects)
        weapon = self.highestSolutionChance(self.possibleWeapons)
        room = self.highestSolutionChance(self.possibleRooms)
        return suspect, weapon, room
    
    def processNewSuggestions(self):
        public_log = self.game.getPublicSuggestionLog(self)
        for rec in public_log:
            if rec["turn"] <= self.lastProcessedTurn:
                continue  # Already processed
            
            suggester = rec["suggester"]
            responder = rec["responder"]
            suggestion_cards = rec["suggestion"]
            card_shown = rec["card_shown"]
            skipped_players = rec.get("skipped_players", [])
            
            # If someone showed a card, cross off that card for the responder
            if responder:
                if card_shown:
                    self.crossOff(responder, card_shown)
                    #CHECK For the rest of the players, increase chances of their having the other cards ->
                    for owner in self.owners:
                        if not owner == responder:
                            for card in self.game.cards:
                                if (not card == card_shown) and (not self.getProbability(owner, card) == 1) and (not self.getProbability(owner, card) == 0):
                                    prob = self.ownersAndCards[owner][card]
                                    if prob<0.97:
                                        self.setProbability(owner, card, prob + 0.03)
                                    else:
                                        self.setProbability(owner, card, prob  + (1-prob)/2)
                                
                    self.normalizeCardAcrossPlayers(card_shown, self.owners)
                else:
                    # Players who skipped can't have any of the suggested cards
                    for skipped_player in skipped_players:
                        for card in suggestion_cards:
                            if self.getProbability(skipped_player, card) != 0:
                                self.setProbability(skipped_player, card, 0)
                            #CHECK increase the probability of their having the other cards 
                        for card in self.game.cards:
                            if self.getProbability(skipped_player, card) != 0 and self.getProbability(skipped_player, card) != 1:
                                prob = self.ownersAndCards[skipped_player][card]
                                if prob  < 0.95:
                                    self.setProbability(skipped_player, card, prob + 0.05)
                                else:
                                    self.setProbability(skipped_player, card,  prob  + (1-prob)/2)
                            # CHECK increase the probability of the card being with the remaining players
                            self.normalizeCardAcrossPlayers(card, skipped_players)
                            

            else:
                for card in suggestion_cards:
                    for player in self.owners:
                        if player not in [suggester, 'Solution']:
                            self.setProbability(player, card, 0)
                    
                    self.normalizeCardAcrossPlayers(card, [suggester, 'Solution'])
                                
                                
            self.privateSuggestionLog.append(rec)               
            
            self.runBackwardInference()
            self.lastProcessedTurn = rec['turn']

    
    def normalizeCardAcrossPlayers(self, card, players):
        fractions = []
        for player in players:
            prob = self.ownersAndCards[player][card]
            fractions.append((player, prob))

        
        total = sum(frac for _, frac in fractions)
        if total == 0:
            return 

        # Step 3: Normalize each player's fraction
        normalized = []
        for player, frac in fractions:
            new_frac = frac / total
            self.ownersAndCards[player][card] = new_frac
            
            
    
    def runBackwardInference(self):
        for rec in self.privateSuggestionLog:
            responder = rec["responder"]
            if not responder or rec["card_shown"]:  # Nothing to infer
                continue
            
            suggestion_cards = rec["suggestion"]
            turn = rec["turn"]
            
            # Check what the responder *might* have at this point
            possible_cards = []
            for card in suggestion_cards:
                prob = self.ownersAndCards[responder][card]
                if prob > 0:
                    possible_cards.append(card)
            
            if len(possible_cards) == 1:
                # Only one possible card => must be the one shown
                shown_card = possible_cards[0]
                if not self.game.hasHuman:
                    print(f"Turn {int(turn/len(self.players))}: Deduced {responder} showed {shown_card}")
                logging.info(f"Turn {int(turn/len(self.players))}: Deduced {responder} showed {shown_card}")
                self.crossOff(responder, shown_card)  # sets to (1,1), zeros others
                
                rec["card_shown"] = shown_card  # Save for future use

                for owner in self.owners:
                        if not owner == responder:
                            for card in self.game.cards:
                                if (not card == shown_card) and (not self.getProbability(owner, card) == 1) and (not self.getProbability(owner, card) == 0):
                                    prob = self.ownersAndCards[owner][card]
                                    if prob <0.97:
                                        self.setProbability(owner, card, prob + 0.03)
                                    else: 
                                        self.setProbability(owner,  card,  prob + (1- prob)/2 )
                                
                self.normalizeCardAcrossPlayers(shown_card, self.owners)

    def playTurn(self, obs, valid_mask):
        if not self.inGame:
            return
        logging.info(self.name+"'s possible suspects "+str(self.possibleSuspects))
        logging.info(self.name+"'s possible weapons "+str(self.possibleWeapons))
        logging.info(self.name+"'s possible rooms "+str(self.possibleRooms))
        logging.info(self.name+"'s belief matrix "+str(self.ownersAndCards))
        
        self.processNewSuggestions()
        perp, weapon, room = self.chooseSuggestion()
        responder, card = self.game.makeSuggestion(self, perp, weapon, room)
        #Eliminates the new card
        if not responder is None:
            if not self.game.hasHuman:
                print(f"{responder.name} showed a card - {card.name}.")
            else: print(f"{responder.name} showed a card")
            logging.info(f"{responder.name} showed a card - {card.name}.")
            self.crossOff(responder, card)
            for owner in self.owners:
                        if not owner == responder:
                            for card in self.game.cards:
                                if (not card == card) and (not self.getProbability(owner, card) == 1) and (not self.getProbability(owner, card) == 0):
                                    prob = self.ownersAndCards[owner][card]
                                    if prob <0.97:
                                        self.setProbability(owner, card, prob + 0.03)
                                    else: 
                                        self.setProbability(owner,  card,  prob + (1- prob)/2 )
                                
            self.normalizeCardAcrossPlayers(card, self.owners)
            self.processNewSuggestions()
            
        if responder is None:
            if(perp not in self.cards and weapon not in self.cards and room not in self.cards):
                if self.makeAccusation(perp, weapon, room):
                    return self.name

                
        if len(self.possibleSuspects) == 1 and len(self.possibleRooms) == 1 and len(self.possibleWeapons) == 1:
            if self.makeAccusation(self.possibleSuspects[0], self.possibleWeapons[0], self.possibleRooms[0]):
                return self.name