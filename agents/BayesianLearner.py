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
        
    def removeCardFromPossibleCategories(self, card):
        """Remove a card from possible categories when we know a player has it (not in solution)"""
        if card.getType() == 'Suspect':
            if card in self.possibleSuspects:    
                self.possibleSuspects.remove(card)
                logging.info(f"Removed {card} from possible suspects - a player has it")
                
        elif card.getType() == 'Weapon':
            if card in self.possibleWeapons:
                self.possibleWeapons.remove(card)
                logging.info(f"Removed {card} from possible weapons - a player has it")
                
        else:
            if card in self.possibleRooms: 
                self.possibleRooms.remove(card)
                logging.info(f"Removed {card} from possible rooms - a player has it")
                
    def removeOtherCardsFromPossibleCategories(self, card):
        """Remove other cards from possible categories when we know one card is in the solution"""
        if card.getType() == 'Suspect':
            cards_to_remove = [otherCard for otherCard in self.possibleSuspects if otherCard != card]
            for otherCard in cards_to_remove:    
                self.possibleSuspects.remove(otherCard)
                # Set these other cards' solution probability to 0
                self.setProbability("Solution", otherCard, 0)
                logging.info(f"Removed {otherCard} from possible suspects - {card} is the solution suspect")
                
        elif card.getType() == 'Weapon':
            cards_to_remove = [otherCard for otherCard in self.possibleWeapons if otherCard != card]
            for otherCard in cards_to_remove:    
                self.possibleWeapons.remove(otherCard)
                # Set these other cards' solution probability to 0
                self.setProbability("Solution", otherCard, 0)
                logging.info(f"Removed {otherCard} from possible weapons - {card} is the solution weapon")
                
        else:
            cards_to_remove = [otherCard for otherCard in self.possibleRooms if otherCard != card]
            for otherCard in cards_to_remove:    
                self.possibleRooms.remove(otherCard)
                # Set these other cards' solution probability to 0
                self.setProbability("Solution", otherCard, 0)
                logging.info(f"Removed {otherCard} from possible rooms - {card} is the solution room")
    
    def checkForSolutionCards(self):
        """Check for cards that must be in the solution or ruled out from solution"""
        for card in self.game.cards:
            solution_prob = self.getProbability("Solution", card)
            # Case 1: If solution probability is 0, remove from possible categories
            # (This means a player definitely has it)
            if solution_prob == 0:
                self.removeCardFromPossibleCategories(card)
            
            # Case 2: If solution probability is 1, remove other cards from same category
            elif solution_prob >= 1.0:
                logging.info(f"Confirmed {card} is in the solution")
                self.removeOtherCardsFromPossibleCategories(card)
            
            # Case 3: If no player can have the card, it must be in the solution
            else:
                non_solution_total = sum(self.getProbability(owner, card) 
                                       for owner in self.owners if owner != "Solution")
                if non_solution_total == 0 and solution_prob < 1.0:
                    logging.info(f"No player can have {card} - must be in solution")
                    # Set solution probability to 1 and zero out all others
                    for owner in self.owners:
                        if owner == "Solution":
                            self.setProbability(owner, card, 1.0)
                        else:
                            self.setProbability(owner, card, 0)
                    
                    # Remove other cards from same category since this one is the solution
                    self.removeOtherCardsFromPossibleCategories(card)
    
    #Initial cross off and belief matrix set up
    def initialCrossOff(self):
        self.createBeliefMatrix()
        logging.info("Creating belief matrix")
        logging.info(self.ownersAndCards)
        logging.info("Crossing off")
        for card in self.cards:
            self.crossOff(self, card)
        
        for card in self.game.cards:
            if card not in self.cards:
                self.setProbability(self, card, 0)

        self.checkForSolutionCards()
    
    #Cross off a card from possibilities
    def crossOff(self, owner, card):
        # Remove from possible categories since a player has it
        if(card.getType() == 'Suspect'):
            if(card in self.possibleSuspects):    
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
            if(card in self.possibleWeapons):
                self.possibleWeapons.remove(card)
                
        else:
            if(card in self.possibleRooms): 
                self.possibleRooms.remove(card)
        
        # Set probabilities: owner has it (1), everyone else doesn't (0)
        for op in self.owners:
            self.setProbability(op, card, 0)
        
        self.setProbability(owner, card, 1)
        
        # Redistribute probabilities for other cards that this owner might have
        for other_card in self.ownersAndCards[owner]:
            if other_card == card:
                continue
            prob = self.ownersAndCards[owner][other_card]
            if prob == 1 or prob == 0:
                continue
            if prob>0.04:
                self.setProbability(owner, other_card, prob-0.04)
            else:
                self.setProbability(owner, other_card, prob/2)
        
        # Check if any other cards must now be in the solution
        self.checkForSolutionCards()
    
    def chooseSuggestion(self):
        suspect = self.highestSolutionChance(self.possibleSuspects)
        weapon = self.highestSolutionChance(self.possibleWeapons)
        room = self.highestSolutionChance(self.possibleRooms)
        return suspect, weapon, room
    
    def processNewSuggestions(self):
        public_log = self.game.getPublicSuggestionLog(self)
        print(public_log)
        print("last processed turn " + str(self.lastProcessedTurn))
        for rec in public_log:
            if rec["turn"] <= self.lastProcessedTurn:
                continue  # Already processed
            logging.info("Processing turn: "+str(rec["turn"]))
            suggester = rec["suggester"]
            responder = rec["responder"]
            suggestion_cards = rec["suggestion"]
            card_shown = rec["card_shown"]
            skipped_players = rec.get("skipped_players", [])
            
            # If someone showed a card, cross off that card for the responder
            if responder:
                if card_shown:
                    self.crossOff(responder, card_shown)
                     # Players who skipped can't have any of the suggested cards
                    for skipped_player in skipped_players:
                        for card in suggestion_cards:
                            logging.info(f"Processing - set {skipped_player.name}'s {card.name} to 0")
                            if self.getProbability(skipped_player, card) != 0:
                                self.setProbability(skipped_player, card, 0)
                                
                    
                    # For the rest of the players, increase chances of their having the other cards                                
                    for owner in self.owners:
                        if not owner == responder:
                            for card in self.game.cards:
                                if (not card == card_shown) and (not owner == self) and (not self.getProbability(owner, card) == 1) and (not self.getProbability(owner, card) == 0):
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
                            logging.info(f"Processing - set {skipped_player.name}'s {card.name} to 0")
                            if self.getProbability(skipped_player, card) != 0:
                                self.setProbability(skipped_player, card, 0)
                        
                        # Increase the probability of their having the other cards 
                        for card in self.game.cards:
                            if self.getProbability(skipped_player, card) != 0 and self.getProbability(skipped_player, card) != 1:
                                prob = self.ownersAndCards[skipped_player][card]
                                if prob  < 0.95:
                                    self.setProbability(skipped_player, card, prob + 0.05)
                                else:
                                    self.setProbability(skipped_player, card,  prob  + (1-prob)/2)
                        
                        # Normalize the suggested cards across remaining players
                        for card in suggestion_cards:
                            remaining_players = [p for p in self.owners if p != skipped_player]
                            self.normalizeCardAcrossPlayers(card, remaining_players)
                    
                    # After processing skipped players, check for solution cards
                    self.checkForSolutionCards()

            else:
                # No one responded - suggester must have all cards or they're in solution
                for card in suggestion_cards:
                    for player in self.owners:
                        if player not in [suggester, 'Solution']:
                            self.setProbability(player, card, 0)
                    
                    self.normalizeCardAcrossPlayers(card, [suggester, 'Solution'])
                
                # Check if any cards must be in solution after this update
                self.checkForSolutionCards()
                                
            self.privateSuggestionLog.append(rec)               
            
            self.runBackwardInference()
            self.lastProcessedTurn = rec['turn']

    
    def normalizeCardAcrossPlayers(self, card, players):
        fractions = []
        for player in players:
            if player == self:
                continue
            prob = self.ownersAndCards[player][card]
            fractions.append((player, prob))

        total = sum(frac for _, frac in fractions)
        if total == 0:
            return 

        # Normalize each player's fraction
        for player, frac in fractions:
            new_frac = frac / total
            self.ownersAndCards[player][card] = new_frac
        
        # After normalization, check if solution status changed
        self.checkForSolutionCards()
            
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
                    print(f"Turn {int(turn)}: Deduced {responder} showed {shown_card}")
                logging.info(f"Turn {int(turn)}: Deduced {responder} showed {shown_card}")
                self.crossOff(responder, shown_card)  # This will also check for solution cards
                
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
        
        self.processNewSuggestions()
        perp, weapon, room = self.chooseSuggestion()
        responder, shown_card = self.game.makeSuggestion(self, perp, weapon, room)
        
        # Eliminates the new card
        if not responder is None:
            if not self.game.hasHuman:
                print(f"{responder.name} showed a card - {shown_card.name}.")
            else: print(f"{responder.name} showed a card")
            self.crossOff(responder, shown_card)
            
            # Update probabilities for other players
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
            self.processNewSuggestions()
            
        if responder is None:
            if(perp not in self.cards and weapon not in self.cards and room not in self.cards):
                if self.makeAccusation(perp, weapon, room):
                    return self.name

        # Check if we can make a definitive accusation
        if len(self.possibleSuspects) == 1 and len(self.possibleRooms) == 1 and len(self.possibleWeapons) == 1:
            if self.makeAccusation(self.possibleSuspects[0], self.possibleWeapons[0], self.possibleRooms[0]):
                return self.name
            
        
        logging.info(self.name+"'s belief matrix "+str(self.ownersAndCards))