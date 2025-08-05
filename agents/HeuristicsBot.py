from ClueBasics.Player import Player
import random
import math
from fractions import Fraction

class HeuristicsBot(Player):
    
    def __init__(self, name, game, type):
        super.__init__(name, game, type)
        self.lastProcessedTurn = 0 
    
    #------------------------------------------------------
    # Helpers
    
    def entropy(self, card):
            probs = (self.getProbability(owner, card) for owner in self.owners)
            return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def highestSolutionChance(self, category):
        max_prob = max(self.getProbability("Solution", card) for card in category)
        tied_cards = [card for card in category if self.getProbability("Solution", card) == max_prob]
        
        return max(tied_cards, key=self.entropy)
        
    
    #Initial cross off and belief matrix set up
    def initialCrossOff(self):
        self.createBeliefMatrix()
        for card in self.cards:
            self.crossOff(self, card)
    
    #Cross off a card from possibilities
    def crossOff(self, owner, card):
        if(card.getType() == 'Suspect'):
                self.possibleSuspects.remove(card)
                
        elif (card.getType() == 'Weapon'):
                self.possibleWeapons.remove(card)
                
        else: 
                self.possibleRooms.remove(card)
        
        for op in self.owners:
            self.setProbability(op, card, 0, 1)
        
        self.setProbability(owner, card, 1, 1)
        
        for other_card in self.ownersAndCards[owner]:
            if other_card == card:
                continue
            num, denom = self.ownersAndCards[owner][other_card]
            if denom > 1 and num > 0:
                self.setProbability(owner, other_card, num - 1, denom - 1)
    
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
                                    num, denom = self.ownersAndCards[owner][card]
                                    self.setProbability(owner, card, num, denom-1)
                                
                    self.normalizeCardAcrossPlayers(card_shown, self.owners)
                else:
                    # Players who skipped can't have any of the suggested cards
                    for skipped_player in skipped_players:
                        for card in suggestion_cards:
                            if self.getProbability(skipped_player, card) != 0:
                                self.setProbability(skipped_player, card, 0 , 1)
                            #CHECK increase the probability of their having the other cards 
                        for card in self.game.cards:
                            if self.getProbability(skipped_player, card) != 0 and self.getProbability(skipped_player, card) != 1:
                                num, denom = self.ownersAndCards[skipped_player][card]
                                self.setProbability(skipped_player, card, num, denom-3)
                            # CHECK increase the probability of the card being with the remaining players
                            self.normalizeCardAcrossPlayers(card, skipped_players)
                            
                
                    portion = Fraction(1, len(suggestion_cards))  # e.g. 1/3
                    others = [p for p in self.owners if p != responder]

                    for card in suggestion_cards:
                            # Step 1: Set responder’s value explicitly to 1/len
                            self.ownersAndCards[responder][card] = (portion.numerator, portion.denominator)

                            # Step 2: Calculate remaining portion to assign to others
                            remaining = Fraction(1, 1) - portion

                            # Step 3: Get current total of others' probabilities
                            total_other_prob = sum(
                                Fraction(*self.ownersAndCards[p][card])
                                for p in others if self.ownersAndCards[p][card][1] > 0
                            )

                            if total_other_prob == 0:
                                # No other belief — just zero them all
                                for p in others:
                                    self.ownersAndCards[p][card] = (0, 1)
                            else:
                                # Scale others to sum to remaining
                                for p in others:
                                    num, denom = self.ownersAndCards[p][card]
                                    if denom == 0:
                                        continue
                                    frac = Fraction(num, denom)
                                    new_frac = frac * (remaining / total_other_prob)
                                    self.ownersAndCards[p][card] = (new_frac.numerator, new_frac.denominator)

                            self.normalizeCardAcrossPlayers(card, self.owners)

            else:
                for card in suggestion_cards:
                    for player in self.owners:
                        if player not in [suggester, 'SOLUTION']:
                            self.setProbability(player, card, 0, 1)
                    
                    self.normalizeCardAcrossPlayers(card, [suggester, 'SOLUTION'])
                                
                                
                            
            
            # TODO: Implement backward inference logic here 
            
            self.lastProcessedTurn = rec["turn"]

    
    def normalizeCardAcrossPlayers(self, card, players):
        fractions = []
        for player in players:
            num, denom = self.ownersAndCards[player][card]
            if denom == 0:
                fractions.append((player, Fraction(0, 1)))
            else:
                fractions.append((player, Fraction(num, denom)))

        
        total = sum(frac for _, frac in fractions)
        if total == 0:
            return 

        # Step 3: Normalize each player's fraction
        normalized = []
        for player, frac in fractions:
            new_frac = frac / total
            normalized.append((player, new_frac))

        # Step 4: Find common denominator to convert all to integer fractions
        denominators = [f.denominator for _, f in normalized]
        common_denom = math.lcm(*denominators)

        for player, frac in normalized:
            scaled_num = frac.numerator * (common_denom // frac.denominator)
            self.ownersAndCards[player][card] = (scaled_num, common_denom)
            
            
    
    def runBackwardInference(self):
        public_log = self.game.getPublicSuggestionLog(self)
        
        for rec in public_log:
            responder = rec["responder"]
            if not responder or rec["card_shown"]:  # Nothing to infer
                continue
            
            suggestion_cards = rec["suggestion"]
            turn = rec["turn"]
            
            # Check what the responder *might* have at this point
            possible_cards = []
            for card in suggestion_cards:
                num, denom = self.ownersAndCards[responder][card]
                if denom != 0 and Fraction(num, denom) > 0:
                    possible_cards.append(card)
            
            if len(possible_cards) == 1:
                # Only one possible card => must be the one shown
                shown_card = possible_cards[0]
                self.crossOff(responder, shown_card)  # sets to (1,1), zeros others
                print(f"Turn {turn}: Deduced {responder} showed {shown_card}")
                rec["card_shown"] = shown_card  # Save for future use

                for owner in self.owners:
                        if not owner == responder:
                            for card in self.game.cards:
                                if (not card == shown_card) and (not self.getProbability(owner, card) == 1) and (not self.getProbability(owner, card) == 0):
                                    num, denom = self.ownersAndCards[owner][card]
                                    self.setProbability(owner, card, num, denom-1)
                                
                self.normalizeCardAcrossPlayers(shown_card, self.owners)


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
