import time 
import random
from .Card import Card
from collections import deque
import logging 

class GameRules:
    """Class to run and manage the main game loop and the game mechanisms"""
    totalCards = 21
    ROOMS = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
    SUSPECTS = ["Colonel Mustard", "Reverend Green", "Miss Scarlet", "Mrs. Peacock", "Mrs White", "Professor Plum" ]
    WEAPONS = ["Knife", "Candlestick", "Rope", "Revolver", "Wrench", "Lead Pipe"]
    
    def __init__(self, players, hasHuman = False):
        self.players = players
        self.numPlayers = len(players)
        self.suspectCards = {}
        self.weaponCards = {}
        self.roomCards = {}
        self.deck = {}
        self.cards = []
        self.hasHuman = hasHuman
        self.suggestionLog = deque(maxlen=10) # Using a deque to limit memory usage
        self.turn=1
        self.gameTurn = 0
        self.solution = None # Will be set in reset_game()
        self.last_reward = 0.0 # New attribute to store the last calculated reward
        
        for card in self.SUSPECTS:
            suspect = Card("Suspect", card)
            self.suspectCards[card] = suspect
            self.deck[card] = suspect
            self.cards.append(suspect)
            
        for card in self.WEAPONS:
            weapon = Card("Weapon", card) 
            self.weaponCards[card] = (weapon)  
            self.deck[card] = weapon
            self.cards.append(weapon)
            
        for card in self.ROOMS:
            room = Card("Room", card)
            self.roomCards[card] = (room)
            self.deck[card] = room
            self.cards.append(room)
            
        for i, card in enumerate(self.cards):
            card.global_index = i
        # We need a solution set up from the beginning
        self.updateSolution()
        
        for player in players:
            player.setOpponents([p for p in players if p != player])

    def updateSolution(self):
        """Randomly selects a new solution for the game"""
        suspect = random.choice(self.SUSPECTS)
        weapon = random.choice(self.WEAPONS)
        room = random.choice(self.ROOMS)
        self.solution = {
            "suspect": self.suspectCards[suspect],
            "weapon": self.weaponCards[weapon],
            "room": self.roomCards[room]
        }
    
    def reset_game(self):
        """Resets the game state for a new episode."""
        self.turn = 1
        self.gameTurn = 0
        self.suggestionLog.clear()
        self.updateSolution()
        for player in self.players:
            player.cards = []
            player.inGame = True

    def makeAccusation(self, player, perp, weapon, room):
        """
        Processes an accusation.
        Returns True if the accusation is correct, False otherwise.
        """
        is_correct = (
            perp == self.solution["Suspect"] and
            weapon == self.solution["Weapon"] and
            room == self.solution["Room"]
        )
        
        if is_correct:
            # Game ends
            self.last_reward = self.calculate_accusation_reward(True)
            print(f"Correct accusation by {player.name}! The solution was {perp.name}, {weapon.name}, {room.name}.")
            return True
        else:
            # Player is eliminated
            player.inGame = False
            self.last_reward = self.calculate_accusation_reward(False)
            print(f"Incorrect accusation by {player.name}. The solution was not {perp.name}, {weapon.name}, {room.name}.")
            return False

    def get_last_reward(self):
        return self.last_reward
    
    
    def makeSuggestion(self, player, perp, weapon, room):
        """Implements the suggestion mechanism. Returns the responder and the card shown""" 
        print(f"{player.name} suggests: {perp} with {weapon} in {room}")
        logging.info(f"{player.name} suggests: {perp} with {weapon} in {room}")
        suggestionCards = [perp, weapon, room]
        playerPos = self.players.index(player) 
        i = (playerPos +1) % len(self.players)
        
        suggestion_record = {
        "turn": self.turn,  # or a turn counter if you have one
        "suggester": player,
        "suggestion": suggestionCards,
        "ambiguous": True,
        "responder": None,
        "card_shown": None,
        "possible_shown_cards": set(suggestionCards),
        "skipped_players" : []
        }
        
        while(i!= playerPos):
            print(self.players[i], end = " ")
            print("is checking their hand")
            logging.info(self.players[i].name + " is checking their hand")
            
            matching_cards = [c for c in suggestionCards if self.players[i].hasCard(c)]
            if matching_cards:
                # Ask player which card to reveal (RLPlayer overrides this)
                cardShown = self.players[i].revealCard(matching_cards)
            else:
                cardShown = None

            
            if(cardShown!=None):
                suggestion_record['responder'] = self.players[i]
                suggestion_record['card_shown'] = cardShown
                self.suggestionLog.append(suggestion_record)
                logging.info(f"{self.players[i].name} showed a card - {cardShown.name}" )
                # reward the suggester a small positive signal if someone shows a card (info gain)
                try:
                    # Suggester is 'player' argument to makeSuggestion
                    sugg = player
                    reward_for_suggester = 0.1  # tuneable; small positive reward for getting info
                    next_obs = self.get_observation_for(sugg)
                    done = False
                    if hasattr(sugg, "store_transition"):
                        sugg.store_transition(reward_for_suggester, next_obs, done)
                except Exception:
                    pass
                
                return self.players[i], cardShown 
            
            suggestion_record['skipped_players'].append(self.players[i])
            i = (i + 1) % len(self.players)     
        
        self.suggestionLog.append(suggestion_record)
        try:
            sugg = player
            reward_for_suggester = 0.2  # stronger signal — suggestion narrowed down
            next_obs = self.get_observation_for(sugg)
            done = False
            if hasattr(sugg, "store_transition"):
                sugg.store_transition(reward_for_suggester, next_obs, done)
        except Exception:
            pass    
        return None, None
    
    
    def getPublicSuggestionLog(self, player):
        """Returns a log of the suggestions made, adjusting the visibility of the cards shown based on the player calling this function"""
        public_log = []
        for rec in self.suggestionLog:
            entry ={
                "turn": rec["turn"],
                "suggester": rec["suggester"],
                "suggestion": list(rec["suggestion"]),
                "responder": rec["responder"] if rec["responder"] else None,
                "card_shown": None,  # hide the card shown here
                "ambiguous": rec["ambiguous"],
                "skipped_players": rec.get("skipped_players")
            }
            if player == rec["suggester"] or player == rec["responder"]:
                entry["card_shown"] = rec["card_shown"]
                entry["ambiguous"] = False
            public_log.append(entry)
        return public_log

    
    def dealCards(self):
        """Deals and rules out initial cards"""
        self.solution = {
            "Suspect" : random.choice(list(self.suspectCards.values())),
            "Weapon" : random.choice(list(self.weaponCards.values())),
            "Room" : random.choice(list(self.roomCards.values()))
        }
        
        #Take out solution cards from deck
        self.deck.pop((self.solution.get("Suspect")).getName())
        self.deck.pop((self.solution.get("Weapon")).getName())
        self.deck.pop((self.solution.get("Room")).getName())
        
        logging.info("Solution cards are "+(self.solution.get("Suspect")).getName()+ ", "+self.solution.get("Weapon").getName()+", "+self.solution.get("Room").getName())
        
        deckCards = list(self.deck.keys())
        random.shuffle(deckCards)
        playerIter =0
        for card in deckCards:
            self.players[playerIter].isDealt(self.deck.get(card))
            playerIter+=1
            if(playerIter==len(self.players)):
                playerIter = 0
                
        for player in self.players:
            print(player.name + " has " + str(player.numCards)+" cards")
            logging.info(player.name + " has " + str(player.numCards)+" cards")
            player.initialCrossOff()
            if player.type == "Human":
                    player.revealCards()
                    
                    
    def checkAllPlayers(self):
        """Checks to see if there are still enough players left"""   
        count =0
        for player in self.players:
            if(player.inGame):
                count+=1
            if(count > 1):
                return True
        return False
    
    
    def findWinner(self):
        """Returns the first player still in the game"""
        for player in self.players:
            if(player.inGame):
                return player
            
    
    # ... (unchanged game loop methods)
    def gameLoop(self):
        """Main game loop - this is now handled by the trainer"""
        # The trainer class will manage the game loop and player turns.
        # This method is now effectively deprecated in the RL context.
        pass
    
    # --- NEW METHODS FOR RL REWARDS ---
    
    def calculate_accusation_reward(self, is_correct):
        """Calculates the reward for an accusation."""
        self.last_reward = 1000.0 if is_correct else -1000.0
        return self.last_reward

    def calculate_suggestion_reward(self, responder, rl_player, last_log_len):
        """
        Calculates a reward for a suggestion.
        A positive reward if an opponent responds with a card, 
        and a small negative reward if they do not.
        """
        if responder and responder != rl_player:
            # Gained new information
            self.last_reward = 1.0
        else:
            # No new info from an opponent, a less efficient move
            self.last_reward = -0.5
        return self.last_reward

    def calculate_reveal_reward(self, revealed_card):
        """Calculates a reward for a reveal. More complex rewards could be added."""
        # A simple reward: maybe a small positive reward for revealing a less valuable card
        # Or a negative reward for revealing a valuable card
        self.last_reward = 0.1 # Placeholder: a small positive reward for now
        return self.last_reward
    
    def get_last_reward(self):
        return self.last_reward

    # ... (unchanged RL API methods)
    def getObservation(self, player):
        """Returns the observation for a given player"""
        from rl.utils import build_observation
        return build_observation(self, player)

    def getValidActionsMask(self, player):
        """Returns a boolean mask of the valid actions a player can make."""
        from rl.utils import build_action_mask
        return build_action_mask(self, player)