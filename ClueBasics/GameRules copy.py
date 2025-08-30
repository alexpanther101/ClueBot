import time 
import random
from .Card import Card



class GameRules:
    """Class to run and manage the main game loop and the game mechanisms"""
    totalCards = 21
    ROOMS = ["Study", "Hall", "Lounge", "Library", "Billiard Room", "Dining Room", "Conservatory", "Ballroom", "Kitchen"]
    SUSPECTS = ["Colonel Mustard", "Reverend Green", "Miss Scarlet", "Mrs. Peacock", "Mrs White", "Professor Plum" ]
    WEAPONS = ["Knife", "Candlestick", "Rope", "Revolver", "Wrench", "Lead Pipe"]
    # gameboard = [[]]
    
    def __init__(self, players):
        self.players = players
        self.suspectCards = {}
        self.weaponCards = {}
        self.roomCards = {}
        self.deck = {}
        self.cards = []
        self.suggestionLog = []
        self.turn=0
        self.gameTurn = 0
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
                    
        self.solution = {
            "Suspect" : random.choice(list(self.suspectCards.values())),
            "Weapon" : random.choice(list(self.weaponCards.values())),
            "Room" : random.choice(list(self.roomCards.values()))
        }
        
        #Take out solution cards from deck
        self.deck.pop((self.solution.get("Suspect")).getName())
        self.deck.pop((self.solution.get("Weapon")).getName())
        self.deck.pop((self.solution.get("Room")).getName())

#--------------------- Game mechanics -----------------------------------------------------------------------
        
    def makeAccusation(self, player, perp, weapon, room):
        """Player makes an accusation. Returns True if correct, False otherwise."""
        print(f"{player.name} accuses {perp.name} with a {weapon.name} in the {room.name}")
        
        correct = (
            self.solution.get("Suspect") == perp and
            self.solution.get("Weapon") == weapon and
            self.solution.get("Room") == room
        )
        
        if correct:
            print(f"{player.name} has won!")
            if hasattr(player, "store_transition"):
                player.store_transition(1.0, None, True)
            return True
        else:
            if hasattr(player, "store_transition"):
                player.store_transition(-1.0, None, True)
            player.inGame = False
            return False

    
    
    
    def makeSuggestion(self, player, perp, weapon, room):
        """Implements the suggestion mechanism. Returns the responder and the card shown""" 
        print(f"{player.name} suggests: {perp} with {weapon} in {room}")
        
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
            
         
    def gameLoop(self):
        """Main game loop"""   
        self.dealCards()
        while(True):
            self.gameTurn+=1
            print("Turn "+str(self.gameTurn))
            if not (self.checkAllPlayers()):
                break
            
            for player in self.players:
                obs = self.getObservation(player)
                valid_mask = self.getValidActionsMask(player)
                winner = player.playTurn(obs, valid_mask)
                self.turn+=1
                if(winner):
                    return

#---------------------------------- RL API -------------------------------------------------------

def getObservation(self, player):
    """Returns the observation for a given player
       For now, it returns the flattened belief matrix --> later will add turn information
    
    """
    return player.getFlattenedBeliefs()

def getValidActionsMask(self, player):
    """Returns a boolean mask of the valid actions a player can make.
    Length = action space (suggestion + accusations + card reveals)
    """
    from rl.utils import build_action_mask
    return build_action_mask(self, player)

         
    
    
  