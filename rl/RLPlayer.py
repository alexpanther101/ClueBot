#subclass of your Player that acts as the interface between agent and game (calls DQNAgent.select_action, handles reveal decisions, passes rewards to trainer)

from ClueBasics.Player import Player
import numpy as np
from rl.utils import build_observation, decode_action, get_card_from_action_idx

class RLPlayer(Player):
    def __init__(self, name, game, type="RL"):
        super().__init__(name, game, type)
        self.agent = None  # Will be set by trainer
        self.trainer = None  # Will be set by trainer
        self.last_obs = None
        self.last_action = None
        self.last_turn_log_len = 0 # To track new suggestions for reward
        
    def get_flattened_belief(self):
        """Override to use the parent's method with proper naming"""
        return np.array(self.getFlattenedBeliefs(), dtype=np.float32)
    
    def store_transition(self, reward, next_obs, done):
        """Store transition in replay buffer if training"""
        if (self.trainer and self.trainer.replay_buffer and 
            self.last_obs is not None and self.last_action is not None):
            
            # Get current valid mask
            from rl.utils import build_action_mask
            valid_mask = build_action_mask(self.game, self)
            
            self.trainer.replay_buffer.push(
                self.last_obs,
                self.last_action,
                reward,
                next_obs,
                done,
                valid_mask
            )
            # Reset for next turn
            self.last_obs = None
            self.last_action = None
    
    def chooseSuggestion(self):
        """
        Action selection is now handled by the playTurn method,
        so this method is no longer needed.
        """
        pass

    # --- NEW METHODS FOR RL LOGIC ---

    def playTurn(self, obs, valid_mask):
        """
        The main method for the RLPlayer's turn.
        It decides on an action (suggestion or accusation) and executes it.
        """
        # Store the observation and select an action
        self.last_obs = obs
        self.last_action = self.agent.select_action(obs, valid_mask, self.trainer.epsilon)
        
        # Decode the action
        action_type, payload = decode_action(self.game, self.last_action)

        if action_type == "suggest":
            s_idx, w_idx, r_idx = payload
            suspect = self.game.SUSPECTS[s_idx]
            weapon = self.game.WEAPONS[w_idx]
            room = self.game.ROOMS[r_idx]
            
            # Execute the suggestion
            # The makeSuggestion call will handle player responses and return a reward
            responder, card = self.makeSuggestion(suspect, weapon, room)

            # Check if a card was revealed, which is a key learning signal.
            # We provide a small reward for a successful suggestion.
            reward = self.game.calculate_suggestion_reward(responder, self, self.last_turn_log_len)
            self.last_turn_log_len = len(self.game.suggestionLog)
            
            # The game loop will pass the new state to us, so we don't store the transition here.
            
        elif action_type == "accuse":
            s_idx, w_idx, r_idx = payload
            suspect_card = self.game.suspectCards[self.game.SUSPECTS[s_idx]]
            weapon_card = self.game.weaponCards[self.game.WEAPONS[w_idx]]
            room_card = self.game.roomCards[self.game.ROOMS[r_idx]]
            
            # Execute the accusation
            is_correct_accusation = self.makeAccusation(suspect_card, weapon_card, room_card)
            
            # The game rules will calculate the final reward for an accusation
            reward = self.game.calculate_accusation_reward(is_correct_accusation)
            
            # This is a terminal action, so we store the transition immediately.
            next_obs = build_observation(self.game, self)
            self.store_transition(reward, next_obs, done=True)
            
            # Return the winner for the game loop to handle
            if is_correct_accusation:
                return self.name
            else:
                return False
        
        return False
        
    def revealCard(self, matching_cards):
        """
        Now uses the agent's policy to select which card to reveal.
        It must be one of the matching cards.
        """
        # Build a temporary valid mask for only the reveal actions
        valid_mask = np.zeros(self.agent.output_dim, dtype=bool)
        matching_card_indices = [card.global_index for card in matching_cards]
        
        # We assume the reveal actions are at the end of the action space
        # This requires updates to `utils.py` and the `DQNAgent`'s output dimension
        from rl.utils import reveal_index_from_card
        for card in matching_cards:
            reveal_idx = reveal_index_from_card(self.game, card)
            valid_mask[reveal_idx] = True

        # Select the reveal action using the agent's policy
        # A new observation can be passed if needed, but for simplicity, we use the current one
        obs = build_observation(self.game, self)
        reveal_action_idx = self.agent.select_action(obs, valid_mask, epsilon=0) # No exploration for reveals
        
        # Decode the action to get the card
        revealed_card = get_card_from_action_idx(self.game, reveal_action_idx)

        # The reveal action itself could also be part of the training loop
        # We can calculate a small reward for a strategic reveal
        reward = self.game.calculate_reveal_reward(revealed_card)
        self.store_transition(reward, obs, done=False) # store the transition
        
        return revealed_card

    # Deprecated function from parent class. We will now handle this with revealCard.
    def showCard(self, matching_cards):
        return self.revealCard(matching_cards)
    
    def should_accuse(self):
        """Simple heuristic for deciding when to allow an accusation."""
        try:
            suspect_prob = max(self.getProbability("Solution", card) 
                            for card in self.game.suspectCards.values())
            weapon_prob = max(self.getProbability("Solution", card) 
                            for card in self.game.weaponCards.values())
            room_prob = max(self.getProbability("Solution", card) 
                        for card in self.game.roomCards.values())
            return suspect_prob * weapon_prob * room_prob > 0.8
        except:
            return True  # Allow accusations if probability calculation fails