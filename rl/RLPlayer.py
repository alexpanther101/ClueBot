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
        self.last_turn_log_len = 0
        self.cumulative_episode_reward = 0  # Track total reward
        self.last_suggestion = None  # Track last suggestion made
        
    def get_flattened_belief(self):
        """Override to use the parent's method with proper naming"""
        return np.array(self.getFlattenedBeliefs(), dtype=np.float32)
    
    def store_transition(self, reward, next_obs, done):
        """Store transition in replay buffer if training"""
        if (self.trainer and 
            hasattr(self.trainer, 'replay_buffer') and 
            self.trainer.replay_buffer and 
            self.last_obs is not None and 
            self.last_action is not None):
            
            # Get current valid mask for next state
            from rl.utils import build_action_mask
            valid_mask = build_action_mask(self.game, self)
            
            # Add to cumulative reward
            self.cumulative_episode_reward += reward
            
            self.trainer.replay_buffer.push(
                self.last_obs,
                self.last_action,
                reward,
                next_obs,
                done,
                valid_mask
            )
            
        # Reset for next turn (do this regardless of whether we stored)
        self.last_obs = None
        self.last_action = None
    
    def chooseSuggestion(self):
        """Action selection is now handled by the playTurn method"""
        pass

    def playTurn(self, obs, valid_mask):
        """
        The main method for the RLPlayer's turn.
        It decides on an action (suggestion or accusation) and executes it.
        """
        if not self.inGame:
            return False
        
        # Store the observation and select an action
        self.last_obs = obs
        self.last_action = self.agent.select_action(obs, valid_mask, self.trainer.epsilon)
        
        # Handle invalid action
        if self.last_action < 0 or not valid_mask[self.last_action]:
            print(f"Warning: {self.name} selected invalid action {self.last_action}")
            # Find first valid action as fallback
            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) > 0:
                self.last_action = valid_actions[0]
                print(f"Using fallback action: {self.last_action}")
            else:
                print(f"No valid actions available!")
                return False
        
        # Decode the action
        action_type, payload = decode_action(self.game, self.last_action)

        if action_type == "suggest":
            s_idx, w_idx, r_idx = payload
            suspect_card = self.game.suspectCards[self.game.SUSPECTS[s_idx]]
            weapon_card = self.game.weaponCards[self.game.WEAPONS[w_idx]]
            room_card = self.game.roomCards[self.game.ROOMS[r_idx]]
            
            # Store suggestion for avoiding repetition
            self.last_suggestion = (s_idx, w_idx, r_idx)
            
            # Store state before suggestion for tracking
            old_log_len = len(self.game.suggestionLog)
            
            # Execute the suggestion
            responder, card = self.makeSuggestion(suspect_card, weapon_card, room_card)
            
            # The reward was calculated in GameRules.makeSuggestion using Reward class
            reward = self.game.get_last_reward()
            
            # Store transition
            next_obs = build_observation(self.game, self)
            self.store_transition(reward, next_obs, done=False)
            
            # Update log length tracker
            self.last_turn_log_len = len(self.game.suggestionLog)
            
            return False
            
        elif action_type == "accuse":
            s_idx, w_idx, r_idx = payload
            suspect_card = self.game.suspectCards[self.game.SUSPECTS[s_idx]]
            weapon_card = self.game.weaponCards[self.game.WEAPONS[w_idx]]
            room_card = self.game.roomCards[self.game.ROOMS[r_idx]]
            
            print(f"{self.name} is making an accusation: {suspect_card.name}, {weapon_card.name}, {room_card.name}")
            
            # Execute the accusation
            is_correct = self.makeAccusation(suspect_card, weapon_card, room_card)
            
            # The reward was calculated in GameRules.makeAccusation using Reward class
            reward = self.game.get_last_reward()
            
            # Store final transition
            next_obs = build_observation(self.game, self)
            self.store_transition(reward, next_obs, done=True)
            
            # Return winner name if correct
            if is_correct:
                return self.name
            else:
                # Player is now eliminated (handled in makeAccusation)
                return False
        
        elif action_type == "reveal":
            # This shouldn't happen during playTurn
            print(f"Warning: Reveal action selected during playTurn")
            return False
        
        return False
    
    def revealCard(self, matching_cards):
        """
        Enhanced card reveal logic that considers information value.
        """
        if not matching_cards:
            return None
            
        # If only one option, return it
        if len(matching_cards) == 1:
            return matching_cards[0]
        
        # Strategic card selection based on information theory
        best_card = self.select_best_reveal_card(matching_cards)
        
        # Track revealed card for training
        if hasattr(self.trainer, 'revealed_cards'):
            if self not in self.trainer.revealed_cards:
                self.trainer.revealed_cards[self] = set()
            self.trainer.revealed_cards[self].add(best_card)
        
        # Small chance of random selection for exploration during training
        if (self.trainer and hasattr(self.trainer, 'epsilon') and 
            hasattr(self.trainer, 'total_episodes') and self.trainer.total_episodes < 200):
            import random
            if random.random() < self.trainer.epsilon * 0.2:  # 20% of epsilon early in training
                return random.choice(matching_cards)
        
        return best_card
    
    def select_best_reveal_card(self, matching_cards):
        """
        Select the best card to reveal based on information theory.
        Priority: reveal cards that give away the least strategic information.
        """
        card_scores = {}
        
        for card in matching_cards:
            score = 0.0
            
            # Prefer revealing cards we have multiple of in the same category
            card_type = card.getType()
            same_type_count = sum(1 for c in self.cards if c.getType() == card_type)
            score += same_type_count * 2  # Higher is better
            
            # Prefer revealing cards that have been revealed before
            if hasattr(self.trainer, 'revealed_cards') and self in self.trainer.revealed_cards:
                if card in self.trainer.revealed_cards[self]:
                    score += 10  # Already known, safe to reveal again
            
            # Prefer revealing cards that seem less likely to be in the solution
            solution_prob = self.getProbability("Solution", card)
            score += (1.0 - solution_prob) * 5  # Lower solution probability = higher score
            
            # Avoid revealing cards that would help opponents too much
            # (This is a simplified heuristic)
            if solution_prob < 0.1:  # Very unlikely to be solution
                score += 3
            elif solution_prob > 0.8:  # Very likely solution
                score -= 5
            
            card_scores[card] = score
        
        # Return card with highest score
        best_card = max(card_scores.keys(), key=lambda c: card_scores[c])
        return best_card

    def showCard(self, matching_cards):
        """Deprecated - use revealCard instead"""
        return self.revealCard(matching_cards)
    
    def should_accuse(self):
        """
        Enhanced heuristic for deciding when to allow an accusation.
        """
        try:
            # Never accuse too early
            if self.game.turn < 10:
                return False
            
            # Calculate confidence for each category
            suspect_confidence = self.get_category_confidence('Suspect')
            weapon_confidence = self.get_category_confidence('Weapon')
            room_confidence = self.get_category_confidence('Room')
            
            # Calculate joint confidence
            joint_confidence = suspect_confidence * weapon_confidence * room_confidence
            
            # Progressive confidence threshold based on game progress
            if self.game.turn < 20:
                threshold = 0.9
            elif self.game.turn < 40:
                threshold = 0.8
            elif self.game.turn < 60:
                threshold = 0.7
            else:
                threshold = 0.6  # Desperation mode
            
            # Also check if we have definitive solutions for each category
            definitive_solutions = (
                len(self.possibleSuspects) == 1 and
                len(self.possibleWeapons) == 1 and
                len(self.possibleRooms) == 1
            )
            
            return joint_confidence > threshold or definitive_solutions
        
        except Exception as e:
            print(f"Error in should_accuse: {e}")
            # Fallback: allow after many turns
            return self.game.turn > 50
    
    def get_category_confidence(self, card_type):
        """
        Calculate confidence for a specific card category (Suspect, Weapon, Room).
        """
        if card_type == 'Suspect':
            possible_cards = self.possibleSuspects
        elif card_type == 'Weapon':
            possible_cards = self.possibleWeapons
        elif card_type == 'Room':
            possible_cards = self.possibleRooms
        else:
            return 0.0
        
        if not possible_cards:
            return 0.0
        
        # Get the maximum solution probability in this category
        max_prob = max(self.getProbability("Solution", card) for card in possible_cards)
        return max_prob
    
    def get_best_suggestion_cards(self):
        """
        Get the best cards to suggest based on information theory.
        Returns cards that are most likely to provide new information.
        """
        best_suspect = self.get_highest_entropy_card(self.possibleSuspects)
        best_weapon = self.get_highest_entropy_card(self.possibleWeapons)
        best_room = self.get_highest_entropy_card(self.possibleRooms)
        
        return best_suspect, best_weapon, best_room
    
    def get_highest_entropy_card(self, card_list):
        """
        Get the card with highest entropy (uncertainty) for maximum information gain.
        """
        if not card_list:
            return None
        
        best_card = None
        max_entropy = -1
        
        for card in card_list:
            entropy = self.calculate_card_entropy(card)
            if entropy > max_entropy:
                max_entropy = entropy
                best_card = card
        
        return best_card if best_card else card_list[0]
    
    def calculate_card_entropy(self, card):
        """
        Calculate the entropy for a specific card across all possible owners.
        Higher entropy means more uncertainty = more information to gain.
        """
        import math
        entropy = 0.0
        
        for owner in self.owners:
            prob = self.getProbability(owner, card)
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy