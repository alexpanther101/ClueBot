import math

class Reward:
    def __init__(self, game_rules):
        self.game = game_rules
        self.previous_entropy = {}  # Track entropy reduction per player
        self.previous_confidences = {}  # Track confidence improvements
        self.suggestion_history = {}  # Track suggestion patterns
        
    def calculate_suggestion_reward(self, suggester, responder, card_shown, 
                                   skipped_players, suggestion_cards):
        """
        Calculate reward for a suggestion with improved signal quality.
        """
        reward = 0.0
        
        # Base rewards for information gain
        if responder and responder != suggester:
            if card_shown:
                # Direct information gain - base reward
                reward += 15.0
                
                # Bonus for high-uncertainty elimination
                entropy_before = self._calculate_card_entropy(suggester, card_shown)
                if entropy_before > 1.8:  # High uncertainty
                    reward += 8.0
                elif entropy_before > 1.2:  # Medium uncertainty
                    reward += 4.0
                else:
                    reward += 1.0  # Low uncertainty but still info
                
            else:
                # Ambiguous response still provides some info
                reward += 5.0
        
        # Significant reward for eliminating possibilities via skipped players
        info_gain_from_skips = len(skipped_players) * len(suggestion_cards)
        reward += info_gain_from_skips * 1.0
        
        # Strategic suggestion bonus
        strategy_bonus = self._calculate_strategic_bonus(suggester, suggestion_cards)
        reward += strategy_bonus
        
        # Penalty for redundant or poor suggestions
        redundancy_penalty = self._calculate_redundancy_penalty(suggester, suggestion_cards)
        reward -= redundancy_penalty
        
        # Bonus for diverse suggestions (avoid repetition)
        diversity_bonus = self._calculate_diversity_bonus(suggester, suggestion_cards)
        reward += diversity_bonus
        
        # Progress toward solution bonus
        solution_progress = self._calculate_solution_progress_bonus(suggester)
        reward += solution_progress
        
        return reward
    
    def calculate_accusation_reward(self, is_correct, player, confidence):
        """
        Calculate reward for an accusation with improved timing incentives.
        """
        if is_correct:
            # Base win reward
            base_reward = 200.0
            
            # Efficiency bonus (encourage faster wins)
            turn_bonus = max(0, 100 * (1 - self.game.turn / 150))
            
            # Confidence bonus (reward accurate probability assessment)
            if confidence > 0.95:
                confidence_bonus = 50.0
            elif confidence > 0.85:
                confidence_bonus = 30.0
            elif confidence > 0.75:
                confidence_bonus = 15.0
            else:
                confidence_bonus = 5.0
            
            total_reward = base_reward + turn_bonus + confidence_bonus
            
        else:
            # Wrong accusation penalty
            base_penalty = -150.0
            
            # Severity based on confidence and timing
            if confidence < 0.3:
                # Very reckless accusation
                confidence_penalty = -100.0
            elif confidence < 0.5:
                # Somewhat reckless
                confidence_penalty = -50.0
            elif confidence < 0.7:
                # Reasonable attempt
                confidence_penalty = -20.0
            else:
                # High confidence but wrong (bad luck or faulty reasoning)
                confidence_penalty = -10.0
            
            # Early wrong accusations are worse
            if self.game.turn < 20:
                timing_penalty = -50.0
            elif self.game.turn < 40:
                timing_penalty = -25.0
            else:
                timing_penalty = 0.0
            
            total_reward = base_penalty + confidence_penalty + timing_penalty
            
        return total_reward
    
    def calculate_reveal_reward(self, revealer, revealed_card, matching_cards):
        """
        Calculate reward for revealing a card when refuting.
        """
        reward = 0.0
        
        # Base reward for cooperation
        reward += 2.0
        
        # Strategic reveal bonus
        if len(matching_cards) > 1:
            # Had choice, evaluate the strategic value
            card_value = self._evaluate_card_reveal_value(revealer, revealed_card)
            reward += card_value
        else:
            # No choice, small bonus for compliance
            reward += 1.0
            
        # Prefer revealing cards with lower solution probability
        solution_prob = revealer.getProbability("Solution", revealed_card)
        if solution_prob < 0.2:
            reward += 3.0  # Safe reveal
        elif solution_prob > 0.8:
            reward -= 2.0  # Risky reveal
            
        return reward
    
    def calculate_step_reward(self, player):
        """
        Calculate intermediate rewards for learning progress.
        """
        reward = 0.0
        
        # Reward entropy reduction (information gain)
        current_entropy = self._calculate_total_entropy(player)
        player_id = id(player)
        
        if player_id in self.previous_entropy:
            entropy_reduction = self.previous_entropy[player_id] - current_entropy
            if entropy_reduction > 0:
                reward += entropy_reduction * 3.0
        
        self.previous_entropy[player_id] = current_entropy
        
        # Reward for increasing solution confidence
        current_confidence = self._calculate_solution_certainty(player)
        
        if player_id in self.previous_confidences:
            confidence_gain = current_confidence - self.previous_confidences[player_id]
            if confidence_gain > 0:
                reward += confidence_gain * 10.0
        
        self.previous_confidences[player_id] = current_confidence
        
        # Bonus for narrowing possibilities
        total_possibilities = (len(player.possibleSuspects) + 
                             len(player.possibleWeapons) + 
                             len(player.possibleRooms))
        
        if total_possibilities <= 6:  # Getting close to solution
            reward += 2.0
        elif total_possibilities <= 9:  # Making good progress
            reward += 1.0
            
        return reward
    
    def _calculate_card_entropy(self, player, card):
        """Calculate entropy for a specific card's ownership distribution"""
        entropy = 0.0
        for owner in player.owners:
            prob = player.getProbability(owner, card)
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def _calculate_total_entropy(self, player):
        """Calculate total entropy across all cards"""
        total = 0.0
        for card in self.game.cards:
            total += self._calculate_card_entropy(player, card)
        return total
    
    def _calculate_solution_certainty(self, player):
        """Calculate how certain the player is about the solution"""
        certainties = []
        
        # Get max probability for each category
        for cards in [player.possibleSuspects, player.possibleWeapons, player.possibleRooms]:
            if cards:
                max_prob = max(player.getProbability("Solution", card) for card in cards)
                certainties.append(max_prob)
            else:
                certainties.append(0)
        
        # Return average certainty (more stable than product)
        return sum(certainties) / len(certainties) if certainties else 0
    
    def _calculate_strategic_bonus(self, player, suggestion_cards):
        """Reward strategic suggestions that maximize expected information gain"""
        bonus = 0.0
        
        # Reward suggestions about high-entropy (uncertain) cards
        total_entropy = 0.0
        for card in suggestion_cards:
            entropy = self._calculate_card_entropy(player, card)
            total_entropy += entropy
            
            if entropy > 2.0:
                bonus += 3.0  # Very uncertain card
            elif entropy > 1.5:
                bonus += 2.0  # Moderately uncertain
            elif entropy > 1.0:
                bonus += 1.0  # Somewhat uncertain
                
        # Average entropy bonus
        avg_entropy = total_entropy / len(suggestion_cards)
        if avg_entropy > 1.8:
            bonus += 5.0
            
        # Reward testing likely solution combinations
        solution_probs = [player.getProbability("Solution", card) 
                         for card in suggestion_cards]
        
        if all(p > 0.4 for p in solution_probs):
            bonus += 4.0  # Testing probable solution
        elif all(p > 0.2 for p in solution_probs):
            bonus += 2.0  # Testing possible solution
            
        return bonus
    
    def _calculate_redundancy_penalty(self, player, suggestion_cards):
        """Penalize asking about cards we already have definitive knowledge about"""
        penalty = 0.0
        
        known_cards = 0
        for card in suggestion_cards:
            # Check if we know who has this card
            for owner in player.owners:
                if owner != "Solution" and player.getProbability(owner, card) >= 0.99:
                    known_cards += 1
                    penalty += 5.0  # Heavy penalty for known cards
                    break
                    
            # Lighter penalty if we know it's not in solution
            if player.getProbability("Solution", card) <= 0.01:
                penalty += 2.0
                
        # Extra penalty if too many cards are already known
        if known_cards >= 2:
            penalty += 10.0  # Waste of a turn
            
        return penalty
    
    def _calculate_diversity_bonus(self, player, suggestion_cards):
        """Bonus for making diverse suggestions (avoiding repetition)"""
        player_id = id(player)
        
        if player_id not in self.suggestion_history:
            self.suggestion_history[player_id] = []
        
        # Convert suggestion to a comparable format
        suggestion_key = tuple(sorted(card.name for card in suggestion_cards))
        
        # Check how recently this suggestion was made
        history = self.suggestion_history[player_id]
        
        bonus = 0.0
        if suggestion_key not in history[-5:]:  # Not in recent 5 suggestions
            bonus += 3.0
        elif suggestion_key not in history[-10:]:  # Not in recent 10 suggestions
            bonus += 1.0
        else:
            # Repeated suggestion - penalty
            bonus -= 5.0
        
        # Add to history
        history.append(suggestion_key)
        if len(history) > 20:  # Keep limited history
            history.pop(0)
            
        return bonus
    
    def _calculate_solution_progress_bonus(self, player):
        """Bonus for making progress toward identifying the solution"""
        bonus = 0.0
        
        # Bonus for having definitive knowledge in any category
        if len(player.possibleSuspects) == 1:
            bonus += 5.0
        if len(player.possibleWeapons) == 1:
            bonus += 5.0
        if len(player.possibleRooms) == 1:
            bonus += 5.0
            
        # Bonus for narrowing down possibilities
        total_remaining = (len(player.possibleSuspects) + 
                          len(player.possibleWeapons) + 
                          len(player.possibleRooms))
        
        if total_remaining <= 3:  # Very close to solution
            bonus += 10.0
        elif total_remaining <= 6:  # Making good progress
            bonus += 5.0
        elif total_remaining <= 9:  # Some progress
            bonus += 2.0
            
        return bonus
    
    def _evaluate_card_reveal_value(self, revealer, revealed_card):
        """Evaluate the strategic value of revealing a specific card"""
        value = 0.0
        
        # Prefer revealing cards that are less valuable strategically
        solution_prob = revealer.getProbability("Solution", revealed_card)
        
        if solution_prob < 0.1:
            value += 3.0  # Safe to reveal
        elif solution_prob < 0.3:
            value += 1.0  # Reasonably safe
        elif solution_prob > 0.7:
            value -= 3.0  # Risky to reveal
            
        # Prefer revealing cards from categories where we have more information
        card_type = revealed_card.getType()
        same_type_owned = sum(1 for c in revealer.cards if c.getType() == card_type)
        
        if same_type_owned > 1:
            value += 2.0  # Have other cards in this category
            
        return value