#subclass of your Player that acts as the interface between agent and game (calls DQNAgent.select_action, handles reveal decisions, passes rewards to trainer)

from ClueBasics import Player
class RLPlayer(Player):
    def choose_reveal_card(self, matching_cards):
        # RL chooses reveal via action index from DQN
        obs = self.game.get_observation_for(self)
        valid_mask = self.game.get_valid_action_mask(self)
        action = self.agent.select_action(obs, valid_mask)
        
        # Map action back to card choice (rl/utils.py handles this)
        from rl.utils import decode_reveal_action
        return decode_reveal_action(action, matching_cards)
