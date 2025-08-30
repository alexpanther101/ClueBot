from ClueBasics.Player import Player

class HumanPlayer(Player):
    def __init__(self, name, game, type):
        super().__init__(name, game, type)
        self.game.hasHuman = True
        
    def chooseCard(self, all_cards):
        if(len(all_cards) == 1):
            return all_cards[0]
        
        for i, card in enumerate(all_cards):
            print(f"{i + 1}. {card.name}")
        while True:
            try:
                choice = int(input("Enter the number of your choice: ")) - 1
                if 0 <= choice < len(all_cards):
                    return all_cards[choice]
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a valid number.")

    def chooseSuggestion(self):
        print(f"\n{self.name}, make a suggestion.")
        print("Choose a suspect:")
        suspect = self.chooseCard(list(self.game.suspectCards.values()))
        print("Choose a weapon:")
        weapon = self.chooseCard(list(self.game.weaponCards.values()))
        print("Choose a room:")
        room = self.chooseCard(list(self.game.roomCards.values()))
        return suspect, weapon, room

    def chooseAccusation(self):
        print(f"\n{self.name}, make an accusation.")
        print("Choose a suspect:")
        suspect = self.chooseCard( list(self.game.suspectCards.values()))
        print("Choose a weapon:")
        weapon = self.chooseCard(list(self.game.weaponCards.values()))
        print("Choose a room:")
        room = self.chooseCard(list(self.game.roomCards.values()))
        return suspect, weapon, room

    def refuteSuggestion(self, suggestionCards):
        print("\nYou have been asked to refute the following suggestion:")
        for card in suggestionCards:
            print(f" - {card.name}")
        
        matching_cards = [card for card in self.cards if card in suggestionCards]
        if not matching_cards:
            print("You cannot refute this suggestion.\n")
            return None
        
        print("\nYou may choose one of your cards to show:")
        chosen = self.chooseCard(matching_cards)
        return chosen

    def revealCard(self, matching_cards):
        print("\nYou may choose one of your cards to show:")
        chosen = self.chooseCard(matching_cards)
        return chosen

    def playTurn(self, obs, valid_mask):
        print(f"\n===== {self.name}'s Turn =====")
        print("Your cards:")
        for card in self.cards:
            print(f" - {card.name}")

        
        suspect, weapon, room = self.chooseSuggestion()
        owner, card = self.makeSuggestion(suspect, weapon, room)
        if owner and card:
            print(f"{owner} showed you the card: {card.name}")
        else:
            print("No one could disprove your suggestion.")
        
        choice = input("Would you like to accuse? Type (Y) or (N) ").strip().lower()
        
        while True:    
            if choice == 'y':
                suspect, weapon, room = self.chooseAccusation()
                correct = self.makeAccusation(suspect, weapon, room)
                if correct:
                    print("Your accusation was CORRECT. You win!")
                    return self.name
                else:
                    print("Your accusation was wrong. You're out of the game.")
                    self.inGame = False
                break
            elif choice == 'n':
                break
            else:
                choice = input("Please Type (Y) or (N)")

    def updateBeliefs(self):
        print("\n[Optional] Belief update not implemented. You may track manually.")
