Phase 1: Core Game Simulation
🎯 Goal: Build a functional virtual version of Clue that bots can play

1. Game Setup

    Implement GameRules class

    Define Player class (can be bot or human)

    Randomly select solution cards (1 suspect, 1 weapon, 1 room)

    Shuffle and deal the rest to players

    Game Loop

        Turn-based structure

        On each turn, allow:

        Suggestion (suspect, weapon, room)

    Response (next player shows one matching card if any)

    Card Visibility

        Track cards in each player’s hand

        Track which card is shown to which player

        Add a suggestion log: who suggested what, who showed what (or didn’t)

    Game End Condition

        Correct accusation → win



🔍 Phase 2: Information Tracking & Belief Modeling
🎯 Goal: Give bots memory and reasoning capabilities

2. History

        Log every suggestion (who suggested, what cards, who responded, what was shown)

        Each player gets their own copy of visible history

___________________________________________________________________________________DONE____________________________________________________________

    Card Likelihood Matrix

        For each player and each card, store a probability they have it:

    Responses (or lack thereof)

🎓 Phase 3: Bot Architecture
🎯 Goal: Add logic-based and learning-based bots

 1) TriggerHappyBot - Accuse when no one shows
 2) EliminationBot - Accuse when its can cross off through its suggestions
 3) HeuristicsBot - Suggest based on a belief matrix, tracking cards shown indirectly, setting people that dont have certain cards to 0, and when a card is shown, which of the three guessed - probabilities 
        Inference
        Uncertain-first strategy?
        
4) Bluff-based - Throws off on purpose by bluffing own cards and tries to actively reinforce others' mistaken beliefs
5) MirroBot - Mirrors others suggestions to confuse ever so often
6) Policy switching bot 
Learning Bot

Wrap game logic in an RL-compatible interface:

get_state()

get_legal_actions()

step(action)

get_reward()

🧠 Phase 4: Reinforcement Learning Integration
🎯 Goal: Train bots to improve strategy over time

Define State Representation

Your hand

Cards you've seen

Belief matrix

Suggestion history

Turn number or phase

Define Action Space

Suggestions: all combinations of suspect/weapon/room

Card Risk Levels - Sort cards by priority and reveal the ones that are less important or less visible

Opponent belief matrix - where do they think that card is

Bluffing (guess own cards in suggestion)

Information masking - dont make conclusions obvious (related to other's beliefs and suggestion log  )

Define Reward Signal

Win = +1

Loss = 0 or -1

Bonus: Reward inference accuracy or information gain

Implement RL Agent

Start with DQN, PPO, or policy gradient

Train via self-play

♟ Phase 5: Game Theory & Advanced Inference
🎯 Goal: Add strategic, opponent-aware intelligence

Counterfactual Regret Minimization (CFR)

Model Clue as a multi-agent extensive-form game

Implement CFR on top of simulator

Track tendencies: Do they bluff? Reuse same suggestions?



