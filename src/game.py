"""
6 qui Prend - Core Game Logic
Base game classes and simple agents
"""

import numpy as np
import random

# Constants
NB_TURNS = 10
NB_CARDS = 104
NB_ROWS = 4
CARDS_PER_ROWS = 5
NB_PLAYERS = 2


class Card:
    def __init__(self, value):
        assert 1 <= value <= NB_CARDS
        self.value = value
        if value % 55 == 0:
            self.bullheads = 7
        elif value % 11 == 0:
            self.bullheads = 5
        elif value % 10 == 0:
            self.bullheads = 3
        elif value % 10 == 5:
            self.bullheads = 2
        else:
            self.bullheads = 1
    
    def __str__(self):
        return f" |{self.value:3d}(*{self.bullheads}*)| "


class Deck:
    def __init__(self):
        self.cards = [Card(i) for i in range(1, NB_CARDS + 1)]
        np.random.shuffle(self.cards)

    def draw(self):
        assert len(self.cards) > 0
        return self.cards.pop()


class Gameboard:
    def __init__(self, deck):
        self.deck = deck
        self.board = [[deck.draw()] for _ in range(NB_ROWS)]
    
    def clear_row(self, row):
        bullheads = sum([card.bullheads for card in self.board[row][:-1]])
        self.board[row] = [self.board[row][-1]]
        return bullheads
    
    def can_play_card(self, card):
        return any([card.value > row[-1].value for row in self.board])
    
    def play_card(self, card):
        assert self.can_play_card(card)
        row = max((i for i in range(NB_ROWS) if self.board[i][-1].value < card.value), 
                  key=lambda i: self.board[i][-1].value)
        self.board[row].append(card)
        bullheads = 0
        if len(self.board[row]) > CARDS_PER_ROWS:
            bullheads = self.clear_row(row)
        return bullheads

    def replace_row(self, card, row):
        assert not self.can_play_card(card)
        self.board[row].append(card)
        bullheads = self.clear_row(row)
        return bullheads

    def __str__(self):
        return "\n".join([
            "=-----------=" * CARDS_PER_ROWS + "\n" + \
            " ".join([str(card) for card in row]) for row in self.board]) \
            + "\n" + "=-----------=" * CARDS_PER_ROWS


class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.bullheads = 0
    
    def choose_card(self, gameboard, all_played_cards):
        print(self)
        try:
            card = int(input(f"{self.name}, choose a card: "))
            assert card in [c.value for c in self.hand], "Card not in hand"
        except:
            print("Please choose a valid card")
            return self.choose_card(gameboard, all_played_cards)
        return Card(card)
    
    def choose_row(self, gameboard, all_played_cards):
        try:
            row = int(input(f"{self.name}, choose a row: "))
            assert 1 <= row <= NB_ROWS, "Row not in range"
        except:
            print("Please choose a valid row")
            return self.choose_row(gameboard, all_played_cards)
        return row

    def __str__(self):
        return f"{self.name}: {','.join(str(card) for card in self.hand)}"


class RandomAgent(Player):
    """Random agent that selects cards uniformly at random"""
    def __init__(self, name="RandomAgent"):
        super().__init__(name)
    
    def choose_card(self, gameboard, all_played_cards):
        return random.choice(self.hand)
    
    def choose_row(self, gameboard, all_played_cards):
        return random.randint(0, NB_ROWS - 1)


class GreedyAgent(Player):
    """Greedy agent that tries to minimize immediate bullhead penalty"""
    def __init__(self, name="GreedyAgent"):
        super().__init__(name)
    
    def choose_card(self, gameboard, all_played_cards):
        # Try to find a card that can be placed safely (won't cause row to overflow)
        best_card = None
        min_penalty = float('inf')
        
        for card in self.hand:
            penalty = self._estimate_penalty(card, gameboard)
            if penalty < min_penalty:
                min_penalty = penalty
                best_card = card
        
        return best_card if best_card else self.hand[0]
    
    def _estimate_penalty(self, card, gameboard):
        """Estimate the penalty for playing a card"""
        # Check if card can be placed
        placeable_rows = [(i, row) for i, row in enumerate(gameboard.board) 
                         if row[-1].value < card.value]
        
        if not placeable_rows:
            # Card will force row replacement - find row with minimum penalty
            return min(sum(c.bullheads for c in row) for row in gameboard.board)
        
        # Find the row where card would be placed
        target_row_idx = max(placeable_rows, key=lambda x: x[1][-1].value)[0]
        target_row = gameboard.board[target_row_idx]
        
        # Check if this would cause overflow
        if len(target_row) >= CARDS_PER_ROWS:
            return sum(c.bullheads for c in target_row)
        
        return 0  # No immediate penalty
    
    def choose_row(self, gameboard, all_played_cards):
        # Choose row with minimum bullheads
        min_bullheads = float('inf')
        best_row = 0
        
        for i, row in enumerate(gameboard.board):
            bullheads = sum(card.bullheads for card in row)
            if bullheads < min_bullheads:
                min_bullheads = bullheads
                best_row = i
        
        return best_row


class Game:
    def __init__(self, players, display=True):
        self.display = display
        # Initialize the deck and gameboard
        self.deck = Deck()
        self.gameboard = Gameboard(self.deck)
        self.players = players
        self.all_played_cards = []
        # Initialize the players
        self.init_players()
        
        # For RL training: track rewards per turn
        self.turn_rewards = {player: [] for player in players}

    def init_players(self):
        for player in self.players:
            player.hand = []
            player.bullheads = 0
            for _ in range(NB_TURNS):
                player.hand.append(self.deck.draw())
            player.hand.sort(key=lambda card: card.value)
    
    def get_cards(self):
        chosen_cards = []
        for player in self.players:
            card = player.choose_card(self.gameboard, self.all_played_cards)
            chosen_cards.append((player, card))
            player.hand.remove(card)
        return chosen_cards    
    
    def play_card(self, player, card):
        row = 0
        bullheads_before = player.bullheads
        
        if self.gameboard.can_play_card(card):
            bullheads = self.gameboard.play_card(card)
            player.bullheads += bullheads
        else:
            if self.display:
                print(f"{player.name} choose a row to replace")
            row = player.choose_row(self.gameboard, self.all_played_cards)
            bullheads = self.gameboard.replace_row(card, row)
            player.bullheads += bullheads
        
        if self.display:
            print(f"{player.name} got {bullheads} bullheads")
        
        # Track reward for RL agents (negative bullheads collected this turn)
        reward = -(player.bullheads - bullheads_before)
        self.turn_rewards[player].append(reward)
        
        # Store reward in PPO agent's memory if applicable
        if hasattr(player, 'store_reward'):
            player.store_reward(reward, done=False)
        
        return row

    def turn(self):      
        # Get the cards played by each player
        actions = self.get_cards()
        actions.sort(key=lambda x: x[1].value)            
        
        # Replace a row if the lowest card cannot be placed next in the row
        for player, card in actions.copy():
            row = self.play_card(player, card)
            actions[actions.index((player, card))] = (player, card, row)
        
        # Add the played cards to the list of all played cards
        self.all_played_cards.extend([card for _, card, _ in actions])
        
        return actions

    def play(self):
        for _ in range(NB_TURNS):
            if self.display:
                print(self.gameboard)
            self.turn()
        
        if self.display:
            # Determine the winner (player with the fewest bullheads)
            winner = min(self.players, key=lambda player: player.bullheads)
            print(f"The winner is {winner.name} with {winner.bullheads} bullheads")
            
            # Print the bullheads for each player
            for player in self.players:
                print(f"{player.name} got {player.bullheads} bullheads")
        
        # Return game results
        return {player: player.bullheads for player in self.players}


if __name__ == "__main__":
    # Demo game
    print("="*60)
    print("6 qui Prend! - Demo Game")
    print("="*60)
    
    random_agent = RandomAgent("Random")
    greedy_agent = GreedyAgent("Greedy")
    
    game = Game([random_agent, greedy_agent], display=True)
    results = game.play()
    
    print("\nFinal Results:")
    for player, bullheads in results.items():
        print(f"  {player.name}: {bullheads} bullheads")
