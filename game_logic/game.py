import numpy as np
import game_logic.game_constants as gc
from game_logic.card import *
from game_logic.player import Player


class Gameboard:
    def __init__(self, deck):
        """
        Initialize a gameboard with a deck.

        Args:
            deck (Deck): The deck from which to draw cards.
        """
        self.rows = [[] for i in range(gc.NB_ROWS)]
        self.deck = deck

        #Initialize the gameboard with cards
        for i in range(gc.NB_ROWS):
            self.rows[i].append(self.deck.draw())

    def can_add_card(self, card):
        """
        Check if a card can be added to the gameboard.

        Args:
            card (Card): The card to check.
        Returns:
            row_number (int): The number of the row to which the card can be added (-1 if it can't be added).
        """
        gap = gc.NB_CARDS
        row = -1
        for i in range(gc.NB_ROWS):
            temp = card.value - self.rows[i][-1].value
            if 0 < temp < gap:
                gap = temp
                row = i
        return row

    def add_card(self, card):
        """
        Add a card to the gameboard, the card is added to the row with the lowest value gap.

        Args:
            card (Card): The card to add to the gameboard.
        Returns:
            penalty (int): The number of penalty points the player gets for adding the card.
        """
        row = self.can_add_card(card)
        penalty = 0
        if (len(self.rows[row]) < gc.CARDS_PER_ROWS):
            self.rows[row].append(card)
        else:
            for i in range(gc.CARDS_PER_ROWS):
                penalty += self.rows[row].pop().bullheads
            self.rows[row].append(card)
        return penalty

    def replace_row(self, row_number, card):
        """
        Replace a row with a new card.

        Args:
            row_number (int): The number of the row to replace.
            card (Card): The card to replace the row with.

        Returns:
            penalty (int): The number of penalty points the player gets for replacing the row.
        """
        penalty = 0
        for i in range(len(self.rows[row_number])):
            penalty += self.rows[row_number].pop().bullheads
        self.rows[row_number].append(card)
        return penalty

    def afficher(self):
        """
        Return a string representation of the gameboard.
        """
        print("=-----------==-----------==-----------==-----------==-----------==-----------=")
        for row in self.rows:
            for card in row:
                print(str(card), end=" ")
            print()
            print("=-----------==-----------==-----------==-----------==-----------==-----------=")


class Game:
    def __init__(self, players=[]):
        """
        Initialize a game with a list of players.

        Args:
            players (list): The list of players.
        """
        self.deck = Deck()
        self.gameboard = Gameboard(self.deck)
        self.players = players
        for player in self.players:
            self.init_player(player)

    def init_player(self, player):
        # Initialize the players' hand
        for _ in range(gc.NB_TURNS):
            player.add_card_to_hand(self.deck.draw())

    def play(self):
        """
        Play the game.
        """
        for _ in range(gc.NB_TURNS):
            card_of_players = []
            index_of_players = [i for i in range(len(self.players))]

            # Show the gameboard
            self.gameboard.afficher()
            
            for player in self.players:
                # The player choose a card to play
                print(player)
                card_value = None 
                while card_value not in [card.value for card in player.hand]:
                    card_value = player.choose_card()
                    if card_value not in [card.value for card in player.hand]:
                        print("You don't have this card in your hand.")
                    
                card_of_players.append(player.play_card(card_value))

            # Sort the players by card value
            index_of_players = sorted(index_of_players, key=lambda x: card_of_players[x].value)
            
            # Print the card played by the players
            for index in index_of_players:
                player = self.players[index]
                card = card_of_players[index]
                print(str(player.name) + " plays " + str(card))

            # For each player
            for index in index_of_players:
                player = self.players[index]
                card = card_of_players[index]
                row_number = self.gameboard.can_add_card(card)                

                # If the player can add a card to the gameboard
                if row_number != -1:
                    penalty = self.gameboard.add_card(card)

                else:
                    # Else, the player has to choose a row to replace
                    row_number = player.choose_row()
                    penalty = self.gameboard.replace_row(row_number - 1, card)

                player.penalty_points += penalty

        # End of the game
        self.end_game()

    def end_game(self):
        """
        End the game : print the winner and the penalty points of each player.
        """
        # Sort the players by penalty points
        index_of_players = [i for i in range(len(self.players))]
        penalty_points = [player.penalty_points for player in self.players]
        index_of_players = [x for _, x in sorted(zip(penalty_points, index_of_players))]

        # Print the winner and the penalty points of each player
        print("The winner is " + str(self.players[index_of_players[0]]))
        for index in index_of_players:
            player = self.players[index]
            print(player.name + " : " + str(player.penalty_points) + " penalty points")

if __name__ == "__main__":
    players = [Player("Player 1"), Player("Player 2")]
    game = Game(players)
    game.play()
