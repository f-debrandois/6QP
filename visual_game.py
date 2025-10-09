"""
6 qui Prend - Visual Game with Tkinter
Play against AI agents with a graphical interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sys
from pathlib import Path
import threading

# Add src and scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from game import Player, RandomAgent, GreedyAgent, Game, Card, NB_ROWS # type: ignore
from train_ppo import PPOAgent # type: ignore
from train_reinforce import RLAgent # type: ignore


class HumanPlayer(Player):
    """Human player that uses GUI for card selection"""
    def __init__(self, name="Human", gui_callback=None):
        super().__init__(name)
        self.gui_callback = gui_callback
        self.selected_card = None
        self.selected_row = None
        self.waiting = False
        
    def choose_card(self, gameboard, all_played_cards):
        """Wait for GUI to provide card selection"""
        self.waiting = True
        self.selected_card = None
        
        # Trigger GUI to request card selection
        if self.gui_callback:
            self.gui_callback('choose_card', gameboard, all_played_cards, self.hand)
        
        # Wait for selection (blocking)
        while self.selected_card is None:
            pass  # In real implementation, this will be event-driven
            
        self.waiting = False
        return self.selected_card
    
    def choose_row(self, gameboard, all_played_cards):
        """Wait for GUI to provide row selection"""
        self.waiting = True
        self.selected_row = None
        
        # Trigger GUI to request row selection
        if self.gui_callback:
            self.gui_callback('choose_row', gameboard, all_played_cards, None)
        
        # Wait for selection
        while self.selected_row is None:
            pass
            
        self.waiting = False
        return self.selected_row


class VisualGame:
    def __init__(self, root):
        self.root = root
        self.root.title("6 qui Prend!")
        self.root.geometry("1400x900")
        
        # Paths
        self.project_dir = Path(__file__).parent
        self.images_dir = self.project_dir / 'images'
        self.cards_dir = self.images_dir / 'cards'
        
        # Game state
        self.game = None
        self.human_player = None
        self.ai_player = None
        self.game_thread = None
        self.game_active = False
        self.waiting_for_input = False
        self.input_type = None  # 'card' or 'row'
        
        # Card images cache
        self.card_images = {}
        self.load_card_images()
        
        # GUI elements
        self.selected_card_index = None
        self.card_buttons = []
        self.row_buttons = []
        
        # Show welcome screen
        self.show_welcome_screen()
    
    def load_card_images(self):
        """Pre-load card images"""
        try:
            # Load card backs and special cards
            for img_name in ['backside.png', 'empty_card.png', 'last_card.png']:
                img_path = self.cards_dir / img_name
                if img_path.exists():
                    img = Image.open(img_path)
                    img = img.resize((80, 120), Image.Resampling.LANCZOS)
                    self.card_images[img_name] = ImageTk.PhotoImage(img)
            
            # Load numbered cards (1-104)
            for i in range(1, 105):
                img_path = self.cards_dir / f"{i}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    img = img.resize((80, 120), Image.Resampling.LANCZOS)
                    self.card_images[i] = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Warning: Could not load card images: {e}")
    
    def show_welcome_screen(self):
        """Show welcome screen with opponent selection"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="6 qui Prend!", 
                               font=('Helvetica', 36, 'bold'))
        title_label.pack(pady=20)
        
        # Welcome image (if available)
        welcome_img_path = self.images_dir / 'welcome.png'
        if welcome_img_path.exists():
            try:
                img = Image.open(welcome_img_path)
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(main_frame, image=photo)
                img_label.image = photo  # Keep reference
                img_label.pack(pady=20)
            except Exception as e:
                print(f"Could not load welcome image: {e}")
        
        # Instructions
        ttk.Label(main_frame, text="Select your opponent:", 
                 font=('Helvetica', 16)).pack(pady=10)
        
        # Opponent selection frame
        opponent_frame = ttk.LabelFrame(main_frame, text="Choose Opponent", padding=20)
        opponent_frame.pack(pady=20)
        
        # Random opponent
        ttk.Button(opponent_frame, text="🎲 Random Agent (Easy)", 
                  command=lambda: self.start_game('random'),
                  width=30).pack(pady=5)
        
        # Greedy opponent
        ttk.Button(opponent_frame, text="🧠 Greedy Agent (Medium)", 
                  command=lambda: self.start_game('greedy'),
                  width=30).pack(pady=5)
        
        # PPO opponent (if model exists)
        ppo_model = self.project_dir / 'saved_models' / 'ppo_agent.pth'
        if ppo_model.exists():
            ttk.Button(opponent_frame, text="🤖 PPO Agent (Hard)", 
                      command=lambda: self.start_game('ppo'),
                      width=30).pack(pady=5)
        
        # REINFORCE opponent (if model exists)
        rf_model = self.project_dir / 'saved_models' / 'reinforce_agent.pth'
        if rf_model.exists():
            ttk.Button(opponent_frame, text="🎯 REINFORCE Agent (Hard)", 
                      command=lambda: self.start_game('reinforce'),
                      width=30).pack(pady=5)
        
        # Quit button
        ttk.Button(main_frame, text="Quit", 
                  command=self.root.quit).pack(pady=20)
    
    def start_game(self, opponent_type):
        """Initialize and start a new game"""
        # Create players
        self.human_player = HumanPlayer("You", gui_callback=self.handle_game_event)
        
        if opponent_type == 'random':
            self.ai_player = RandomAgent("Random Bot")
        elif opponent_type == 'greedy':
            self.ai_player = GreedyAgent("Greedy Bot")
        elif opponent_type == 'ppo':
            self.ai_player = PPOAgent("PPO Bot")
            model_path = self.project_dir / 'saved_models' / 'ppo_agent.pth'
            self.ai_player.load(model_path)
            self.ai_player.set_training(False)
        elif opponent_type == 'reinforce':
            self.ai_player = RLAgent("REINFORCE Bot")
            model_path = self.project_dir / 'saved_models' / 'reinforce_agent.pth'
            self.ai_player.load(model_path)
            self.ai_player.set_training(False)
        
        # Show game screen
        self.show_game_screen()
        
        # Start game in separate thread
        self.game_active = True
        self.game_thread = threading.Thread(target=self.run_game, daemon=True)
        self.game_thread.start()
    
    def show_game_screen(self):
        """Display the main game interface"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Opponent info
        top_frame = ttk.LabelFrame(main_frame, text=f"Opponent: {self.ai_player.name}", padding=10)
        top_frame.pack(fill=tk.X, pady=5)
        
        self.opponent_score_label = ttk.Label(top_frame, text="Bullheads: 0", 
                                             font=('Helvetica', 14))
        self.opponent_score_label.pack()
        
        # Middle section - Game board
        board_frame = ttk.LabelFrame(main_frame, text="Game Board", padding=10)
        board_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.board_frame = board_frame
        self.row_frames = []
        
        for i in range(NB_ROWS):
            row_container = ttk.Frame(board_frame)
            row_container.pack(fill=tk.X, pady=5)
            
            ttk.Label(row_container, text=f"Row {i+1}:", 
                     font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT, padx=5)
            
            row_frame = ttk.Frame(row_container)
            row_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.row_frames.append(row_frame)
        
        # Bottom section - Player hand
        hand_frame = ttk.LabelFrame(main_frame, text="Your Hand", padding=10)
        hand_frame.pack(fill=tk.X, pady=5)
        
        self.hand_frame = hand_frame
        self.player_score_label = ttk.Label(hand_frame, text="Bullheads: 0", 
                                           font=('Helvetica', 14, 'bold'))
        self.player_score_label.pack()
        
        self.cards_container = ttk.Frame(hand_frame)
        self.cards_container.pack(pady=10)
        
        # Status section
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Game starting...", 
                                     font=('Helvetica', 12), foreground='blue')
        self.status_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Main Menu", 
                  command=self.return_to_menu).pack(side=tk.LEFT, padx=5)
    
    def handle_game_event(self, event_type, gameboard, all_played_cards, hand):
        """Handle game events (card/row selection requests)"""
        self.waiting_for_input = True
        self.input_type = event_type
        
        if event_type == 'choose_card':
            self.root.after(0, lambda: self.request_card_selection(gameboard, hand))
        elif event_type == 'choose_row':
            self.root.after(0, lambda: self.request_row_selection(gameboard))
    
    def request_card_selection(self, gameboard, hand):
        """GUI request for card selection"""
        self.status_label.config(text="🃏 Choose a card to play", foreground='green')
        self.update_board_display(gameboard)
        self.update_hand_display(hand, selectable=True)
    
    def request_row_selection(self, gameboard):
        """GUI request for row selection"""
        self.status_label.config(text="⚠️  Your card is too low! Choose a row to replace", 
                                foreground='red')
        self.update_board_display(gameboard, row_selectable=True)
    
    def select_card(self, card_index):
        """Player selected a card"""
        if not self.waiting_for_input or self.input_type != 'choose_card':
            return
        
        self.selected_card_index = card_index
        card = self.human_player.hand[card_index]
        
        # Confirm selection
        self.status_label.config(text=f"✓ Selected card {card.value}", foreground='blue')
        self.human_player.selected_card = card
        self.waiting_for_input = False
        
        # Disable hand buttons
        self.update_hand_display(self.human_player.hand, selectable=False)
    
    def select_row(self, row_index):
        """Player selected a row"""
        if not self.waiting_for_input or self.input_type != 'choose_row':
            return
        
        self.status_label.config(text=f"✓ Selected row {row_index + 1}", foreground='blue')
        self.human_player.selected_row = row_index
        self.waiting_for_input = False
        
        # Disable row buttons
        for btn in self.row_buttons:
            btn.destroy()
        self.row_buttons.clear()
    
    def update_board_display(self, gameboard, row_selectable=False):
        """Update the game board display"""
        for i, row_frame in enumerate(self.row_frames):
            # Clear previous cards
            for widget in row_frame.winfo_children():
                widget.destroy()
            
            # Display row cards
            for card in gameboard.board[i]:
                if card.value in self.card_images:
                    card_label = ttk.Label(row_frame, image=self.card_images[card.value])
                    card_label.image = self.card_images[card.value]
                else:
                    card_label = ttk.Label(row_frame, text=f"{card.value}\n({card.bullheads})", 
                                          relief=tk.RAISED, width=8, height=4)
                card_label.pack(side=tk.LEFT, padx=2)
            
            # Add selection button if needed
            if row_selectable:
                btn = ttk.Button(row_frame, text="Select This Row", 
                               command=lambda r=i: self.select_row(r))
                btn.pack(side=tk.LEFT, padx=10)
                self.row_buttons.append(btn)
    
    def update_hand_display(self, hand, selectable=False):
        """Update player's hand display"""
        # Clear previous cards
        for widget in self.cards_container.winfo_children():
            widget.destroy()
        self.card_buttons.clear()
        
        # Display hand cards
        for i, card in enumerate(hand):
            card_frame = ttk.Frame(self.cards_container)
            card_frame.pack(side=tk.LEFT, padx=5)
            
            if selectable:
                if card.value in self.card_images:
                    btn = tk.Button(card_frame, image=self.card_images[card.value],
                                   command=lambda idx=i: self.select_card(idx),
                                   cursor='hand2', relief=tk.RAISED, bd=3)
                    btn.image = self.card_images[card.value]
                else:
                    btn = ttk.Button(card_frame, text=f"{card.value}\n({card.bullheads})", 
                                   command=lambda idx=i: self.select_card(idx),
                                   width=8)
                btn.pack()
                self.card_buttons.append(btn)
            else:
                if card.value in self.card_images:
                    card_label = ttk.Label(card_frame, image=self.card_images[card.value])
                    card_label.image = self.card_images[card.value]
                else:
                    card_label = ttk.Label(card_frame, text=f"{card.value}\n({card.bullheads})", 
                                          relief=tk.RAISED, width=8, height=4)
                card_label.pack()
    
    def update_scores(self):
        """Update score displays"""
        if self.human_player and self.ai_player:
            self.player_score_label.config(text=f"Bullheads: {self.human_player.bullheads}")
            self.opponent_score_label.config(text=f"Bullheads: {self.ai_player.bullheads}")
    
    def run_game(self):
        """Run the game loop in a separate thread"""
        try:
            self.game = Game([self.human_player, self.ai_player], display=False)
            
            # Game loop
            for turn in range(10):
                self.root.after(0, lambda t=turn: self.status_label.config(
                    text=f"Turn {t+1}/10", foreground='blue'))
                
                # Update displays
                self.root.after(0, lambda: self.update_board_display(self.game.gameboard))
                self.root.after(0, lambda: self.update_hand_display(self.human_player.hand))
                
                # Run turn
                self.game.turn()
                
                # Update scores
                self.root.after(0, self.update_scores)
                
                # Small delay for better UX
                import time
                time.sleep(0.5)
            
            # Game over
            self.root.after(0, self.show_game_over)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Game error: {e}"))
            self.root.after(0, self.return_to_menu)
    
    def show_game_over(self):
        """Show game over screen"""
        self.game_active = False
        
        # Determine winner
        if self.human_player.bullheads < self.ai_player.bullheads:
            result = "🎉 You Win!"
            color = 'green'
        elif self.human_player.bullheads > self.ai_player.bullheads:
            result = "😔 You Lose!"
            color = 'red'
        else:
            result = "🤝 It's a Tie!"
            color = 'blue'
        
        message = f"{result}\n\nYour score: {self.human_player.bullheads} bullheads\n"
        message += f"{self.ai_player.name} score: {self.ai_player.bullheads} bullheads"
        
        self.status_label.config(text=result, foreground=color, font=('Helvetica', 18, 'bold'))
        
        # Show result dialog
        messagebox.showinfo("Game Over", message)
        
        # Return to menu
        self.return_to_menu()
    
    def return_to_menu(self):
        """Return to main menu"""
        self.game_active = False
        self.show_welcome_screen()


def main():
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = VisualGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
