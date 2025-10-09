"""
REINFORCE Agent Training Script
Train a REINFORCE (vanilla policy gradient) agent
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from game import Player, RandomAgent, GreedyAgent, Game, NB_CARDS, NB_ROWS, CARDS_PER_ROWS # type: ignore
from models import PolicyNetwork, device # type: ignore
from utils import evaluate_agent, plot_training_results # type: ignore


class RLAgent(Player):
    """Reinforcement Learning agent using REINFORCE algorithm"""
    def __init__(self, name="RLAgent", learning_rate=0.001):
        super().__init__(name)
        self.policy_net = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # For training
        self.saved_log_probs = []
        self.rewards = []
        self.training = True
    
    def get_state(self, gameboard, all_played_cards):
        """Convert game state to neural network input"""
        state = []
        
        # Row information: last card value, row length, and total penalty
        for row in gameboard.board:
            state.append(row[-1].value / NB_CARDS)  # Last card value (normalized)
            state.append(len(row) / CARDS_PER_ROWS)  # Row length (normalized)
        
        # Row penalties (normalized by max possible penalty per row)
        # Max penalty per row is roughly 7*5 = 35 (5 cards with ~7 bullheads each)
        for row in gameboard.board:
            penalty = sum(card.bullheads for card in row)
            state.append(penalty / 35.0)  # Normalize penalty
        
        # Hand representation: binary mask for all possible cards
        hand_mask = [0.0] * NB_CARDS
        for card in self.hand:
            hand_mask[card.value - 1] = 1.0
        
        state.extend(hand_mask)
        
        return torch.FloatTensor(state).to(device)
    
    def choose_card(self, gameboard, all_played_cards):
        state = self.get_state(gameboard, all_played_cards)
        
        # Get valid actions (cards in hand)
        valid_cards = [card.value - 1 for card in self.hand]  # Convert to 0-indexed
        
        # Forward pass through policy network
        logits = self.policy_net(state)
        
        # Mask invalid actions (cards not in hand)
        mask = torch.full((NB_CARDS,), float('-inf')).to(device)
        mask[valid_cards] = 0
        masked_logits = logits + mask
        
        # Get action probabilities
        probs = torch.softmax(masked_logits, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Save log probability for training
        if self.training:
            self.saved_log_probs.append(dist.log_prob(action))
        
        # Convert action to card
        card_value = action.item() + 1  # Convert back to 1-indexed
        chosen_card = next(card for card in self.hand if card.value == card_value)
        
        return chosen_card
    
    def choose_row(self, gameboard, all_played_cards):
        # For row selection, use a simple heuristic (greedy)
        # This could be extended to a second network if needed
        min_bullheads = float('inf')
        best_row = 0
        
        for i, row in enumerate(gameboard.board):
            bullheads = sum(card.bullheads for card in row)
            if bullheads < min_bullheads:
                min_bullheads = bullheads
                best_row = i
        
        return best_row
    
    def update_policy(self, gamma=0.99):
        """Update policy using REINFORCE algorithm"""
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalize returns for stability
        returns = torch.FloatTensor(returns).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Perform backpropagation
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode data
        loss_value = loss.item()
        self.saved_log_probs = []
        self.rewards = []
        
        return loss_value
    
    def set_training(self, mode=True):
        """Set training mode"""
        self.training = mode
        self.policy_net.train(mode)
    
    def save(self, path):
        """Save agent's network"""
        torch.save({
            'algorithm': 'reinforce',
            'policy_net': self.policy_net.state_dict()
        }, path)
    
    def load(self, path):
        """Load agent's network"""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])


def train_rl_agent(num_episodes=1000, opponent_type='random', learning_rate=0.001, gamma=0.99,
                   save_path='reinforce_agent.pth'):
    """
    Train the RL agent against a baseline opponent
    
    Args:
        num_episodes: Number of training episodes
        opponent_type: 'random' or 'greedy' opponent
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        save_path: Path to save the trained model
    
    Returns:
        rl_agent: Trained RL agent
        training_stats: Dictionary with training statistics
    """
    print("\n" + "="*70)
    print(" REINFORCE TRAINING ".center(70, "="))
    print("="*70)
    print(f"\nEpisodes: {num_episodes}")
    print(f"Opponent: {opponent_type.capitalize()}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma (discount): {gamma}")
    
    # Initialize agents
    rl_agent = RLAgent("RLAgent", learning_rate=learning_rate)
    
    if opponent_type == 'random':
        opponent = RandomAgent("Opponent")
    elif opponent_type == 'greedy':
        opponent = GreedyAgent("Opponent")
    else:
        raise ValueError("opponent_type must be 'random' or 'greedy'")
    
    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_bullheads': [],
        'opponent_bullheads': [],
        'wins': [],
        'losses': [],
        'avg_reward_100': [],
        'win_rate_100': []
    }
    
    print(f"\nTraining RL agent against {opponent_type} opponent...")
    
    for episode in range(num_episodes):
        # Play one game
        game = Game([rl_agent, opponent], display=False)
        results = game.play()
        
        # Store rewards from the episode
        episode_reward = sum(game.turn_rewards[rl_agent])
        rl_agent.rewards = game.turn_rewards[rl_agent]
        
        # Update policy
        loss = rl_agent.update_policy(gamma=gamma)
        
        # Track statistics
        rl_bullheads = results[rl_agent]
        opponent_bullheads = results[opponent]
        
        stats['episode_rewards'].append(episode_reward)
        stats['episode_bullheads'].append(rl_bullheads)
        stats['opponent_bullheads'].append(opponent_bullheads)
        stats['wins'].append(1 if rl_bullheads < opponent_bullheads else 0)
        stats['losses'].append(1 if rl_bullheads > opponent_bullheads else 0)
        
        # Calculate rolling averages
        if episode >= 99:
            avg_reward = np.mean(stats['episode_rewards'][-100:])
            win_rate = np.mean(stats['wins'][-100:])
            stats['avg_reward_100'].append(avg_reward)
            stats['win_rate_100'].append(win_rate)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-100:])
            avg_bullheads = np.mean(stats['episode_bullheads'][-100:])
            win_rate = np.mean(stats['wins'][-100:])
            print(f"  Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Bullheads: {avg_bullheads:.2f} | "
                  f"Win Rate: {win_rate:.2%}")
    
    print(f"\n{'='*70}")
    print(f"Training complete! Final 100-episode win rate: {stats['win_rate_100'][-1]:.2%}")
    print(f"{'='*70}")
    
    # Save the trained model
    rl_agent.save(save_path)
    print(f"\nModel saved to '{save_path}'")
    
    return rl_agent, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a REINFORCE agent for 6 qui Prend')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--opponent', type=str, default='random', choices=['random', 'greedy'],
                       help='Type of opponent to train against')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--save', type=str, default='reinforce_agent.pth', help='Path to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Train the agent
    trained_agent, training_stats = train_rl_agent(
        num_episodes=args.episodes,
        opponent_type=args.opponent,
        learning_rate=args.lr,
        gamma=args.gamma,
        save_path=args.save
    )
    
    # Plot training results
    plot_training_results(training_stats, title="REINFORCE Training Results",
                         save_path='reinforce_training_results.png')
    
    # Evaluate the trained agent
    print("\n" + "="*70)
    print(" FINAL EVALUATION ".center(70, "="))
    print("="*70)
    
    random_opponent = RandomAgent("Random")
    greedy_opponent = GreedyAgent("Greedy")
    
    evaluate_agent(trained_agent, random_opponent, num_games=500, agent_name="REINFORCE Agent vs Random")
    evaluate_agent(trained_agent, greedy_opponent, num_games=500, agent_name="REINFORCE Agent vs Greedy")
