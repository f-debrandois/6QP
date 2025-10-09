"""
PPO Agent Training Script
Train a PPO agent from scratch or continue training an existing agent
"""

import numpy as np
import random
import torch
import torch.optim as optim
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from game import Player, RandomAgent, GreedyAgent, Game, NB_CARDS, NB_ROWS, CARDS_PER_ROWS # type: ignore
from models import PolicyNetwork, ValueNetwork, PPOMemory, device # type: ignore
from utils import evaluate_agent, plot_training_results # type: ignore


class PPOAgent(Player):
    """PPO (Proximal Policy Optimization) agent"""
    def __init__(self, name="PPOAgent", learning_rate=0.0003, clip_epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        super().__init__(name)
        
        # Networks
        self.policy_net = PolicyNetwork().to(device)
        self.value_net = ValueNetwork().to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Memory buffer
        self.memory = PPOMemory()
        
        # Training state
        self.training = True
        self.current_episode_step = 0
    
    def get_state(self, gameboard, all_played_cards):
        """Convert game state to neural network input"""
        state = []
        
        # Row information: last card value, row length, and total penalty
        for row in gameboard.board:
            state.append(row[-1].value / NB_CARDS)  # Last card value (normalized)
            state.append(len(row) / CARDS_PER_ROWS)  # Row length (normalized)
        
        # Row penalties (normalized by max possible penalty per row)
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
        with torch.no_grad() if not self.training else torch.enable_grad():
            logits = self.policy_net(state)
            value = self.value_net(state)
        
        # Mask invalid actions (cards not in hand)
        mask = torch.full((NB_CARDS,), float('-inf')).to(device)
        mask[valid_cards] = 0
        masked_logits = logits + mask
        
        # Get action probabilities
        probs = torch.softmax(masked_logits, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store in memory for training (reward and done will be added later)
        if self.training:
            self.memory.store(state, action.item(), log_prob, 0.0, value, False)
        
        # Convert action to card
        card_value = action.item() + 1  # Convert back to 1-indexed
        chosen_card = next(card for card in self.hand if card.value == card_value)
        
        return chosen_card
    
    def choose_row(self, gameboard, all_played_cards):
        # For row selection, use a simple heuristic (greedy)
        min_bullheads = float('inf')
        best_row = 0
        
        for i, row in enumerate(gameboard.board):
            bullheads = sum(card.bullheads for card in row)
            if bullheads < min_bullheads:
                min_bullheads = bullheads
                best_row = i
        
        return best_row
    
    def store_reward(self, reward, done=False):
        """Store reward for the last action"""
        if self.training and len(self.memory.rewards) > 0:
            self.memory.rewards[-1] = reward
            self.memory.dones[-1] = done
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Add dummy next value (0 for terminal state)
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1].item()
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t].item()
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + values
        
        return advantages, returns
    
    def update_policy(self, gamma=0.99, gae_lambda=0.95, update_epochs=4, batch_size=None):
        """Update policy using PPO algorithm"""
        if len(self.memory.states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
        
        # Get all experiences
        states, actions, old_log_probs, rewards, values, dones = self.memory.get_batches()
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, gamma, gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(update_epochs):
            # Forward pass through networks
            logits = self.policy_net(states)
            new_values = self.value_net(states)
            
            # Compute action probabilities for taken actions
            new_log_probs = []
            entropies = []
            
            for i in range(len(states)):
                # Get valid actions from state (hand mask is last 104 dims)
                hand_mask = states[i, -NB_CARDS:]
                valid_cards = (hand_mask > 0).nonzero(as_tuple=True)[0]
                
                # Mask invalid actions
                mask = torch.full((NB_CARDS,), float('-inf')).to(device)
                mask[valid_cards] = 0
                masked_logits = logits[i] + mask
                
                # Get distribution
                probs = torch.softmax(masked_logits, dim=0)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs.append(dist.log_prob(actions[i]))
                entropies.append(dist.entropy())
            
            new_log_probs = torch.stack(new_log_probs)
            entropies = torch.stack(entropies)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped for stability)
            value_loss_unclipped = (new_values - returns.detach()) ** 2
            value_clipped = values.detach() + torch.clamp(
                new_values - values.detach(), 
                -self.clip_epsilon, 
                self.clip_epsilon
            )
            value_loss_clipped = (value_clipped - returns.detach()) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # Entropy bonus (encourage exploration)
            entropy_loss = -entropies.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backpropagation
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
        
        # Clear memory
        self.memory.clear()
        
        return {
            'policy_loss': total_policy_loss / update_epochs,
            'value_loss': total_value_loss / update_epochs,
            'entropy': -total_entropy / update_epochs,
            'total_loss': (total_policy_loss + total_value_loss) / update_epochs
        }
    
    def set_training(self, mode=True):
        """Set training mode"""
        self.training = mode
        self.policy_net.train(mode)
        self.value_net.train(mode)
    
    def save(self, path):
        """Save agent's networks"""
        torch.save({
            'algorithm': 'ppo',
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict()
        }, path)
    
    def load(self, path):
        """Load agent's networks"""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])


def train_ppo_agent(num_episodes=3000, opponent_type='greedy', learning_rate=0.0003, 
                    gamma=0.99, gae_lambda=0.95, update_epochs=4, 
                    clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                    save_path='ppo_agent.pth', agent=None):
    """
    Train a PPO agent from scratch or continue training an existing agent
    
    Args:
        num_episodes: Number of training episodes
        opponent_type: 'random' or 'greedy' opponent
        learning_rate: Learning rate for the optimizers
        gamma: Discount factor for future rewards
        gae_lambda: Lambda for Generalized Advantage Estimation
        update_epochs: Number of epochs to update policy per episode
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        save_path: Path to save the trained model
        agent: Existing PPO agent (if None, create new one)
    
    Returns:
        ppo_agent: Trained PPO agent
        training_stats: Dictionary with training statistics
    """
    print("\n" + "="*70)
    print(" PPO TRAINING ".center(70, "="))
    print("="*70)
    print(f"\nEpisodes: {num_episodes}")
    print(f"Opponent: {opponent_type.capitalize()}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Hyperparameters: gamma={gamma}, GAE_lambda={gae_lambda}")
    print(f"                clip={clip_epsilon}, update_epochs={update_epochs}, entropy_coef={entropy_coef}")
    
    # Initialize agent (or use provided one)
    if agent is None:
        ppo_agent = PPOAgent("PPOAgent", learning_rate=learning_rate, 
                             clip_epsilon=clip_epsilon, 
                             value_coef=value_coef, 
                             entropy_coef=entropy_coef)
    else:
        ppo_agent = agent
    
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
        'win_rate_100': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }
    
    print(f"\nTraining PPO agent against {opponent_type} opponent...")
    
    for episode in range(num_episodes):
        # Play one game
        game = Game([ppo_agent, opponent], display=False)
        results = game.play()
        
        # Mark last step as done
        if hasattr(ppo_agent, 'store_reward') and len(ppo_agent.memory.dones) > 0:
            ppo_agent.memory.dones[-1] = True
        
        # Update policy using PPO
        loss_dict = ppo_agent.update_policy(gamma=gamma, gae_lambda=gae_lambda, 
                                           update_epochs=update_epochs)
        
        # Track statistics
        episode_reward = sum(game.turn_rewards[ppo_agent])
        rl_bullheads = results[ppo_agent]
        opponent_bullheads = results[opponent]
        
        stats['episode_rewards'].append(episode_reward)
        stats['episode_bullheads'].append(rl_bullheads)
        stats['opponent_bullheads'].append(opponent_bullheads)
        stats['wins'].append(1 if rl_bullheads < opponent_bullheads else 0)
        stats['losses'].append(1 if rl_bullheads > opponent_bullheads else 0)
        stats['policy_losses'].append(loss_dict.get('policy_loss', 0))
        stats['value_losses'].append(loss_dict.get('value_loss', 0))
        stats['entropies'].append(loss_dict.get('entropy', 0))
        
        # Calculate rolling averages
        if episode >= 99:
            stats['avg_reward_100'].append(np.mean(stats['episode_rewards'][-100:]))
            stats['win_rate_100'].append(np.mean(stats['wins'][-100:]))
        
        # Print progress
        if (episode + 1) % 200 == 0:
            avg_reward = stats['avg_reward_100'][-1] if stats['avg_reward_100'] else 0
            avg_bullheads = np.mean(stats['episode_bullheads'][-100:])
            win_rate = stats['win_rate_100'][-1] if stats['win_rate_100'] else 0
            print(f"  Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Bullheads: {avg_bullheads:.2f} | "
                  f"Win Rate: {win_rate:.2%}")
    
    print(f"\n{'='*70}")
    print(f"Training complete! Final 100-episode win rate: {stats['win_rate_100'][-1]:.2%}")
    print(f"{'='*70}")
    
    # Save the trained model
    ppo_agent.save(save_path)
    print(f"\nModel saved to '{save_path}'")
    
    return ppo_agent, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent for 6 qui Prend')
    parser.add_argument('--episodes', type=int, default=3000, help='Number of training episodes')
    parser.add_argument('--opponent', type=str, default='greedy', choices=['random', 'greedy'],
                       help='Type of opponent to train against')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--update-epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--clip', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--save', type=str, default='ppo_agent.pth', help='Path to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create agent
    ppo_agent = PPOAgent("PPOAgent", learning_rate=args.lr, 
                         clip_epsilon=args.clip, 
                         value_coef=0.5, 
                         entropy_coef=args.entropy_coef)
    
    # Load pretrained weights if provided
    if args.pretrained:
        print(f"Loading pre-trained model from '{args.pretrained}'...")
        ppo_agent.load(args.pretrained)
        print("Pre-trained model loaded successfully!")
    
    # Train the agent
    trained_agent, training_stats = train_ppo_agent(
        num_episodes=args.episodes,
        opponent_type=args.opponent,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        clip_epsilon=args.clip,
        entropy_coef=args.entropy_coef,
        save_path=args.save,
        agent=ppo_agent  # Pass the (potentially pre-trained) agent
    )
    
    # Plot training results
    plot_training_results(training_stats, title="PPO Training Results",
                         save_path='ppo_training_results.png')
    
    # Evaluate the trained agent
    print("\n" + "="*70)
    print(" FINAL EVALUATION ".center(70, "="))
    print("="*70)
    
    random_opponent = RandomAgent("Random")
    greedy_opponent = GreedyAgent("Greedy")
    
    evaluate_agent(trained_agent, random_opponent, num_games=500, agent_name="PPO Agent vs Random")
    evaluate_agent(trained_agent, greedy_opponent, num_games=500, agent_name="PPO Agent vs Greedy")
