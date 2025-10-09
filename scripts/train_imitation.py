"""
Imitation Learning (Pre-training) Script
Pre-train agents using behavioral cloning on greedy agent
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

from game import GreedyAgent, Game, NB_CARDS, NB_TURNS # type: ignore
from train_ppo import PPOAgent
from models import device # type: ignore
from utils import plot_imitation_results, evaluate_agent # type: ignore
from game import RandomAgent, GreedyAgent as GreedyEvaluator # type: ignore


def pretrain_on_greedy(ppo_agent, num_episodes=1000, learning_rate=0.001):
    """
    Pre-train PPO agent by imitating greedy agent (behavioral cloning)
    
    Args:
        ppo_agent: PPO agent to pre-train
        num_episodes: Number of demonstration episodes
        learning_rate: Learning rate for imitation learning
    
    Returns:
        pretrain_stats: Training statistics
    """
    print("\n" + "="*70)
    print(" IMITATION LEARNING (PRE-TRAINING) ".center(70, "="))
    print("="*70)
    print(f"\nEpisodes: {num_episodes}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Expert: Greedy Agent")
    
    # Create greedy agent for demonstrations
    greedy = GreedyAgent("Greedy")
    
    # Optimizer for imitation learning
    imitation_optimizer = optim.Adam(ppo_agent.policy_net.parameters(), lr=learning_rate)
    
    # Statistics
    stats = {
        'losses': [],
        'accuracies': [],
        'avg_loss_100': [],
        'avg_acc_100': []
    }
    
    total_correct = 0
    total_actions = 0
    
    print("\nPre-training on greedy agent demonstrations...")
    
    for episode in range(num_episodes):
        # Create a game with greedy agent
        game = Game([greedy], display=False)
        
        episode_losses = []
        episode_correct = 0
        episode_actions = 0
        
        # Play through the game and collect demonstrations
        for turn in range(NB_TURNS):
            if len(greedy.hand) == 0:
                break
            
            # Get state
            state = ppo_agent.get_state(game.gameboard, game.all_played_cards)
            
            # Get greedy's action (expert demonstration)
            greedy_card = greedy.choose_card(game.gameboard, game.all_played_cards)
            greedy_action = greedy_card.value - 1  # Convert to 0-indexed
            
            # Get valid actions
            valid_cards = [c.value - 1 for c in greedy.hand]
            
            # Remove the card from hand (simulate playing)
            greedy.hand.remove(greedy_card)
            
            # Forward pass through policy network
            logits = ppo_agent.policy_net(state)
            
            # Mask invalid actions
            mask = torch.full((NB_CARDS,), float('-inf')).to(device)
            mask[valid_cards] = 0
            masked_logits = logits + mask
            
            # Compute cross-entropy loss (supervised learning)
            loss = nn.functional.cross_entropy(
                masked_logits.unsqueeze(0), 
                torch.tensor([greedy_action]).to(device)
            )
            
            # Backpropagation
            imitation_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo_agent.policy_net.parameters(), 1.0)
            imitation_optimizer.step()
            
            # Track accuracy
            predicted_action = masked_logits.argmax().item()
            if predicted_action == greedy_action:
                episode_correct += 1
                total_correct += 1
            episode_actions += 1
            total_actions += 1
            
            episode_losses.append(loss.item())
            
            # Simulate card placement (simplified - just track cards played)
            game.all_played_cards.append(greedy_card)
        
        # Track statistics
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0
        episode_accuracy = episode_correct / episode_actions if episode_actions > 0 else 0
        
        stats['losses'].append(avg_episode_loss)
        stats['accuracies'].append(episode_accuracy)
        
        # Calculate rolling averages
        if episode >= 99:
            stats['avg_loss_100'].append(np.mean(stats['losses'][-100:]))
            stats['avg_acc_100'].append(np.mean(stats['accuracies'][-100:]))
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_loss = np.mean(stats['losses'][-100:])
            avg_acc = np.mean(stats['accuracies'][-100:])
            print(f"  Episode {episode + 1}/{num_episodes} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {avg_acc:.2%}")
    
    final_accuracy = total_correct / total_actions if total_actions > 0 else 0
    print(f"\n{'='*70}")
    print(f"Pre-training complete!")
    print(f"Final accuracy: {final_accuracy:.2%}")
    print(f"Total demonstrations: {total_actions}")
    print(f"{'='*70}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train PPO agent using imitation learning')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of demonstration episodes')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save', type=str, default='pretrained_agent.pth', help='Path to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create PPO agent to pre-train
    agent = PPOAgent("PretrainedAgent")
    
    # Pre-train the agent
    pretrain_stats = pretrain_on_greedy(agent, num_episodes=args.episodes, 
                                       learning_rate=args.lr)
    
    # Save the pre-trained model
    agent.save(args.save)
    print(f"\nPre-trained model saved to '{args.save}'")
    
    # Plot imitation learning results
    plot_imitation_results(pretrain_stats, save_path='imitation_learning_results.png')
    
    # Evaluate the pre-trained agent
    print("\n" + "="*70)
    print(" PRE-TRAINED AGENT EVALUATION ".center(70, "="))
    print("="*70)
    
    random_opponent = RandomAgent("Random")
    greedy_opponent = GreedyEvaluator("Greedy")
    
    evaluate_agent(agent, random_opponent, num_games=500, agent_name="Pre-trained Agent vs Random")
    evaluate_agent(agent, greedy_opponent, num_games=500, agent_name="Pre-trained Agent vs Greedy")
