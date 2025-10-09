"""
Self-Play Training Script
Train agents using self-play against improving versions of themselves
"""

import numpy as np
import random
import torch
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from game import Game, RandomAgent, GreedyAgent # type: ignore
from train_ppo import PPOAgent
from utils import evaluate_agent, plot_selfplay_results # type: ignore


def train_with_self_play(initial_agent, num_generations=10, episodes_per_gen=300,
                        learning_rate=0.0003, gamma=0.99, gae_lambda=0.95,
                        update_epochs=4, clip_epsilon=0.2, value_coef=0.5, 
                        entropy_coef=0.01, save_path='selfplay_agent.pth'):
    """
    Train agent using self-play against improving versions of itself
    
    Args:
        initial_agent: Starting PPO agent (can be pre-trained)
        num_generations: Number of self-play generations
        episodes_per_gen: Episodes to train per generation
        Other args: PPO hyperparameters
        save_path: Path to save the best agent
    
    Returns:
        final_agent: Best agent from self-play
        stats: Training statistics across generations
    """
    print("\n" + "="*70)
    print(" SELF-PLAY TRAINING ".center(70, "="))
    print("="*70)
    print(f"\nGenerations: {num_generations}")
    print(f"Episodes per generation: {episodes_per_gen}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Hyperparameters: gamma={gamma}, GAE_lambda={gae_lambda}")
    print(f"                clip={clip_epsilon}, update_epochs={update_epochs}")
    
    current_agent = initial_agent
    best_agent = initial_agent
    best_win_rate = 0.0
    
    # Statistics
    stats = {
        'generation_win_rates': [],
        'generation_bullheads': [],
        'opponent_bullheads': [],
        'best_win_rates': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }
    
    for generation in range(num_generations):
        print(f"\n{'-'*70}")
        print(f" Generation {generation + 1}/{num_generations} ".center(70, "-"))
        print(f"{'-'*70}")
        
        # Create opponent (copy of previous best agent)
        opponent = PPOAgent(f"Opponent_Gen{generation}")
        opponent.policy_net.load_state_dict(best_agent.policy_net.state_dict())
        opponent.value_net.load_state_dict(best_agent.value_net.state_dict())
        opponent.set_training(False)
        
        # Train current agent against previous best
        gen_stats = {
            'episode_rewards': [],
            'episode_bullheads': [],
            'opponent_bullheads': [],
            'wins': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        for episode in range(episodes_per_gen):
            # Play one game
            game = Game([current_agent, opponent], display=False)
            results = game.play()
            
            # Mark last step as done
            if len(current_agent.memory.dones) > 0:
                current_agent.memory.dones[-1] = True
            
            # Update policy
            loss_dict = current_agent.update_policy(
                gamma=gamma, gae_lambda=gae_lambda, update_epochs=update_epochs
            )
            
            # Track statistics
            agent_bullheads = results[current_agent]
            opp_bullheads = results[opponent]
            
            gen_stats['episode_bullheads'].append(agent_bullheads)
            gen_stats['opponent_bullheads'].append(opp_bullheads)
            gen_stats['wins'].append(1 if agent_bullheads < opp_bullheads else 0)
            gen_stats['policy_losses'].append(loss_dict.get('policy_loss', 0))
            gen_stats['value_losses'].append(loss_dict.get('value_loss', 0))
            gen_stats['entropies'].append(loss_dict.get('entropy', 0))
        
        # Generation summary
        gen_win_rate = np.mean(gen_stats['wins'])
        gen_bullheads = np.mean(gen_stats['episode_bullheads'])
        gen_opp_bullheads = np.mean(gen_stats['opponent_bullheads'])
        
        stats['generation_win_rates'].append(gen_win_rate)
        stats['generation_bullheads'].append(gen_bullheads)
        stats['opponent_bullheads'].append(gen_opp_bullheads)
        stats['policy_losses'].append(np.mean(gen_stats['policy_losses']))
        stats['value_losses'].append(np.mean(gen_stats['value_losses']))
        stats['entropies'].append(np.mean(gen_stats['entropies']))
        
        print(f"\n  Win rate vs previous best: {gen_win_rate:.2%}")
        print(f"  Avg bullheads: {gen_bullheads:.2f} vs {gen_opp_bullheads:.2f}")
        
        # Update best agent if current is better
        if gen_win_rate > best_win_rate:
            print(f"  ✓ New best agent! (Win rate: {gen_win_rate:.2%})")
            best_agent = PPOAgent(f"Best_Gen{generation + 1}")
            best_agent.policy_net.load_state_dict(current_agent.policy_net.state_dict())
            best_agent.value_net.load_state_dict(current_agent.value_net.state_dict())
            best_win_rate = gen_win_rate
        
        stats['best_win_rates'].append(best_win_rate)
    
    print(f"\n{'='*70}")
    print(f"Self-play complete!")
    print(f"Best win rate achieved: {best_win_rate:.2%}")
    print(f"{'='*70}")
    
    # Save the best agent
    best_agent.save(save_path)
    print(f"\nBest agent saved to '{save_path}'")
    
    return best_agent, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent using self-play')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--episodes-per-gen', type=int, default=300, help='Episodes per generation')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--update-epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--clip', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--save', type=str, default='selfplay_agent.pth', help='Path to save best model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create initial agent
    initial_agent = PPOAgent("SelfPlayAgent", learning_rate=args.lr, 
                            clip_epsilon=args.clip, entropy_coef=args.entropy_coef)
    
    # Load pre-trained weights if provided
    if args.pretrained:
        print(f"Loading pre-trained model from '{args.pretrained}'...")
        initial_agent.load(args.pretrained)
        print("Pre-trained model loaded successfully!")
    
    # Train with self-play
    best_agent, selfplay_stats = train_with_self_play(
        initial_agent,
        num_generations=args.generations,
        episodes_per_gen=args.episodes_per_gen,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        clip_epsilon=args.clip,
        entropy_coef=args.entropy_coef,
        save_path=args.save
    )
    
    # Plot self-play results
    plot_selfplay_results(selfplay_stats, save_path='selfplay_results.png')
    
    # Evaluate the best agent
    print("\n" + "="*70)
    print(" FINAL EVALUATION ".center(70, "="))
    print("="*70)
    
    random_opponent = RandomAgent("Random")
    greedy_opponent = GreedyAgent("Greedy")
    
    evaluate_agent(best_agent, random_opponent, num_games=500, agent_name="Self-Play Agent vs Random")
    evaluate_agent(best_agent, greedy_opponent, num_games=500, agent_name="Self-Play Agent vs Greedy")
