"""
Utility Functions for Training and Evaluation
Evaluation metrics, plotting, and helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
from game import Game


def evaluate_agent(agent, opponent, num_games=100, agent_name="Agent"):
    """
    Evaluate an agent against an opponent
    
    Args:
        agent: Agent to evaluate
        opponent: Opponent agent
        num_games: Number of games to play
        agent_name: Name for display
    
    Returns:
        results: Dictionary with evaluation statistics
    """
    agent.set_training(False) if hasattr(agent, 'set_training') else None
    
    results = {
        'bullheads': [],
        'opponent_bullheads': [],
        'wins': 0,
        'losses': 0,
        'ties': 0
    }
    
    for _ in range(num_games):
        game = Game([agent, opponent], display=False)
        game_results = game.play()
        
        agent_bullheads = game_results[agent]
        opponent_bullheads = game_results[opponent]
        
        results['bullheads'].append(agent_bullheads)
        results['opponent_bullheads'].append(opponent_bullheads)
        
        if agent_bullheads < opponent_bullheads:
            results['wins'] += 1
        elif agent_bullheads > opponent_bullheads:
            results['losses'] += 1
        else:
            results['ties'] += 1
    
    print(f"\n{agent_name} Evaluation Results ({num_games} games):")
    print(f"  Win Rate: {results['wins']/num_games:.2%}")
    print(f"  Avg Bullheads: {np.mean(results['bullheads']):.2f} ± {np.std(results['bullheads']):.2f}")
    print(f"  Opponent Avg Bullheads: {np.mean(results['opponent_bullheads']):.2f} ± {np.std(results['opponent_bullheads']):.2f}")
    
    agent.set_training(True) if hasattr(agent, 'set_training') else None
    
    return results


def plot_training_results(stats, title="Training Results", save_path='training_results.png'):
    """
    Plot training statistics
    
    Args:
        stats: Training statistics dictionary
        title: Title for the plot
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Episode Rewards (smoothed)
    ax = axes[0, 0]
    episodes = range(len(stats['episode_rewards']))
    ax.plot(episodes, stats['episode_rewards'], alpha=0.3, label='Episode Reward')
    
    # Add smoothed line
    if len(stats.get('avg_reward_100', [])) > 0:
        ax.plot(range(99, len(stats['episode_rewards'])), 
                stats['avg_reward_100'], 
                linewidth=2, 
                label='100-Episode Average')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bullheads (agent vs opponent)
    ax = axes[0, 1]
    ax.plot(episodes, stats['episode_bullheads'], alpha=0.3, label='Agent Bullheads')
    ax.plot(episodes, stats['opponent_bullheads'], alpha=0.3, label='Opponent Bullheads')
    
    # Add smoothed lines
    if len(episodes) >= 100:
        window = 100
        agent_smoothed = np.convolve(stats['episode_bullheads'], np.ones(window)/window, mode='valid')
        opp_smoothed = np.convolve(stats['opponent_bullheads'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episodes)), agent_smoothed, linewidth=2, label='Agent (100-ep avg)')
        ax.plot(range(window-1, len(episodes)), opp_smoothed, linewidth=2, label='Opponent (100-ep avg)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Bullheads')
    ax.set_title('Bullheads Over Time (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Win Rate
    ax = axes[1, 0]
    if len(stats.get('win_rate_100', [])) > 0:
        ax.plot(range(99, len(episodes)), stats['win_rate_100'], linewidth=2, color='green')
    ax.axhline(y=0.5, color='red', linestyle='--', label='50% Win Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate (100-Episode Moving Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Losses (if available)
    ax = axes[1, 1]
    if 'policy_losses' in stats and len(stats['policy_losses']) > 0:
        ax.plot(episodes, stats['policy_losses'], alpha=0.5, label='Policy Loss')
        if 'value_losses' in stats:
            ax.plot(episodes, stats['value_losses'], alpha=0.5, label='Value Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Loss data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Losses')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining plot saved to {save_path}")
    plt.show()


def plot_imitation_results(stats, save_path='imitation_results.png'):
    """
    Plot imitation learning statistics
    
    Args:
        stats: Imitation learning statistics dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax = axes[0]
    episodes = range(len(stats['losses']))
    ax.plot(episodes, stats['losses'], alpha=0.3)
    if len(stats.get('avg_loss_100', [])) > 0:
        ax.plot(range(99, len(stats['losses'])),
               stats['avg_loss_100'],
               linewidth=2, label='100-Episode Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Imitation Learning Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax = axes[1]
    ax.plot(episodes, stats['accuracies'], alpha=0.3)
    if len(stats.get('avg_acc_100', [])) > 0:
        ax.plot(range(99, len(stats['accuracies'])),
               stats['avg_acc_100'],
               linewidth=2, label='100-Episode Average', color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title('Imitation Learning Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nImitation learning plot saved to {save_path}")
    plt.show()


def plot_selfplay_results(stats, save_path='selfplay_results.png'):
    """
    Plot self-play training statistics
    
    Args:
        stats: Self-play statistics dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Win rates
    ax = axes[0]
    generations = range(1, len(stats['generation_win_rates']) + 1)
    ax.plot(generations, stats['generation_win_rates'],
           'o-', linewidth=2, label='Win rate vs prev best', color='purple')
    ax.plot(generations, stats.get('best_win_rates', []),
           's-', linewidth=2, label='Best win rate', color='orange')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Win Rate')
    ax.set_title('Self-Play Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bullheads
    ax = axes[1]
    ax.plot(generations, stats['generation_bullheads'],
           'o-', linewidth=2, label='Agent', color='blue')
    ax.plot(generations, stats.get('opponent_bullheads', []),
           's-', linewidth=2, label='Opponent', color='red')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Bullheads')
    ax.set_title('Average Bullheads per Generation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSelf-play plot saved to {save_path}")
    plt.show()
