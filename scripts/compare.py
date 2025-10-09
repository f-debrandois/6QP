"""
Model comparison / tournament script.

Usage (examples):
  # Round-robin tournament (2+ agents)
  python compare.py --ppo saved_models/ppo.pth --reinforce saved_models/reinforce.pth --include-baselines --games 200

  # Compare a single agent vs baselines
  python compare.py --ppo saved_models/ppo.pth --include-baselines --games 500
"""

import argparse
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src directory to path so imports work when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project-specific imports (may need type: ignore if mypy complains)
from game import RandomAgent, GreedyAgent, Game  # type: ignore
from train_ppo import PPOAgent  # type: ignore
from train_reinforce import RLAgent  # type: ignore
from utils import evaluate_agent  # type: ignore


def detect_algorithm(model_path):
    """
    Detect the algorithm used in a saved model.
    Returns: 'ppo', 'reinforce', or 'unknown'
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Check for algorithm metadata (new format)
        if isinstance(checkpoint, dict) and 'algorithm' in checkpoint:
            return checkpoint['algorithm']
        
        return 'unknown'
    except Exception as e:
        print(f"Warning: Could not detect algorithm for {model_path}: {e}")
        return 'unknown'


def _safe_load_agent(agent, model_path):
    """Try to load a model on an agent if path exists. Print status messages."""
    if model_path and Path(model_path).exists():
        try:
            agent.load(model_path)
            print(f"✓ Loaded {agent.__class__.__name__} from '{model_path}'")
        except Exception as e:
            print(f"⚠ Failed to load model '{model_path}': {e}. Using untrained agent.")
    else:
        if model_path:
            print(f"⚠ No model file found at '{model_path}', using untrained agent.")
    # Try to set evaluation mode / disable training if available
    if hasattr(agent, "set_training"):
        try:
            agent.set_training(False)
        except Exception:
            pass


def play_head_to_head(agent1, agent2, num_games=100):
    """
    Play two agents head-to-head using Game and return aggregated results.

    We assume Game([agent1, agent2], display=False).play() returns a dict mapping
    agent instance -> bullheads (lower is better). This function runs num_games
    independent games and returns stats.
    """
    agent1_bh = []
    agent2_bh = []

    for _ in range(num_games):
        game = Game([agent1, agent2], display=False)
        results = game.play()
        # Expect results to be keyed by agent object (agent1, agent2)
        bh1 = results.get(agent1)
        bh2 = results.get(agent2)
        # If game.play returns scores in other form, this will raise/produce None — keep explicit checks
        if bh1 is None or bh2 is None:
            raise RuntimeError("Unexpected Game.play() result format. Expected mapping by agent object.")
        agent1_bh.append(bh1)
        agent2_bh.append(bh2)

    agent1_bh = np.array(agent1_bh)
    agent2_bh = np.array(agent2_bh)

    agent1_wins = int(np.sum(agent1_bh < agent2_bh))
    agent2_wins = int(np.sum(agent2_bh < agent1_bh))
    ties = int(np.sum(agent1_bh == agent2_bh))

    return {
        "agent1_avg_bullheads": float(np.mean(agent1_bh)),
        "agent2_avg_bullheads": float(np.mean(agent2_bh)),
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "ties": ties,
        "agent1_bullheads": agent1_bh.tolist(),
        "agent2_bullheads": agent2_bh.tolist(),
    }


def round_robin_tournament(agent_configs, num_games=100):
    """
    Run a round-robin tournament. agent_configs is a list of tuples:
      (agent_name, agent_type, model_path)
    agent_type: 'ppo', 'reinforce', 'random', 'greedy'
    Returns: (results, matchup_results)
      results: dict mapping agent_name -> aggregated stats
      matchup_results: dict mapping "A_vs_B" -> matchup stats
    """
    # Load agents
    agents = []
    for agent_name, agent_type, model_path in agent_configs:
        print("\n" + "-" * 70)
        print(f" Loading: {agent_name} ".center(70, "-"))
        print("-" * 70)
        if agent_type == "ppo":
            agent = PPOAgent(agent_name)
            _safe_load_agent(agent, model_path)
        elif agent_type == "reinforce":
            agent = RLAgent(agent_name)
            _safe_load_agent(agent, model_path)
        elif agent_type == "random":
            agent = RandomAgent(agent_name)
        elif agent_type == "greedy":
            agent = GreedyAgent(agent_name)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agents.append((agent_name, agent))

    # Initialize results storage
    results = {
        name: {"wins": 0, "losses": 0, "ties": 0, "total_bullheads": 0.0, "games_played": 0}
        for name, _ in agents
    }
    matchup_results = {}

    print("\n" + "=" * 70)
    print(f" Starting Tournament ({len(agents)} agents, {num_games} games per matchup) ".center(70, "="))
    print("=" * 70 + "\n")

    total_matchups = len(agents) * (len(agents) - 1) // 2
    current_matchup = 0

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            current_matchup += 1
            name1, agent1 = agents[i]
            name2, agent2 = agents[j]

            print(f"[{current_matchup}/{total_matchups}] {name1} vs {name2} ({num_games} games)...", end=" ")
            matchup = play_head_to_head(agent1, agent2, num_games=num_games)

            matchup_key = f"{name1}_vs_{name2}"
            matchup_results[matchup_key] = {
                "agent1": name1,
                "agent2": name2,
                "agent1_wins": matchup["agent1_wins"],
                "agent2_wins": matchup["agent2_wins"],
                "ties": matchup["ties"],
                "agent1_avg_bullheads": matchup["agent1_avg_bullheads"],
                "agent2_avg_bullheads": matchup["agent2_avg_bullheads"],
            }

            # Update overall results
            results[name1]["wins"] += matchup["agent1_wins"]
            results[name1]["losses"] += matchup["agent2_wins"]
            results[name1]["ties"] += matchup["ties"]
            results[name1]["total_bullheads"] += matchup["agent1_avg_bullheads"] * num_games
            results[name1]["games_played"] += num_games

            results[name2]["wins"] += matchup["agent2_wins"]
            results[name2]["losses"] += matchup["agent1_wins"]
            results[name2]["ties"] += matchup["ties"]
            results[name2]["total_bullheads"] += matchup["agent2_avg_bullheads"] * num_games
            results[name2]["games_played"] += num_games

            win_rate_1 = matchup["agent1_wins"] / num_games * 100
            win_rate_2 = matchup["agent2_wins"] / num_games * 100
            print(f"✓ {name1}: {win_rate_1:.1f}% | {name2}: {win_rate_2:.1f}%")

    # Calculate final statistics
    for name in results:
        gp = results[name]["games_played"]
        results[name]["win_rate"] = results[name]["wins"] / gp if gp > 0 else 0.0
        results[name]["avg_bullheads"] = results[name]["total_bullheads"] / gp if gp > 0 else 0.0

    return results, matchup_results


def compare_agents(agent_configs, num_games=500):
    """
    Compare each agent against Random and Greedy baselines using evaluate_agent().
    Returns a dict mapping agent_name -> results dict containing vs_random and vs_greedy stats.
    """
    results = {}
    random_opp = RandomAgent("Random")
    greedy_opp = GreedyAgent("Greedy")

    for agent_name, agent_type, model_path in agent_configs:
        print("\n" + "=" * 70)
        print(f" Evaluating: {agent_name} ".center(70, "="))
        print("=" * 70)

        # Create the agent instance and load model if present
        if agent_type == "ppo":
            agent = PPOAgent(agent_name)
            _safe_load_agent(agent, model_path)
        elif agent_type == "reinforce":
            agent = RLAgent(agent_name)
            _safe_load_agent(agent, model_path)
        elif agent_type == "random":
            agent = RandomAgent(agent_name)
        elif agent_type == "greedy":
            agent = GreedyAgent(agent_name)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Evaluate vs Random
        print(f"\n🎲 {agent_name} vs Random ({num_games} games)")
        random_results = evaluate_agent(agent, random_opp, num_games=num_games, agent_name=f"{agent_name} vs Random")

        # Evaluate vs Greedy
        print(f"\n🧠 {agent_name} vs Greedy ({num_games} games)")
        greedy_results = evaluate_agent(agent, greedy_opp, num_games=num_games, agent_name=f"{agent_name} vs Greedy")

        # Package results
        results[agent_name] = {
            "vs_random": {
                "win_rate": random_results["wins"] / num_games if num_games > 0 else 0.0,
                "avg_bullheads": float(np.mean(random_results["bullheads"])) if len(random_results.get("bullheads", [])) > 0 else 0.0,
                "std_bullheads": float(np.std(random_results["bullheads"])) if len(random_results.get("bullheads", [])) > 0 else 0.0,
                "raw": random_results,
            },
            "vs_greedy": {
                "win_rate": greedy_results["wins"] / num_games if num_games > 0 else 0.0,
                "avg_bullheads": float(np.mean(greedy_results["bullheads"])) if len(greedy_results.get("bullheads", [])) > 0 else 0.0,
                "std_bullheads": float(np.std(greedy_results["bullheads"])) if len(greedy_results.get("bullheads", [])) > 0 else 0.0,
                "raw": greedy_results,
            },
        }

    return results


def plot_round_robin_results(results, matchup_results, save_path="tournament_results.png"):
    """
    Create a multi-panel plot for round-robin tournament results:
      - Overall win rates
      - Average bullheads (lower is better)
      - Win/Loss/Tie distribution
      - Head-to-head matrix heatmap
    """
    agent_names = list(results.keys())
    n = len(agent_names)
    if n == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle("Round-Robin Tournament Results", fontsize=16, fontweight="bold")

    # Plot 1: Win rates (horizontal)
    ax = axes[0, 0]
    win_rates = [results[name]["win_rate"] * 100 for name in agent_names]
    y_pos = np.arange(len(agent_names))
    bars = ax.barh(y_pos, win_rates, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_names)
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("Overall Win Rate (Round-Robin)")
    for i, val in enumerate(win_rates):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center")

    ax.axvline(50, color="red", linestyle="--", alpha=0.6)

    # Plot 2: Average bullheads (horizontal) (lower is better)
    ax = axes[0, 1]
    avg_bh = [results[name]["avg_bullheads"] for name in agent_names]
    bars = ax.barh(y_pos, avg_bh, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_names)
    ax.set_xlabel("Average Bullheads (Lower is Better)")
    ax.set_title("Average Bullheads")
    for i, val in enumerate(avg_bh):
        ax.text(val + 0.02 if val >= 0 else val - 0.02, i, f"{val:.2f}", va="center")

    # Plot 3: Win/Loss/Tie distribution (stacked bars)
    ax = axes[1, 0]
    wins = [results[name]["wins"] for name in agent_names]
    losses = [results[name]["losses"] for name in agent_names]
    ties = [results[name]["ties"] for name in agent_names]
    x = np.arange(n)
    width = 0.6
    ax.bar(x, wins, width, label="Wins", alpha=0.8)
    ax.bar(x, losses, width, bottom=wins, label="Losses", alpha=0.8)
    bottoms = np.array(wins) + np.array(losses)
    ax.bar(x, ties, width, bottom=bottoms, label="Ties", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_ylabel("Games")
    ax.set_title("Win/Loss/Tie Distribution")
    ax.legend()

    # Plot 4: Head-to-head heatmap
    ax = axes[1, 1]
    matrix = np.full((n, n), np.nan)
    for i, name1 in enumerate(agent_names):
        for j, name2 in enumerate(agent_names):
            if i == j:
                matrix[i, j] = 50.0
            elif i < j:
                key = f"{name1}_vs_{name2}"
                if key in matchup_results:
                    mr = matchup_results[key]
                    total = mr["agent1_wins"] + mr["agent2_wins"] + mr["ties"]
                    matrix[i, j] = mr["agent1_wins"] / total * 100 if total > 0 else np.nan
                    matrix[j, i] = mr["agent2_wins"] / total * 100 if total > 0 else np.nan
                else:
                    # missing matchup (shouldn't happen), mark as nan
                    matrix[i, j] = np.nan
                    matrix[j, i] = np.nan

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_yticklabels(agent_names)
    ax.set_title("Head-to-Head Win Rates (%)")
    # annotate
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "—"
            else:
                txt = f"{val:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)

    # Adjust figure size and place colorbar to the right
    fig.set_size_inches(14, 10)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Win Rate (%)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTournament results plot saved to {save_path}")
    plt.show()


def plot_comparison_bar_chart(results, save_path="comparison_results.png"):
    """
    Plot comparison summary for single-agent comparisons (vs random and vs greedy).
    Expected results format: dict agent_name -> {'vs_random': {...}, 'vs_greedy': {...}}
    """
    agent_names = list(results.keys())
    n = len(agent_names)
    if n == 0:
        print("No results to plot.")
        return

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    random_wr = [results[name]["vs_random"]["win_rate"] * 100 for name in agent_names]
    greedy_wr = [results[name]["vs_greedy"]["win_rate"] * 100 for name in agent_names]

    ax.bar(x - width / 2, random_wr, width, label="vs Random")
    ax.bar(x + width / 2, greedy_wr, width, label="vs Greedy")

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Agent Win Rate vs Baselines")
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.legend()
    for i, (r, g) in enumerate(zip(random_wr, greedy_wr)):
        ax.text(i - width / 2, r + 1.0, f"{r:.1f}%", ha="center")
        ax.text(i + width / 2, g + 1.0, f"{g:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {save_path}")
    plt.show()


def print_tournament_table(results):
    """Print formatted tournament results table."""
    print("\n" + "=" * 70)
    print(" TOURNAMENT SUMMARY ".center(70, "="))
    print("=" * 70)
    print(f"{'Agent':<30} {'Wins':<8} {'Losses':<8} {'Ties':<8} {'Win%':<8} {'Avg BH':<10}")
    print("-" * 70)
    sorted_agents = sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True)
    for agent_name, stats in sorted_agents:
        wins = stats["wins"]
        losses = stats["losses"]
        ties = stats["ties"]
        win_rate = stats["win_rate"] * 100
        avg_bh = stats["avg_bullheads"]
        print(f"{agent_name:<30} {wins:<8} {losses:<8} {ties:<8} {win_rate:>6.1f}%    {avg_bh:>6.2f}")
    print("=" * 70)
    if sorted_agents:
        winner_name, winner_stats = sorted_agents[0]
        print(f"\n🏆 WINNER: {winner_name} (Win Rate: {winner_stats['win_rate']*100:.1f}%)")
    print("=" * 70)


def print_comparison_table(results):
    """Print summary table for single-agent comparisons vs baselines."""
    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY ".center(70, "="))
    print("=" * 70)
    print(f"{'Agent':<30} {'vs Random WR%':<15} {'vs Random BH':<15} {'vs Greedy WR%':<15} {'vs Greedy BH':<15}")
    print("-" * 90)
    # Sort by vs_greedy win rate by default
    sorted_agents = sorted(results.items(), key=lambda x: x[1]["vs_greedy"]["win_rate"], reverse=True)
    for agent_name, stats in sorted_agents:
        random_wr = stats["vs_random"]["win_rate"] * 100
        random_bh = stats["vs_random"]["avg_bullheads"]
        greedy_wr = stats["vs_greedy"]["win_rate"] * 100
        greedy_bh = stats["vs_greedy"]["avg_bullheads"]
        print(f"{agent_name:<30} {random_wr:>6.1f}%    {random_bh:>9.2f}     {greedy_wr:>6.1f}%    {greedy_bh:>9.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL agents using round-robin tournament or vs baselines (auto-detect).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--models", type=str, nargs='+', default=[], 
                       help="List of model paths to compare (algorithm auto-detected)")
    parser.add_argument("--ppo", type=str, default=None, help="Path to pure PPO model (legacy)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pre-trained (imitation) model (legacy)")
    parser.add_argument("--pretrained-ppo", type=str, default=None, help="Path to pre-trained + PPO fine-tuned model (legacy)")
    parser.add_argument("--selfplay", type=str, default=None, help="Path to self-play trained model (legacy)")
    parser.add_argument("--reinforce", type=str, default=None, help="Path to REINFORCE model (legacy)")
    parser.add_argument("--include-baselines", action="store_true", help="Include random and greedy baselines")
    parser.add_argument("--games", type=int, default=200, help="Number of games per matchup/evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save plot (optional)")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Build agent config list
    agent_configs = []
    
    # New system: auto-detect from --models list
    if args.models:
        for i, model_path in enumerate(args.models):
            if not Path(model_path).exists():
                print(f"Warning: Model file not found: {model_path}")
                continue
            
            algo = detect_algorithm(model_path)
            if algo == 'unknown':
                print(f"Warning: Could not detect algorithm for {model_path}, skipping")
                continue
            
            # Create a descriptive name based on filename
            model_name = Path(model_path).stem
            agent_configs.append((model_name, algo, model_path))
    
    # Legacy system: specific arguments (for backward compatibility)
    if args.ppo:
        agent_configs.append(("Pure PPO", "ppo", args.ppo))
    if args.pretrained:
        agent_configs.append(("Pre-trained (IL)", "ppo", args.pretrained))
    if args.pretrained_ppo:
        agent_configs.append(("Pre-trained + PPO", "ppo", args.pretrained_ppo))
    if args.selfplay:
        agent_configs.append(("Self-Play", "ppo", args.selfplay))
    if args.reinforce:
        agent_configs.append(("REINFORCE", "reinforce", args.reinforce))

    if args.include_baselines:
        agent_configs.append(("Random Baseline", "random", None))
        agent_configs.append(("Greedy Baseline", "greedy", None))

    if not agent_configs:
        print("❌ Error: No agents specified for comparison! Use --help to see available options.")
        return

    # Auto-detect mode
    if len(agent_configs) >= 2:
        # Round-robin tournament mode
        results, matchup_results = round_robin_tournament(agent_configs, num_games=args.games)
        print_tournament_table(results)
        plot_path = args.save_plot if args.save_plot else "tournament_results.png"
        plot_round_robin_results(results, matchup_results, save_path=plot_path)
    else:
        # Single-agent comparison mode (vs baselines)
        results = compare_agents(agent_configs, num_games=args.games)
        print_comparison_table(results)
        plot_path = args.save_plot if args.save_plot else "comparison_results.png"
        plot_comparison_bar_chart(results, save_path=plot_path)

    print("\n" + "=" * 70)
    print(" COMPLETE ".center(70, "="))
    print("=" * 70)


if __name__ == "__main__":
    main()
