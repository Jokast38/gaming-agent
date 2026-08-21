"""Entraîne l'agent DQN sur Snake et sauvegarde le meilleur modèle.

Usage:
    python train_dqn.py --episodes 1000 --run-name dqn_essai1
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agent.dqn_agent import DQNAgent
from game.snake_env import SnakeEnv


def train(episodes=1000, run_name="dqn_run", seed=None, epsilon_decay=0.995):
    if seed is not None:
        import random
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = SnakeEnv(render=False)
    agent = DQNAgent(epsilon_decay=epsilon_decay)

    scores = []
    best_score = -1
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state

        agent.decay_epsilon()
        scores.append(score)

        if score > best_score:
            best_score = score
            agent.save(run_dir / "best_agent.pt")

        if (ep + 1) % 50 == 0:
            avg = np.mean(scores[-50:])
            print(f"Episode {ep + 1}/{episodes} - score: {score} - moyenne(50): {avg:.2f} - epsilon: {agent.epsilon:.3f}")

    agent.save(run_dir / "final_agent.pt")

    with open(run_dir / "scores.json", "w") as f:
        json.dump(scores, f)

    _plot_curve(scores, run_dir / "learning_curve.png", run_name)

    print(f"\nMeilleur score: {best_score}")
    print(f"Score moyen (100 derniers épisodes): {np.mean(scores[-100:]):.2f}")
    print(f"Agents sauvegardés dans: {run_dir}")

    return scores


def _plot_curve(scores, out_path, run_name, window=50):
    scores = np.array(scores)
    moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, label="Score par épisode")
    plt.plot(range(window - 1, len(scores)), moving_avg, label=f"Moyenne mobile ({window})")
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.title(f"Courbe de progression - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Courbe sauvegardée: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--run-name", type=str, default="dqn_run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    args = parser.parse_args()
    train(episodes=args.episodes, run_name=args.run_name, seed=args.seed, epsilon_decay=args.epsilon_decay)
