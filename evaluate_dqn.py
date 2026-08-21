"""Recharge un agent DQN sauvegardé (script indépendant de train_dqn.py) et le fait rejouer.

Usage:
    python evaluate_dqn.py --model runs/dqn_essai1/best_agent.pt --episodes 20 --render
"""
import argparse

import numpy as np

from agent.dqn_agent import DQNAgent
from game.snake_env import SnakeEnv


def evaluate(model_path, episodes=20, render=False):
    agent = DQNAgent.load(model_path)
    env = SnakeEnv(render=render)

    scores = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, greedy=True)
            state, _, done, score = env.step(action)
        scores.append(score)
        print(f"Episode {ep + 1}/{episodes} - score: {score}")

    env.close()
    print(f"\nScore moyen agent DQN sur {episodes} parties : {np.mean(scores):.2f}")
    print(f"Max: {max(scores)} | Min: {min(scores)}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes, render=args.render)
