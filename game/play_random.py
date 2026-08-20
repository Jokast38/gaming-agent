"""Agent aléatoire jouable à tout moment : sert de référence de score.

Usage:
    python game/play_random.py --episodes 20 --render
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from game.snake_env import SnakeEnv


def run(episodes=20, render=False):
    env = SnakeEnv(render=render)
    scores = []
    for ep in range(episodes):
        env.reset()
        done = False
        while not done:
            action = random.randint(0, 2)
            _, _, done, score = env.step(action)
        scores.append(score)
        print(f"Episode {ep + 1}/{episodes} - score: {score}")
    env.close()
    print(f"\nScore moyen agent aléatoire sur {episodes} parties : {sum(scores) / len(scores):.2f}")
    print(f"Max: {max(scores)} | Min: {min(scores)}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run(episodes=args.episodes, render=args.render)
