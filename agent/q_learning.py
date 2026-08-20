"""Agent Q-learning tabulaire pour Snake.

État: tuple de 11 booléens (2048 états possibles) -> table Q en dictionnaire.
Action: 0 (tout droit), 1 (droite), 2 (gauche).
"""
import json
import pickle
import random
from pathlib import Path

import numpy as np

N_ACTIONS = 3


class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _state_key(self, state):
        return tuple(int(x) for x in state)

    def _ensure_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(N_ACTIONS)

    def choose_action(self, state, greedy=False):
        key = self._state_key(state)
        self._ensure_state(key)
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done):
        key = self._state_key(state)
        next_key = self._state_key(next_state)
        self._ensure_state(key)
        self._ensure_state(next_key)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])

        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = data["q_table"]
        return agent
