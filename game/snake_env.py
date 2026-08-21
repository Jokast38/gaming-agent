"""Environnement Snake minimal, jouable avec ou sans rendu graphique (PyGame)."""
import random
from collections import namedtuple
from enum import Enum

import numpy as np

BLOCK_SIZE = 20
GRID_W, GRID_H = 20, 15  # cases -> fenêtre 400x300
SPEED = 40  # FPS en mode rendu

Point = namedtuple("Point", ["x", "y"])


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# ordre horaire utilisé pour tourner à droite/gauche
_CLOCKWISE = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]


class SnakeEnv:
    """Action = 0 (tout droit), 1 (tourner à droite), 2 (tourner à gauche)."""

    def __init__(self, render=False, max_steps_without_food=100):
        self.render_enabled = render
        self.max_steps_without_food = max_steps_without_food
        self._screen = None
        self._clock = None
        self._font = None
        if self.render_enabled:
            self._init_render()
        self.reset()

    def _init_render(self):
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((GRID_W * BLOCK_SIZE, GRID_H * BLOCK_SIZE))
        pygame.display.set_caption("Snake - Gaming Agent")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 20)

    def reset(self):
        self.direction = Direction.RIGHT
        cx, cy = GRID_W // 2, GRID_H // 2
        self.head = Point(cx, cy)
        self.snake = [self.head, Point(cx - 1, cy), Point(cx - 2, cy)]
        self.score = 0
        self.steps_since_food = 0
        self.frame = 0
        self._place_food()
        return self.get_state()

    def _place_food(self):
        while True:
            p = Point(random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
            if p not in self.snake:
                self.food = p
                return

    def step(self, action):
        """action in {0,1,2}. Retourne (state, reward, done, score)."""
        self.frame += 1

        if self.render_enabled:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self._pygame.quit()
                    quit()

        prev_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False

        # la limite de pas sans manger s'allonge avec la taille du serpent, sinon un
        # serpent long n'a plus le temps de parcourir la grille pour trouver la nourriture
        step_limit = self.max_steps_without_food + 20 * len(self.snake)

        if self._is_collision() or self.steps_since_food > step_limit:
            done = True
            reward = -10
            if self.render_enabled:
                self._draw()
            return self.get_state(), reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()
            self.steps_since_food += 1
            # reward shaping : petit bonus/malus selon qu'on se rapproche ou s'éloigne
            # de la nourriture, pour aider l'agent tabulaire à naviguer
            new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            reward = 1 if new_dist < prev_dist else -1

        if self.render_enabled:
            self._draw()
            self._clock.tick(SPEED)

        return self.get_state(), reward, done, self.score

    def _is_collision(self, point=None):
        point = point or self.head
        if point.x < 0 or point.x >= GRID_W or point.y < 0 or point.y >= GRID_H:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        idx = _CLOCKWISE.index(self.direction)
        if action == 0:
            new_dir = _CLOCKWISE[idx]
        elif action == 1:
            new_dir = _CLOCKWISE[(idx + 1) % 4]
        else:
            new_dir = _CLOCKWISE[(idx - 1) % 4]
        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        self.head = Point(x, y)

    def get_state(self):
        """État discret : dangers immédiats + à 2 cases, direction, position nourriture, taille."""
        head = self.head
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        point_l2 = Point(head.x - 2, head.y)
        point_r2 = Point(head.x + 2, head.y)
        point_u2 = Point(head.x, head.y - 2)
        point_d2 = Point(head.x, head.y + 2)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # danger tout droit (1 case)
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # danger à droite (1 case)
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            # danger à gauche (1 case)
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            # danger tout droit (2 cases) : anticipation
            (dir_r and self._is_collision(point_r2))
            or (dir_l and self._is_collision(point_l2))
            or (dir_u and self._is_collision(point_u2))
            or (dir_d and self._is_collision(point_d2)),
            # danger à droite (2 cases)
            (dir_u and self._is_collision(point_r2))
            or (dir_d and self._is_collision(point_l2))
            or (dir_l and self._is_collision(point_u2))
            or (dir_r and self._is_collision(point_d2)),
            # danger à gauche (2 cases)
            (dir_d and self._is_collision(point_r2))
            or (dir_u and self._is_collision(point_l2))
            or (dir_r and self._is_collision(point_u2))
            or (dir_l and self._is_collision(point_d2)),
            # direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # position de la nourriture
            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y,
            # taille du serpent : au-delà d'un seuil, le risque de s'auto-enfermer augmente
            len(self.snake) > 15,
        ]
        return np.array(state, dtype=int)

    def _draw(self):
        pygame = self._pygame
        self._screen.fill((0, 0, 0))
        for p in self.snake:
            pygame.draw.rect(
                self._screen, (0, 200, 0), (p.x * BLOCK_SIZE, p.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            )
        pygame.draw.rect(
            self._screen,
            (200, 0, 0),
            (self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
        )
        text = self._font.render(f"Score: {self.score}", True, (255, 255, 255))
        self._screen.blit(text, (5, 5))
        pygame.display.flip()

    def close(self):
        if self.render_enabled:
            self._pygame.quit()
