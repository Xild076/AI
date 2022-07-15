import os

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiDiscrete, MultiBinary
import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def is_even(num):
    return num % 2 == 0


class Snake():
    def __init__(self, size):
        self.matrix = [[0 for i in range(size)] for j in range(size)]
        # Blank = 0, Body = 1, Head = 2, Fruit = 3
        self.snake_head = (size // 2, size // 2)
        self.snake_body = []
        self.direction = 2
        self.score = 0
        self.food = (size // 2 + 2, size // 2)
        self.game_over = False
        self.size = size
        self.survive_time = 0
        self.render_value = {
            0: ' ',
            1: '▄',
            2: 'Ö',
            3: 'ó'
        }

    def gen_food(self):
        done = False
        while not done:
            nums = tuple(random.randint(1, self.size - 1) for _ in range(2))
            if not nums == self.snake_head or not nums in self.snake_body:
                return nums

    def tick(self, action):
        # Even means up = 0 and down = 2, Odd means right = 3 and right = 1
        if action == 4:
            pass
        else:
            if not abs(self.direction - action) == 2:
                self.direction = action
        self.update()

    def update(self):
        x = 0
        y = 0
        if is_even(self.direction):
            x = self.direction - 1
        else:
            y = self.direction - 2
        self.snake_head = tuple([self.snake_head[0] + x, self.snake_head[1] + y])
        eaten = False
        if 0 > self.snake_head[0] or self.snake_head[0] >= self.size or 0 > self.snake_head[1] or self.snake_head[1] >= self.size:
            self.game_over = True
            return
        head_loc = self.matrix[self.snake_head[0]][self.snake_head[1]]
        if head_loc == 3:
            self.score += 10
            self.food = self.gen_food()
            eaten = True
        if head_loc == 1:
            self.game_over = True
            return
        for i in range(len(self.snake_body)):
            if eaten and i + 1 == len(self.snake_body):
                self.snake_body.append(self.snake_body[-1])
            self.snake_body[i] = tuple([self.snake_body[i][0] + x, self.snake_body[i][1] + y])
        self.matrix = [[0 for i in range(self.size)] for j in range(self.size)]
        self.matrix[self.snake_head[0]][self.snake_head[1]] = 2
        self.matrix[self.food[0]][self.food[1]] = 3
        for part in self.snake_body:
            self.matrix[part[0]][part[1]] == 1
        self.survive_time += 1

    def print_matrix(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.render_value.get(self.matrix[i][j]), end=" ")
            print()

    def play(self):
        while not self.game_over:
            self.tick(int(input()))
            self.print_matrix()


class SnakeEnv(Env):
    def __init__(self, size):
        super(SnakeEnv, self).__init__()
        self.snake = Snake(size)
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([[0 for i in range(size)] for j in range(size)]), high=np.array([[1 for i in range(size)] for j in range(size)]), shape=(size,size))
        self.state = self.snake.matrix

    def step(self, action):
        current_score = self.snake.score
        current_time = self.snake.survive_time
        self.snake.tick(action)
        reward = self.snake.score - current_score + self.snake.survive_time - current_time
        done = self.snake.game_over
        self.state = self.snake.matrix
        info = {}
        return np.array(self.state), reward, done, info

    def reset(self):
        self.snake.__init__(self.snake.size)
        return np.array(self.snake.matrix)

    def render(self, mode="human"):
        print(self.snake.print_matrix())
