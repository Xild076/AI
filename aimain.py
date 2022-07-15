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


class ENVIRONMENT(Env):
    def __init__(self):
        super(ENVIRONMENT, self).__init__()
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([-50]), high=np.array([50]))
        self.state = random.randint(-3, 3)
        self.timer = 60

    def step(self, action):
        self.state += action - 1
        self.timer -= 1
        reward = 1 if -1 <= self.state <= 1 else -1
        done = self.timer <= 0
        info = {}
        self.state += random.randint(-1, 1)
        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([random.randint(-3, 3)]).astype(np.float32)
        self.timer = 60
        return self.state

    def render(self, mode="human"):
        print(self.state)


def is_even(num):
    return (num % 2 == 0)


class Snake():
    def __init__(self, size):
        self.matrix = [[0 for i in range(size)] for j in range(size)]
        # Blank = 0, Body = 1, Head = 2, Fruit = 3
        self.snake_head = (2, 2)
        self.snake_body = [(2, 1)]
        self.direction = 2
        self.score = 0
        self.food = (size // 2 + random.randint(1, 3), size // 2 + random.randint(1, 3))
        self.game_over = False
        self.size = size
        self.survive_time = 0
        self.render_value = {
            0: ' ',
            1: '#',
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
        fruit_spawn = True
        x = 0
        y = 0
        if is_even(self.direction):
            x = self.direction - 1
        else:
            y = self.direction - 2
        self.snake_head = tuple([self.snake_head[0] + x, self.snake_head[1] + y])
        self.snake_body.insert(0, tuple(self.snake_head))

        if self.snake_head == self.food:
            self.score += 10
            fruit_spawn = False
        else:
            self.snake_body.pop()
        if not fruit_spawn:
            self.food = self.gen_food()
        if 0 > self.snake_head[0] or self.snake_head[0] >= self.size or 0 > self.snake_head[1] or self.snake_head[
            1] >= self.size:
            self.game_over = True
            return
        for block in self.snake_body[1:]:
            if self.snake_head == block:
                self.game_over = True
                return
        self.matrix = [[0 for i in range(self.size)] for j in range(self.size)]
        self.matrix[self.food[0]][self.food[1]] = 3
        for part in self.snake_body:
            self.matrix[part[0]][part[1]] = 1
        self.matrix[self.snake_head[0]][self.snake_head[1]] = 2
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
        self.observation_space = Box(low=np.array([[0 for i in range(size)] for j in range(size)]),
                                     high=np.array([[4 for i in range(size)] for j in range(size)]), shape=(size, size))
        self.state = self.snake.matrix
        self.size = size

    def step(self, action):
        current_score = self.snake.score
        current_time = self.snake.survive_time
        self.snake.tick(action)
        reward = self.snake.score - current_score
        done = False
        if self.snake.game_over:
            print("GAMEOVER")
            done = True
        self.state = self.snake.matrix
        info = {}
        return np.array(self.state), reward, done, info

    def reset(self):
        self.snake.__init__(self.snake.size)
        return np.array(self.snake.matrix)

    def render(self, mode="human"):
        self.snake.print_matrix()


class Shape_2_Env(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0 for i in range(9)]), high=np.array([3 for i in range(9)]),
                                     shape=(9,))
        self.state = np.array([0 for i in range(9)])
        self.state[5] = 1
        self.state[1] = 2
        self.timer = 60
        self.loc = 5
        self.pt = 1

    def step(self, action):
        if 0 <= self.loc + action - 1 < 9:
            self.loc += action - 1
        else:
            pass
        self.timer -= 1
        if self.loc == self.pt:
            reward = 1
            _ = False
            while not _:
                self.pt = random.randint(0, 8)
                if self.pt != self.loc:
                    _ = True
        else:
            reward = 0
        done = self.timer <= 0
        info = {}
        self.state = np.array([0 for i in range(9)])
        self.state[self.loc] = 1
        self.state[self.pt] = 2
        return self.state, reward, done, info

    def reset(self):
        self.__init__()
        return self.state

    def render(self, mode="human"):
        print(self.state)


env = Shape_2_Env()
tbp = os.path.join('Training', 'Logs')
sp = os.path.join('Training', 'Saved_Models', 'Shape2Test')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tbp)
model.learn(500000)
for i in range(10):
    obs = env.reset()
    done = False
    score = 0
    survive_time = 0
    while not done:
        step, _ = model.predict(obs)
        obs, reward, done, info = env.step(step)
        score += reward
    print(f"SCORE: {score}")


# 10000 timesteps: 5 7 4 2 8 3 2 5 4 9. Av: 4.89. PPO: Forgo
# 20000 timesteps: 3 5 5 3 3 4 3 4 2 5. Av: 4.11. PPO: Forgo
# 50000 timesteps: 10 10 6 2 8 7 6 2 7 5. Av: 7. PPO: 55
# 100000 timesteps: 11 7 6 2 6 2 3 3 3 7. Av: 5.33. PPO: Forgo
# 200000 timesteps: 8 3 8 18 2 11 11 13 3 6. Av: 9.22. PPO: Forgo
# 500000 timesteps: 19 14 16 23 14 17 15 16 15. Av: 16.56. PPO: 56
