import gymnasium as gym
import gym_xarm
import numpy as np

env = gym.make("gym_xarm/XarmLift-controlled-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    action[0:7] = 0.0
    action[0] = 1.0
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
