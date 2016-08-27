import sys
import gym
from gym import envs
import random
import numpy as np
#import matplotlib.pyplot as mpt
import math as ma
import gym.scoreboard.scoring
from time import time
import Box2D


env = gym.make('CarRacing-v0')
env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample()) # take a random action


