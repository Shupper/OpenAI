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
    #env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    ff=observation.flatten()
    print env.action_space.sample()
    print ("observation:",env.observation_space)
    #env.step(env.action_space.sample()) # take a random action


import numpy as np

NCount = 4
HdCount = 4

input = np.random.random((1,NCount))
synapseWeights = np.random.random((NCount,HdCount))

print input
print synapseWeights

output = np.dot(input,synapseWeights)

print output

