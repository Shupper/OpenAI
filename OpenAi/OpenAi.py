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

allowedEnvs=[]

for x in envs.registry.all():
    #print (x,x.nondeterministic,x.reward_threshold,x.timestep_limit,x.trials,x._entry_point,x._env_name,x._kwargs,x._local_only)

    try:
        env = gym.make(x.id)
        allowedEnvs.append(env)
        #print ("id:",x.id)
        #print ("action:",env.action_space)
        #print ("observation:",env.observation_space)
    except:
        print ("Cannot run id:",x.id)
        #print (sys.exc_info()[0])

for es in allowedEnvs:
    #env = gym.make(es.spec.id)
    print ("id:",es.spec.id)
    #print ("action:",env.action_space)
    #print ("observation:",env.observation_space)