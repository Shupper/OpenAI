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

#env = gym.make('CarRacing-v0')
#env.reset()
#for _ in range(10):
#    #env.render()
#    observation, reward, done, info = env.step(env.action_space.sample())
#    ff=observation.flatten()
#    print env.action_space.sample()
#    print ("observation:",env.observation_space)
#    #env.step(env.action_space.sample()) # take a random action

import numpy as np

#Set params
TARGET_FITNESS = -80
MIN_SCORE_ALLOWED = -100 # Dont bother averaging fitness below this score
AVERAGE_N_COUNT = 10 # Number of runs to average the fitness over
MAX_GENERATIONS = 400 
MUTATE_PROBABILITY = 0.2 # How much to randomly mutate weights

filename = 'C:\\temp\\acrobot-v0-2' 
NCount = 4 #Based on observation space
HdCount = 4 #Based on action space
env = gym.make('Acrobot-v0')

#Start the training
parentSynapseWeights = np.random.uniform(-1,1,(NCount,HdCount))
currBestReward = 0.0

env.monitor.start(filename,video_callable=lambda i : False,force=True)
ParentFitness = 0

#Main evo loop
for currentGeneration in range(MAX_GENERATIONS):
        
    childSynapseWeights = MatrixMutate(parentSynapseWeights,MUTATE_PROBABILITY)
   
    childFitness = testFitness(env, childSynapseWeights)
    #childFitness = AIFitness(env,child,AVERAGE_N_COUNT,MIN_SCORE_ALLOWED)

    print "Parent", parentFitness,"Child",childFitness

    if (childFitness > parentFitness):
        
            parent = child
            parentFitness = childFitness
           
    if parentFitness>=TARGET_FITNESS:
        break



def MatrixMutate(v, prob):

    p = np.copy(v)

    for c in range(len(p)):

        for r in range(len(p[c])):

            if prob > random.uniform(0,1):
                p[c][r] = random.uniform(-1,1)

    return p

def getAction(input,synapseWeights):
    return np.dot(input,synapseWeights)

def testFitness(env, synapseWeights):

    cuReward = 0.0
    observation = env.reset()
    input = observation.flatten()
    action = getAction(input,synapseWeights)

    for i in range(1000):
        observation, reward, done, info = env.step(action)
        cuReward = cuReward + reward
        input = observation.flatten()
        action = getAction(input,synapseWeights)

    return cuReward


#print getAction(input,synapseWeights)



