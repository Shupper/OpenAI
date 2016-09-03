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

def MatrixMutate(v, prob):

    p = np.copy(v)

    for c in range(len(p)):

        for r in range(len(p[c])):

            if prob > random.uniform(0,1):
                p[c][r] = random.uniform(-1,1)

    return p

def applyThreshold(value, threshold):

    if(value>threshold):
        return 1
    else:
        return 0

def getAction(input,synapseWeights, type): # 1=Discrete, 2=Box Discrete, 3=Box Cont

    output= np.dot(input,synapseWeights)
    if(type==1):
        return np.argmax(output)
        #return np.sum(map(applyThreshold,output, [0]))
    elif(type==2):
        return map(applyThreshold,output, [0])
    else:
        return output

def testFitness(env, synapseWeights, actionType, showPlay=False):

    cuReward = 0.0
    observation = env.reset()
    input = observation.flatten()
    action = getAction(input,synapseWeights,actionType)

    for i in range(1000):
        if(showPlay): env.render() # uncomment to see the graphic
        observation, reward, done, info = env.step(action)
        cuReward = cuReward + reward
        input = observation.flatten()
        action = getAction(input,synapseWeights,actionType)
        #print action
        if done:
            print("Episode finished after {} timesteps".format(i + 1))
            break 
    return cuReward

import numpy as np

#Set params
TARGET_FITNESS = 200
MIN_SCORE_ALLOWED = -100 # Dont bother averaging fitness below this score
AVERAGE_N_COUNT = 10 # Number of runs to average the fitness over
MAX_GENERATIONS = 1000 
MUTATE_PROBABILITY = 0.2 # How much to randomly mutate weights

filename = 'C:\\temp\\acrobot-v0-2' 
NCount = 8 #Based on observation space
HdCount = 3 #Based on action space
env = gym.make('LunarLander-v2')

print(env.observation_space.low)
print(env.observation_space.high)
for ii in range(10):
    print(env.action_space.sample())

#Start the training
parentSynapseWeights = np.random.uniform(-1,1,(NCount,HdCount))
currBestReward = 0.0

env.monitor.start(filename,video_callable=lambda i : False,force=True)
parentFitness = -1000

#Main evo loop
for currentGeneration in range(MAX_GENERATIONS):
        
    childSynapseWeights = MatrixMutate(parentSynapseWeights,MUTATE_PROBABILITY)
   
    childFitness = testFitness(env, childSynapseWeights,1)
    #childFitness = AIFitness(env,child,AVERAGE_N_COUNT,MIN_SCORE_ALLOWED)

    print "Parent", parentFitness,"Child",childFitness

    if (childFitness > parentFitness):
        
            parentSynapseWeights = childSynapseWeights
            parentFitness = childFitness
           
    if parentFitness>=TARGET_FITNESS:
        break

for aa in range(5):
    raw_input("Press Enter to continue...")
    print testFitness(env, parentSynapseWeights,1,True)



#print getAction(input,synapseWeights)



