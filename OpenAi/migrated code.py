#Uses a simple neural network with hillclimber and random mutation to weights
#The fitness function returns a score that is an average over several runs but if an individual score is bad it will abandon the run
#There is an additional loop to run the whole trial several times to get an average number of steps to solve
#Can be run as is. Might need to change the filename path

import random
import numpy as np
import matplotlib.pyplot as mpt
import math as ma
import gym
import gym.scoreboard.scoring
from time import time

def MatrixCreate(rows, columns):

    return np.zeros((rows, columns), dtype=np.float64)

def MatrixRandomize(v):

    for c in range(len(v)):

        for r in range(len(v[c])):

            v[c][r] = random.uniform(-1,1)

    return v

def MatrixMutate(v, prob):

    p = np.copy(v)

    for c in range(len(p)):

        for r in range(len(p[c])):

            if prob > random.uniform(0,1):
                p[c][r] = random.uniform(-1,1)

    return p

def Update(neuronValues,synapses,row):
        for i in range(len(neuronValues)):
            neuronValues[row,i] = mods(np.sum(neuronValues[row - 1] * synapses[i]))

def mods(x):
    if x < -1:
        x = -1
    if x > 1:
        x = 1
    return x

#filename = 'C:\\temp\\' + str(time())
filename = 'C:\\temp\\acrobot-v0-2' 

TARGET_FITNESS = -80
MIN_SCORE_ALLOWED = -100 # Dont bother averaging fitness below this score
AVERAGE_N_COUNT = 10 # Number of runs to average the fitness over
MAX_GENERATIONS = 400 
MUTATE_PROBABILITY = 0.2 # How much to randomly mutate weights
SOLUTION_ATTEMPT_COUNT = 150 

AVERAGE_N_COUNT_T_VALUE = 1 # Use this to run several times for an overall average of scores

env = gym.make('Acrobot-v0')

# Fitness evaluation
def AIFitness(env,individual,avgCount, bailThreshold):

    RewardBeforeAvg = 0.0

    for averageit in range(avgCount):

        observation = env.reset()
        cuReward = 0.0

        for t in range(199):

            #Create matrix to hold interim values to calculate NN output for a single run
            neuronValues = MatrixCreate(4,4)   
            neuronValues[0,] = observation   

            #Do the calculation
            for i in range(1,4):
                Update(neuronValues,individual,i)
            
            #Grab the final row output
            actualNeuronValues = neuronValues[3,:]

            #env.render() # uncomment to see the graphic

            w = actualNeuronValues[0]
            x = actualNeuronValues[1]
            y = actualNeuronValues[2]
            #z = actualNeuronValues[3]

            # Map an output to an action - This could be generalised better
            if w>x and w>y:
                action=0
            elif x>w and x>y:
                action=1
            else:
                action=2

            observation, reward, done, info = env.step(action)
        
            cuReward = cuReward + reward

            if done:              
                #print("Episode finished after {} timesteps".format(t + 1))
                break 

        RewardBeforeAvg = RewardBeforeAvg + cuReward

        # Break out of averaging loop if result not good enough
        if cuReward<bailThreshold:
            return RewardBeforeAvg/(averageit+1)
              
    return RewardBeforeAvg/(averageit+1)

# Continue to run the trained solution without further mutation 
def AITest(env,individual):

    observation = env.reset()

    for t in range(200):

        #Create matrix to hold interim values to calculate NN out put for a
        #single run
        neuronValues = MatrixCreate(4,4)   
        neuronValues[0,] = observation   

        #Do the calculation
        for i in range(1,4):
            Update(neuronValues,individual,i)
            
        #Grab the final row
        actualNeuronValues = neuronValues[3,:]

        env.render()

        w = actualNeuronValues[0]
        x = actualNeuronValues[1]
        y = actualNeuronValues[2]
        #z = actualNeuronValues[3]

        if w>x and w>y:
            action=0
        elif x>w and x>y:
            action=1
        else:
            action=2

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode done after {} timesteps".format(t + 1))
            break   

    print("Episode completed {} timesteps".format(t + 1))

#array episode_t_values if running many trials
tvals=[]

for overallaverageloop in range(AVERAGE_N_COUNT_T_VALUE):

    # Create initial synapse matrix

    parent = MatrixCreate(4,4)
    parent = MatrixRandomize(parent)

    parentFitness = AIFitness(env,parent,AVERAGE_N_COUNT,MIN_SCORE_ALLOWED) 
    currBest=0.0

    #Main evo loop

    env.monitor.start(filename,video_callable=lambda i : False,force=True)

    for currentGeneration in range(MAX_GENERATIONS):
        
        child = MatrixMutate(parent,MUTATE_PROBABILITY)
   
        childFitness = AIFitness(env,child,AVERAGE_N_COUNT,MIN_SCORE_ALLOWED)

        print "Parent", parentFitness,"Child",childFitness

        if (childFitness > parentFitness):
        
             parent = child
             parentFitness = childFitness
           
        if parentFitness>=TARGET_FITNESS:
            break

    # Target fitness has been reached. Test the solution
    for testrun in range(SOLUTION_ATTEMPT_COUNT):
        print "new run" + str(testrun)
        AITest(env,parent)
       
    score = gym.scoreboard.scoring.score_from_local(filename)
    tvalue = score['episode_t_value']
    tvals.append(tvalue)
       
    env.monitor.close()

#Output the scoreboard scoring
print gym.scoreboard.scoring.score_from_local(filename)
print "Episode_t_values = " + str(tvals)
print "Average steps to solve over " + str(AVERAGE_N_COUNT_T_VALUE) + " trials = " + str(np.average(tvals))

