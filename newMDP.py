import numpy as np
import gym 

class MDP(env):
    self.a = -1
    self.numStates = -1
    self.numActions = -1 
    self.alpha = -1
    self.eta = -1
    self.v = [.1,.1,.1,.1,.1]
    self.policy = [0,0]
    self.gamma = 0.23

    def getPhi(self):
        pass

    def algo1(self):
        h = 0
        totalReward=0
        actions = range(self.numActions)
        states = range(self.numStates)
        a = np.random.choice(actions)

        s = np.random.choice(states)
        while self.termination()==False: 
            observation, reward, done, info = env.step()
            totalReward+=reward
            h+=1

        return h, totalReward

    def termination(self):
        choice = np.random.choice(range(2), p=[self.gamma, 1-self.gamma])
        if choice==0:
            return False
        else: 
            return True

    def algo2(self):

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    
m = MDP(env)
