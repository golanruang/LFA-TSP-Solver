import numpy as np
import gym 
from scipy.special import softmax

class MDP:
    def __init__(self, env):
        self.alpha = 1
        self.numStates = 1000
        self.numActions = 2
        self.eta = 1
        self.v = [.1,.1,.1,.1,.1]
        self.policy = [0,0]
        self.gamma = 0.99
        self.env = env
        self.d = 8

    def algo1(self):
        h = 0
        totalReward=0
        # actions = range(self.numActions)
        # states = range(self.numStates)
        # a = np.random.choice(actions)
        # s = np.random.choice(states)

        observation = self.env.reset()
        action = self.env.action_space.sample()
        # print(observation)
        while self.termination()==False: 
            # self.env.render()
            policy = self.getPolicy(observation, action)
            # print(policy)
            # action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            if done == True:
                self.env.reset()
            # action = self.env.action_space.sample(getPolicy())
            # get all available actions and sample from there using policy
            # print(action)
            totalReward+=reward
            h+=1

        # print("sh: ", observation)
        # print("ah: ", action)
        # print("time steps taken: ", h)
        print("total reward: ", totalReward)
        # return sh and ah here instead of h
        return observation, action, h, totalReward

    def getPolicy(self, s, a):
        """
        pass array with length numActions with phi's for every numAction
        returns array with length numActions that is distribution over actions you can take
        """
        phis = []
        for i in range(self.numActions):
            phis.append(self.phi(s, a))

        # print("softmax: ", softmax(phis))
        return softmax(phis)
        # e_phis = np.exp(phis * self.theta() - np.max(phis * self.theta()))
        # softmax = e_phis / e_phis.sum(axis=0)
        # print("softmax: ", softmax)
        # return softmax

    def termination(self):
        choice = np.random.choice([0,1], p=[self.gamma, 1-self.gamma])
        # print(choice)
        if choice==0:
            # print("termination returned False")
            return False
        else: 
            # print("termination returned True")
            return True

    def algo2(self):
        theta=0
        for t in range(T):
            totalw = 0
            for n in range(N):
                w = 0
                h, reward = algo1()
                # TODO: How to make phi function + alpha function
                # what is the s,a passed to phi - does phi change?? 
                totalw+=w-2*self.alpha * (w*phi(s,a) - algo1())*phi(s,a)
            wt = totalw/N
            theta = theta + self.stepsize * wt
        return theta

    def phi(self, s, a):
        """
        
        """
        # print("action: ", a)
        for i in range(self.numActions):
            v = np.zeros(self.d)
            for i in v:
                if a==0: 
                    v[0:4]=s[0:4]
                else:
                    if a==1: 
                        v[4:8] = -1 * s[0:4]
        # print("v: ", v)
        return v

    def theta(self):
        return np.arange(self.d)

env = gym.make('CartPole-v0')
m = MDP(env)

totalS = 0
totalA = 0
totalH = 0
totalR = 0

for i in range(1000):
    s, a, h, r = m.algo1()
    totalH+=h 
    totalR+=r
    print("iter #: ", i)

print("avg H: ", totalH/1000)
print("avg R: ", totalR/1000)
