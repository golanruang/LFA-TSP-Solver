import numpy as np
import random
import sympy

class MDP(numStates, numActions, gamma):
    self.state = -1 # define number of states --> 
    self.action = -1 # 
    self.timestep = 0
    self.gamma = random.random()
    self.totalRewards = 0
    self.policy = -1
    self.numSteps = 100
    self.theta = -1
    self.phi = -1
    self.w = -1 # linear approximation to q function - in order to update policy, you use q function
    self.stepsize = 1

    def ACFunction(self):
        # runs all of the Q and W stuff (including algorithm 1)
        pass

    def setPhi(self, n):
        # returns array of 0 with d dimensions (d is hyperparameter)
        
        X = np.empty(shape=[0, n])

        for i in range(5):
            for j in range(2):
                X = np.append(X, [[i, j]], axis=0)  

        return X

    def algorithm1(self, policy, k, w):
        # wt argmin(E[Q(s,a)-w*phi(s,a)])

        for i in range(k):
            w = w - self.stepsize * argmin(g(w)) 

    def g(self, Q, w, phi, s, a):
        return (Q(s,a, self.policy) - w*phi(s,a))**2

    def phi(s,a):
        pass

    def updateTheta(self, update):
        self.theta = self.theta+update

    def gradientG(self, s, a):
        return -2*phi(s,a) * (q(s,a) - w(s,a))

    def findQ(self):
        # use stochiastic gradient descent to estimate the q function

        pass
    def argmin(self):
        # run gradient descent given a list of numbers
        # w = w-b(delta)G() 
        pass 

    def terminate(self):
        roll = random.random()
        if roll < 1-self.gamma:
            return True
        return False

    def incrementTime(self):
        self.timestep += 1

    def gradient_descent(self, gradient, start, learn_rate, n_iter):
        gradient, start, learn_rate, n_iter = 50, tolerance = 1e-06
        vector = start
        for _ in range(n_iter):
            diff = -learn_rate*gradient(vector)
            if np.all(np.abs(diff) <= tolerance):
                return
            vector+=diff
        return vector 

    def getExpectation(self, array):
        return np.mean(array)

m = MDP(3,3,0.9)

m.ACFunction()
