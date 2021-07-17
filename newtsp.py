from random import randint
import random
import math

import numpy as np
from scipy.special import softmax

import matplotlib.pyplot as plt
import turtle

class TSP:

    def __init__(self, numCities, stepsize, dimensions, N):

        self.numCities = numCities
        self.theta = np.zeros(numCities)
        self.s = self.generateS(numCities)  # should not define an s
        self.stepsize = stepsize
        self.dimensions = dimensions # area that you are sampling from


    def getNumCities(self):
        return self.numCities

    def getTheta(self):
        return self.theta

    def generateS(self, numCities):         # generate 1 s for every tour
        """generate s given num cities"""
        cities = {}
        cityNum = 0
        xcoord = 0
        for i in range(self.numCities * 2 + 1):
            if i % 2 == 0 and i != 0:
                ycoord = random.random()
                cities[str(cityNum)] = (xcoord, ycoord)
                cityNum += 1
            else:
                xcoord = random.random()

        self.s = cities 
        return cities

    def getDistributions(self,phis):
        """
        multiplies theta and phis together to get a value
        """
        # print("in getDistributions()")
        # print("len phis in getDistributions(): %d" % len(phis))
        distributions = []
        for phi in phis:
            phi = np.dot(phi, self.theta)
            distributions.append(phi)
        
        # print("distributions: ", distributions)
        return distributions                                 # 1 x numCities shaped array

    def softmaxCities(self, history, distributions):
        """
        set softmax as 0 if city is in history - take softmax of distribution and sample
        make sure this works
        """
        # print("len distributions %d" % len(distributions))
        toSoftmax = []
        for i in range(len(distributions)):         # for each city (i = city)
            if str(i) in history:                   # if city in history
                distributions[i] = 0                # set the distribution as 0
            else:
                toSoftmax.append(distributions[i])  # else softmax the value
        # print("to softmax: ", toSoftmax)
        softmaxed = softmax(toSoftmax)
        P = []
        index = 0
        for i in range(len(distributions)):
            if str(i) in history:
                P.append(0)
            else:
                P.append(softmaxed[index])
                index += 1
        # print("P: ", P)
        return P

    def sampleCities(self, softmaxes):
        """
        takes sample of cities from softmax cities
        """
        # print("softmaxes: ", softmaxes)
        nextCity=np.random.choice(range(self.numCities), p=softmaxes)
        return nextCity 

    def phi(self, history, s, nextCity):
        """
        returns 1xnumCities vector of phis
        """
        phis = []
        currCityCoords = self.s[history[-1]]              # tuple of floats
        nextCityCoords = self.s[nextCity]                 # tuple of floats

        for i in range(self.numCities):   
            dist1 = self.distance(currCityCoords, nextCityCoords)
            dist2 = self.distance(currCityCoords, self.s[str(i)])
            # print("dist 1: %f, dist 2: %f" % (dist1, dist2))
            phis.append(dist1+dist2)

        # print("phis: ", phis)
        # print("len phis in phi(): %s" % len(phis))
        return phis

    def distance(self, p0, p1):
        """
        distance between two points (tuples of floats)
        """
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    def getS(self):
        return self.s

    def setS(self, s):
        self.s = s

    def setNumCities(self,numCities):
        self.numCities = numCities

    def loss(self, history, s):
        loss = 0
        for i in range(len(history)-1):
            city1 = history[i]
            city2 = history[i+1]
            dist = self.distance(s[city1], s[city2])
            loss += dist
        return loss
    
    def updateTheta(self, update):
        self.theta = self.theta - np.multiply(self.stepsize, update)

def calculateSum(t, history, s):
    numCities = t.getNumCities()
    sum = np.zeros(numCities)
    Pphis = []
    
    for i in range(numCities):
        nextCity = str(i)
        if nextCity not in history:
            phi = t.phi(history, s, nextCity)
            for j in range(numCities):
                Pphi = t.phi(history, s, str(j))
                Pphis.append(Pphi)

            distributions = t.getDistributions(Pphis)
            softmaxed = t.softmaxCities(history, distributions)
            p = softmaxed[i]

            toSum = np.multiply(p, phi)
            sum = np.add(sum, toSum)
    
    return sum

def sampleTour(history, s, t):
    phis = []
    numCities = t.getNumCities()
    t.setS(s)
    for i in range(numCities-1):
        for j in range(numCities):
            phi = t.phi(history, s, str(j))
            phis.append(phi)

        distributions = t.getDistributions(phis)
        # print("distributions: ", distributions)
        softmaxed = t.softmaxCities(history, distributions)
        nextCity = t.sampleCities(softmaxed)
        phis = []
        history.append(str(nextCity))

    history.append("0")
    return history

    # print("history: ", history)

def logP(t, tour, s):
    history = ["0"]
    Pphis = []
    logGradP = np.zeros(len(tour)-1)

    for i in range(len(tour)-1):                      # for every city i
        currentCity = tour[i]
        nextCity = tour[i+1]
        phi = t.phi(history, s, nextCity)
        sum = calculateSum(t, history, s)

        history.append(nextCity)
        oneIter=np.subtract(phi, sum)
        logGradP=np.add(logGradP, oneIter)

    return logGradP

def bigTerm(t, tour, s):
    loss = t.loss(tour, s)
    print("loss: %f" % loss)
    log_gradient_P = logP(t, tour, s)

    return np.multiply(loss, log_gradient_P), loss

def updateGradient(t, gradient, tours, N, numCities):
    """
    # initialize sum variable outside for
    # for i in range(N):
    # sample S and then sample pi(i)
    # calculate big term
    # add to sum
    """
    print("calculating gradient...")
    gradient = np.zeros(numCities)
    history = ["0"]
    losses=[]

    for i in range(N):
        s=t.generateS(numCities)
        tour = sampleTour(history, s, t)
        #print("tour: %s" % tour)
        term, loss = bigTerm(t, tour, s)
        #print("big term: %s" % term)
        gradient=np.add(gradient,term)
        gradient=gradient/N
        losses.append(loss)
        t.updateTheta(gradient)
        history=["0"]

    print("theta: ", t.getTheta())
    return losses

def plotLoss(losses, numCities):
    x = range(len(losses))
    y = losses
    plt.scatter(x,y)
    plt.xlabel("iter # for %d cities" % numCities)
    plt.ylabel("loss")
    plt.show()

def main():

    N = 10
    numCities = 50
    stepsize = 1
    dimensions = (2,2)
    t = TSP(numCities, stepsize, dimensions, N)
    history = ["0"]
    numTours = 10000
    gradient = np.zeros(numCities)
    tours = []

    losses=updateGradient(t, gradient, tours, N, numCities)
    plotLoss(losses, numCities)

main()
