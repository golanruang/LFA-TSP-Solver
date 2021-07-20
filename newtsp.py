from random import randint
import random
import math

import numpy as np
from scipy.special import softmax

import matplotlib.pyplot as plt
import turtle

class TSP:

    def __init__(self, numCities, stepsize, dimension, N):

        self.numCities = numCities
        self.theta = None
        self.stepsize = stepsize
        self.dimensions = dimension # area that you are sampling from
        self.s = self.generateS(numCities)  # should not define an s
        # self.initTheta = 0

    def getNumCities(self):
        return self.numCities

    def getTheta(self):
        return self.theta

    def generateS(self, numCities):         # generate 1 s for every tour
        """generate s given num cities"""
        cities = {}
        cityNum = 0
        xcoord = 0
        # print("self.dimensions: ", self.dimensions)
        for i in range(self.numCities * 2 + 1):
            if i % 2 == 0 and i != 0:
                ycoord = random.uniform(0,self.dimensions[1])
                cities[str(cityNum)] = (xcoord, ycoord)
                cityNum += 1
            else:
                xcoord = random.uniform(0,self.dimensions[0])

        self.s = cities 
        # print('cities: %s'% cities)
        return cities

    def getDistributions(self,phis):
        """
        multiplies theta and phis together to get a value
        """
        # print("in getDistributions()")
        # print("len phis in getDistributions(): %d" % len(phis))
        distributions = []
        for phi in phis:
            # print("phi: %s\nself.theta: %s" % (phis,self.theta))
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
        # make 
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

    # def phi(self, history, s, nextCity):
    #     phis = []
    #     currCityCoords = self.s[history[-1]]              # tuple of floats
    #     nextCityCoords = self.s[nextCity]                 # tuple of floats

    #     for i in range(self.numCities):
    #         cityICoords = self.s[str(i)]
    #         dist_between_each_city_and_current_city = self.distance(currCityCoords, cityICoords)
    #         dist_between_each_city_and_next_city = self.distance(nextCityCoords, cityICoords)
    #         phis.append(dist_between_each_city_and_current_city)
    #         phis.append(dist_between_each_city_and_next_city)

    #     phis.append(self.distance(currCityCoords, nextCityCoords))
    #     phisDimensions = len(phis)
    #     # if self.initTheta == 0:
    #     #     print("theta initialized")
    #     #     self.theta = np.zeros(phisDimensions)
    #     #     self.initTheta+=1

    #     return phis
    
    def initTheta(self,phi):
        self.theta=np.zeros(len(phi))
        
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
    theta=t.getTheta()
    sum = np.zeros(len(theta))
    
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
    theta = t.getTheta()
    logGradP = np.zeros(len(theta))

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
    # print("loss: %f" % loss)
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
    theta=t.getTheta()
    # print("theta: %s" % theta)
    gradient = np.zeros(len(theta))
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

def plotLoss(losses, numCities, dimensions):
    x = range(len(losses))
    y = losses
    plt.scatter(x,y)
    plt.xlabel("iter # for %d cities in %s dimensions" % (numCities, dimensions))
    plt.ylabel("loss")
    plt.show()

def plotRoute(t,numCities): # plot route
    history = ["0"]
    s = {'0': (9.200070937916058, 1.9005297767191398), '1': (3.107303095290762, 1.8626497329658154), '2': (5.425848660358286, 6.928469066883547), '3': (8.735053194060864, 5.946285255161302), '4': (1.4007237643529602, 0.5497217676924737), '5': (1.5459042521287603, 0.506417696664706), '6': (9.25039672166815, 9.04563043724498), '7': (2.8036884209473802, 7.775626335373714), '8': (8.560514079098972, 8.99416288912472), '9': (7.864592556703397, 5.120782550695817), '10': (1.8389584059232467, 4.6461714902557425), '11': (8.287826686609804, 5.113402659615832), '12': (2.2266638965878593, 4.930670246605951), '13': (4.470997163212656, 7.8831473455517065), '14': (3.0592639483499773, 0.18701700755213935), '15': (3.6528325876373993, 7.229186273884473), '16': (4.081826326343508, 5.235047501244864), '17': (6.8473761676371625, 6.281356111736109), '18': (7.707372962949345, 8.165243438194835), '19': (5.937561217053883, 4.206894036309289), '20': (8.42355758940321, 7.271771503800256), '21': (8.636374157094469, 6.408375044592468), '22': (2.3917535073394225, 6.230257552051858), '23': (5.384218080252502, 4.963074690151763), '24': (2.164400799116719, 4.15780528428655)}

    tour = sampleTour(history, s, t)    # permutation of cities

    for i in range(len(tour)-1):
        firstCity = s[str(tour[i])]
        secondCity = s[str(tour[i+1])]
        x_values = [firstCity[0], secondCity[0]]
        y_values = [firstCity[1], secondCity[1]]
        plt.plot(x_values,y_values)
    plt.show()

    print("loss: %f" % t.loss(tour, s))

def main():
    """
    stepsize 100, 0.1, 0.01 all returned the same permutation for s1

    """
    N = 10
    numCities = 25
    stepsize = 0.05
    dimensions = (10,10)

    t = TSP(numCities, stepsize, dimensions, N)
    gradient = np.zeros(numCities)
    tours = []
    
    initS = t.generateS(numCities)
    t.initTheta(t.phi(["0"], initS, "1"))

    losses=updateGradient(t, gradient, tours, N, numCities)
    avgLoss = sum(losses)/N
    print("avg loss: %f" % avgLoss)
    plotLoss(losses, numCities, dimensions)

    plotRoute(t, numCities)

main()
