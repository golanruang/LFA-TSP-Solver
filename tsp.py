from random import randint
import random
import math

import numpy as np
from scipy.special import softmax

"""
first generate locations of cities
then calculate all permutations
Neural networks will give you the phi 
Make phi's manually instead of using NN
For different permutations you have different phis
Based on the way you choose phis the algorithm will behave differently 
Find the best arrangement of phis 
"""

class TSP:
    def __init__(self, d):
        self.d = d
        self.sample = [0,1,2,4,3,0]
        self.cities = {"0":(0.2,0.2), "1":(0.7,0.4), "2":(0,1,0.6), "3":(0.9,0.9), "4": (0.5,0.4)}
        self.numCities = 5
        pass 

    def createCities(self):
        pass
    
    def sampleTour(self):
        pass

    def softmaxCities(self, phi):
        toSoftmax = []
        for element in phi: 
            toSoftmax.append(element[0])
        softmaxed = softmax(toSoftmax)
        print(softmaxed)
        for i in range(len(phi)):
            phi[i].append(softmaxed[i])
        return phi

    def sampleNextCity(self, softmaxCities):

        softmaxOnly = []
        for city in softmaxCities:
            softmaxOnly.append(city)
        s = np.random.choice(softmaxOnly)
        index = -1
        for i in range(len(softmaxOnly)):
            if softmaxOnly[i]==s:
                index=i

        nextCity = softmaxCities[index][2]
        return nextCity

    def getAvailableCities(self, history):
        availableCities = []
        for city in self.cities.keys():
            if city not in history:
                availableCities.append(city)
        
        return availableCities

    def phi(self, history):
        """
        history = vector of past cities 
        currentCity = int (current city)
        
        returns a list of lists phi with four elements in each sublist: distance, current city (int), next city (int), and previous city (int)
        """
        currCity = 2
        phi = []
        availableCities = self.getAvailableCities(history)
        for i in range(len(availableCities)):
            print("currCity: ", currCity)
            currentCity = self.cities[str(currCity)]
            print("CurrentCity: ", type(currentCity))
            nextCity = self.cities[str(i)]
            prevCity = self.cities[str(history[-1])]
            dist1 = self.distance(currentCity, prevCity)
            dist2 = self.distance(currentCity, nextCity)
            phi.append([dist1+dist2, currCity, i, prevCity])
        # for i in range(len(availableCities)):
        #     for j in range(len(availableCities)):
        #         print("availableCities[i]: ", availableCities[i])
        #         currCity = self.cities[str(availableCities[i])]
        #         print("currCity: ", currCity)
        #         iterCity = self.cities[str(availableCities[j])]
        #         print("iterCity: ", iterCity)
        #         dist1 = self.distance(currCity, iterCity)
        #         dist2 = self.distance(currCity, nextCity)
        #         print("dist1: ", dist1)
        #         print("dist2: ", dist2)
        #         phi.append((dist1 + dist2,j))
        #     phis.append((phi,currCity))
        #     phi=[]
        # print(phis)
        print(phi)
        return phi

    def findPermutations(self):
        permutations = []
        return permutations

    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def main():
    numCities = 5
    t = TSP(5)
    t.phi([1,3])
main()
