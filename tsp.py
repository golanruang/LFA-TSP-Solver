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

"""
July 9th next steps: 
 - create location of cities
 - given locations, generate whole tours 
 - find loss of tour 
 - calculate rest of gradient j formula that's not loss
 - add sum of the gradients 
"""

class TSP:
    def __init__(self, d, cities):
        self.d = d
        self.sample = [0,1,2,4,3]
        self.cities = cities
        self.numCities = 7
        self.currCity = 2
        # self.history = [1,3,6]
        pass 

    def createCities(self):
        pass
    
    def sampleTour(self):
        pass

    def softmaxCities(self, phi):
        """
        adds softmax values to the end of phi list (phi contains distance, current city (int), potential next city (int), and previous city (int)) given phi
        returns new phi list with an extra softmax val at the end
        phi = distance, current city (int), potential next city (int), previous city (int), softmax val
        """
        toSoftmax = []
        for element in phi: 
            toSoftmax.append(element[0])
        softmaxed = softmax(toSoftmax)

        for i in range(len(phi)):
            phi[i].append(softmaxed[i])
        return phi

    def sampleNextCity(self, softmaxCities, history):
        """
        given softmaxes of potential next cities sample a new city and return it's index (int)
        """

        softmaxOnly = []
        for city in softmaxCities:
            softmaxOnly.append(city[-1])
        s = np.random.choice(softmaxOnly)
        index = -1
        print("softmax only: ", softmaxOnly)
        for i in range(len(softmaxCities)): # don't use this list
            if softmaxCities[i][-1]==s:
                index=i

        nextCity = softmaxCities[index][2]
        history.append(self.currCity)
        
        return nextCity

    def getAvailableCities(self, history):
        """
        given history of past cities visited return a list of cities that have not been visited (does not include current city)
        """
        availableCities = []
        for city in self.cities.keys():
            if int(city) not in history and int(city) != self.currCity:
                availableCities.append(city)
        print(availableCities)
        return availableCities

    def phi(self, history):
        """
        history = vector of past cities 
        currentCity = int (current city)
        
        returns a list of lists phi with four elements in each sublist: distance, current city (int), potential next city (int), and previous city (int)
        """
        phi = []
        availableCities = self.getAvailableCities(history)
        for i in range(len(availableCities)):
            currentCity = self.cities[str(self.currCity)]
            nextCity = self.cities[str(availableCities[i])]
            prevCity = self.cities[str(history[-1])]
            dist1 = self.distance(currentCity, prevCity)
            dist2 = self.distance(currentCity, nextCity)
            phi.append([dist1+dist2, self.currCity, availableCities[i], history[-1]])
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
        print("phi in phi(): ", phi)

        print("softmaxed: ", self.softmaxCities(phi))
        nextCity = self.sampleNextCity(self.softmaxCities(phi), history)
        print("next: ", nextCity)
        return phi

    def findPermutations(self):
        permutations = []
        return permutations

    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def plot(path, cities):
    path = [0,5,4,3,2,6,0]
    
def main():
    numCities = 5
    cities = {"0":(0.2,0.2), "1":(0.7,0.4), "2":(0,1,0.6), "3":(0.9,0.9), "4": (0.5,0.4), "5": (0.4,0.3), "6": (0.3,0.2)}
    t = TSP(5, cities) # num cities, stepsize, 
    t.phi([1,3])
main()
