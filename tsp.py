from random import randint
import random
import math

import numpy as np
from scipy.special import softmax

import turtle

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
    def __init__(self, numCities):

        self.numCities = numCities
        self.cities = self.createCities()
        self.currCity = "0"
        self.theta = [random.Random()] * numCities

    def createCities(self):
        cities = {}
        cityNum = 0
        xcoord = 0
        for i in range(self.numCities * 2 + 1):
            if i%2 == 0 and i != 0:
                ycoord = random.random()
                cities[str(cityNum)] = (xcoord, ycoord)
                cityNum += 1
            else:
                xcoord = random.random()

        return cities
        
    
    def sampleTour(self):
        pass

    def getNextCity(self, history):
        print("history: ", history)
        phi = self.phi(history)
        softmaxed = self.softmaxCities(phi)
        nextCity = self.sampleNextCity(softmaxed, history)
        self.currCity = str(nextCity)
        print("next city: ", str(nextCity))
        return nextCity

    def softmaxCities(self, phi):
        """
        adds softmax values to the end of phi list (phi contains distance, current city (int), potential next city (int), and previous city (int)) given phi
        returns new phi list with an extra softmax val at the end
        phi = distance, current city (int), potential next city (int), previous city (int), softmax val
        """
        # toSoftmax = []
        # for element in phi: 
        #     toSoftmax.append(element[0])
        notzero = []
        for val in phi:
            if val != 0:
                notzero.append(val)
        softmaxed = softmax(notzero)
        print("softmaxed: ", softmaxed)
        P = []
        index = 0
        for i in range(len(phi)):
            if phi[i] == 0:
                P.append(0)
            else: 
                P.append(softmaxed[index])
                index+=1
        print("P: ", P)
        return P

    def sampleNextCity(self, softmaxCities, history):
        """
        given softmaxes of potential next cities sample a new city and return it's index (int)
        """
        print("softmax cities in sampleNextCity: ", softmaxCities)
        print(self.cities.keys())
        s = np.random.choice(range(self.numCities), p=softmaxCities)
        nextCity = s
        
        return nextCity

    """
    def getAvailableCities(self, history):
        
        given history of past cities visited return a list of cities that have not been visited (does not include current city)
        
        availableCities = []
        for city in self.cities.keys():
            if city not in history and city != self.currCity:
                availableCities.append(city)
        # print("history: ", history)
        # print("available cities: ", availableCities)
        return availableCities
    """

    def getCities(self):
        return self.cities

    def phi(self, history):
        """
        history = vector of past cities 
        currentCity = int (current city)

        keep dimension returned by phi constant (num cities dimensional) - just set phi as 0 if in history
        
        returns a list of lists phi with four elements in each sublist: distance, current city (int), potential next city (int), and previous city (int)
        """
        phis = []
        for i in range(self.numCities):
            if str(i) in history:
                phis.append(0)
            else:
                currentCity = self.cities[str(self.currCity)]
                nextCity = self.cities[str(i)]
                if len(history) == 0:
                    prevCity = currentCity
                else:
                    # print(str(history[-1]))
                    prevCity = self.cities[str(history[-1])]
                dist1 = self.distance(currentCity, prevCity)
                dist2 = self.distance(currentCity, nextCity)
                # print(dist1+dist2)
                phis.append(dist1+dist2)  
        # print("phis: ", phis)
        return phis

    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    def getTheta(self):
        return self.theta

    def setTheta(self,theta):
        self.theta = theta
    
    def loss(self, history):
        loss = 0
        for i in range(len(history)-1):
            city1 = history[i]
            city2 = history[i+1]
            dist = self.distance(self.cities[city1], self.cities[city2])
            loss += dist
        return loss

def plot(path, cities):
    path = [0,5,4,3,2,6,0]
    wn = turtle.Screen()
    wn.bgcolor("")
    wn.title("Travelling Salesman Problem")
    
def findGradient(tours, t):
    tempHist = ["0"]
    gradient = 0
    for tour in tours:
        gradient = t.loss(tour)
        print("tour: ", tour)
        for city in range(len(tour)-1):
            currentCity = tour[int(city)]
            nextCity = tour[int(city)+1]
            phis = t.phi(tempHist)
            if str(city) not in tempHist: 
                sub = t.softmaxCities(phis[int(city)])


            tempHist.append(city)




def main():
    numCities = 10
    #  = {"0":(0.2,0.2), "1":(0.7,0.4), "2":(0,1,0.6), "3":(0.9,0.9), "4": (0.5,0.4), "5": (0.4,0.3), "6": (0.3,0.2)}
    # history = ["0","6","3","4","1","2","0"]
    history = ["0"]
    loss = 0
    t = TSP(numCities) # num cities, stepsize, 
    tourNum = 1
    gradient = 0
    n = 10
    tours = []
    for i in range(n): # sample n tours
        for i in range(numCities-1): # sample a tour
            history.append(str(t.getNextCity(history)))
        history.append("0")
        print("tour num %d: " % tourNum + str(history))
        print("loss: ", loss)
        tours.append(history)
        history = ["0"]
        tourNum+=1
    # print(tours)
    gradient = findGradient(tours, t)
main()
