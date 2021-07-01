from random import randint
import random
"""
first generate locations of cities
then calculate all permutations
Neural networks will give you the phi 
Make phi's manually instead of using NN
For different permutations you have different phis
Based on the way you choose phis the algorithm will behave differently 
Find the best arrangement of phis 
"""

def step():
    pass

def calculatePermutations():
    pass

def placeCities(grid,cities):
    h = len(grid)
    w = len(grid[0])
    numWs=0
    remainder=0
    
    

def createCities(grid,numCities):
    h = len(grid)
    w = len(grid[0])
    total = h*w 
    cities = []
    while len(cities) < 3:
        city = random.randrange(1,total)
        if city not in cities: 
            cities.append(city)

    return cities 

def displayEnv(grid):
    for row in range(len(grid)):
        print([0] * len(grid[0]))

def main():
    numCities = 4 
    h = 7 
    w = 5 
    grid = [[0 for _ in range(w)] for i in range(h)]

    displayEnv(grid)

    cities = createCities(grid,numCities)
    print(cities)
    grid = placeCities(grid,cities)

main()