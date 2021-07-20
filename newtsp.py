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

    def setTheta(self, theta):
        self.theta = theta

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
    #     phis = [
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
    k=4
    
    # two loops - inner loop 
    for j in range(k):                          # update theta
        print("k: %d"%j)
        for i in range(N):                      # estimate gradient
            s=t.generateS(numCities)
            tour = sampleTour(history, s, t)
            #print("tour: %s" % tour)
            term, loss = bigTerm(t, tour, s)
            #print("big term: %s" % term)
            term = term/N
            gradient=np.add(gradient,term)
            losses.append(loss)

            history=["0"]
        t.updateTheta(gradient)
        gradient = np.zeros(len(theta))

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
    # s = {'0': (9.200070937916058, 1.9005297767191398), '1': (3.107303095290762, 1.8626497329658154), '2': (5.425848660358286, 6.928469066883547), '3': (8.735053194060864, 5.946285255161302), '4': (1.4007237643529602, 0.5497217676924737), '5': (1.5459042521287603, 0.506417696664706), '6': (9.25039672166815, 9.04563043724498), '7': (2.8036884209473802, 7.775626335373714), '8': (8.560514079098972, 8.99416288912472), '9': (7.864592556703397, 5.120782550695817), '10': (1.8389584059232467, 4.6461714902557425), '11': (8.287826686609804, 5.113402659615832), '12': (2.2266638965878593, 4.930670246605951), '13': (4.470997163212656, 7.8831473455517065), '14': (3.0592639483499773, 0.18701700755213935), '15': (3.6528325876373993, 7.229186273884473), '16': (4.081826326343508, 5.235047501244864), '17': (6.8473761676371625, 6.281356111736109), '18': (7.707372962949345, 8.165243438194835), '19': (5.937561217053883, 4.206894036309289), '20': (8.42355758940321, 7.271771503800256), '21': (8.636374157094469, 6.408375044592468), '22': (2.3917535073394225, 6.230257552051858), '23': (5.384218080252502, 4.963074690151763), '24': (2.164400799116719, 4.15780528428655)}
    # 25 cities
    # s={'0': (4.879095977888034, 7.849790697317993), '1': (0.888191686973826, 5.387642853229218), '2': (4.484485692634052, 5.758176984372398), '3': (1.3750802460099487, 2.733084263846175), '4': (4.8407729225564164, 8.265801992628873), '5': (4.726951492208814, 9.251572660068994), '6': (1.1370740664408008, 1.510882603117808), '7': (8.849412980854085, 4.363353764752579), '8': (9.278115708868242, 6.741119558290397), '9': (2.659333823915986, 2.991943361885455), '10': (4.07797886339555, 6.6448638679660625), '11': (6.447209829695806, 9.62454978382974), '12': (0.09601900927804885, 9.08828233613026), '13': (6.7752997191551225, 5.2241024033186925), '14': (4.209906797028435, 0.07358625235730165), '15': (4.2122035988272, 3.7176980777232416), '16': (8.077036235309347, 3.7009940433573063), '17': (1.5613747910947728, 3.2342688608420933), '18': (4.0821624723381635, 6.64391304626961), '19': (2.1037681984285497, 5.399381372888785), '20': (1.92445867472686, 3.4372004522739297), '21': (2.755337856570671, 0.5463708304921566), '22': (4.418392496795359, 8.811751505893193), '23': (2.961961073817548, 4.036206579622887), '24': (6.330875418416793, 4.3183855443785575), '25': (2.196747153826294, 0.17335926766013365), '26': (1.714693965726879, 9.940465128811406), '27': (2.5656684578333033, 7.6214908171778974), '28': (8.85540605495836, 4.467220298952185), '29': (9.208810261845574, 0.13217393173061476), '30': (0.03592239134985742, 6.784578584122601), '31': (2.4920100695209415, 6.063858072854949), '32': (5.450456130269372, 5.132255588975655), '33': (3.7525462510871908, 7.144366451113053), '34': (3.616551300790296, 5.852689814592074), '35': (1.0421534002618482, 4.899202966154133), '36': (7.431299930551435, 0.1380365929632199), '37': (9.41454015246633, 8.846608299240584), '38': (9.622605290241932, 6.775551544627552), '39': (4.138122295993155, 2.9014977788017227), '40': (4.027000988437139, 5.993206571040335), '41': (6.814143708593888, 7.3432580033055554), '42': (5.174681046382899, 4.85901876383259), '43': (4.83210089339082, 6.119915693393048), '44': (8.819276518490838, 1.2104843454811642), '45': (5.452600253963941, 7.812390903721271), '46': (0.09808301971990563, 4.81519930372908), '47': (9.055503413885786, 6.8205156199763515), '48': (5.485345914160543, 1.8364123410513056), '49': (6.308721318099595, 4.234648410687967)}
    # 50 cities
    s={'0': (8.952319373406947, 5.800337761014988), '1': (3.5241918689174767, 3.5786528819962857), '2': (0.5250243320510839, 6.883099506526182), '3': (9.926570263061741, 0.7410044591467502), '4': (1.691038914592966, 1.357631567864882), '5': (5.139173859303917, 9.742485346814586), '6': (2.5004907977481414, 4.320036538074933), '7': (6.01286941258774, 5.762377207938558), '8': (8.152658682990797, 9.548808567053062), '9': (1.0283120816871472, 1.3065593267267106), '10': (1.43331978410983, 4.938316732270484), '11': (6.73511663520058, 4.6704284035344), '12': (0.6388093621977675, 7.363605803153601), '13': (9.327178395301024, 2.105613869126147), '14': (2.0809180153065276, 3.612737756235483), '15': (7.35909527704322, 6.438087216277987), '16': (0.5584208499157484, 1.2946379833563737), '17': (7.845147886606441, 5.953476575577399), '18': (7.255684539484673, 6.429337462239819), '19': (7.381648327426159, 8.890201799239742), '20': (8.438700576955354, 0.1062950913704297), '21': (1.669297571243853, 6.737153285447523), '22': (8.793430379245212, 0.7441302630292745), '23': (9.595443444894386, 5.73020608634727), '24': (8.378680223145833, 4.192719245980063), '25': (7.84785727341424, 9.50440418786783), '26': (3.238574650326269, 3.6834972730490847), '27': (0.8180848969200738, 0.4630685853987604), '28': (5.343615729022707, 8.15053623572741), '29': (5.167307073085836, 3.935934465650285)}

    fig, axs = plt.subplots(2)

    tour = sampleTour(history, s, t)    # permutation of cities

    for i in range(len(tour)-1):
        firstCity = s[str(tour[i])]
        secondCity = s[str(tour[i+1])]
        x_values = [firstCity[0], secondCity[0]]
        y_values = [firstCity[1], secondCity[1]]
        axs[0].plot(x_values,y_values)

    print("loss: %f" % t.loss(tour, s))

    t.setTheta(np.zeros(len(t.getTheta())))

    history = ["0"]
    tour = sampleTour(history, s, t)    # permutation of cities

    for i in range(len(tour)-1):
        firstCity = s[str(tour[i])]
        secondCity = s[str(tour[i+1])]
        x_values = [firstCity[0], secondCity[0]]
        y_values = [firstCity[1], secondCity[1]]
        axs[1].plot(x_values,y_values)

    print("loss: %f" % t.loss(tour, s))

    plt.show()

def main():
    """
    stepsize 100, 0.1, 0.01 all returned the same permutation for s1

    """
    N = 4
    numCities = 25
    stepsize = 0.05
    dimensions = (10,10)

    t = TSP(numCities, stepsize, dimensions, N)
    gradient = np.zeros(numCities)
    tours = []
    
    initS = t.generateS(numCities)
    t.initTheta(t.phi(["0"], initS, "1"))
    # print(initS)

    losses=updateGradient(t, gradient, tours, N, numCities)
    avgLoss = sum(losses)/N
    print("avg loss: %f" % avgLoss)
    plotLoss(losses, numCities, dimensions)

    plotRoute(t, numCities)

main()
