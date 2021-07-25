from newtsp import TSP

from random import randint
# import random
# import math

import numpy as np
from scipy.special import softmax

import matplotlib.pyplot as plt
# import turtle

# import os

# import pandas as pd
# import descartes
# import geopandas as gpd
# from shapely.geometry import Point, Polygon

import time

def calculateSum(t, history, s):
    start=time.time()
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
            # print("len Pphis: ", len(Pphis))
            distributions = t.getDistributions(Pphis)
            # print("distributions: ", distributions)
            softmaxed = t.softmaxCities(history, distributions)
            p = softmaxed[i]

            toSum = np.multiply(p, phi)
            sum = np.add(sum, toSum)
        Pphis=[]

    # end=time.time()
    # print("time taken in calculateSum(): %f" % (end-start))
    
    return sum

def sampleTour(history, s, t):
    # takes around 0.8 seconds
    # start=time.time()
    phis = []
    numCities = t.getNumCities()
    t.setS(s)
    for i in range(numCities-1):
        for j in range(numCities):
            # print("j: %d " % j)
            phi = t.phi(history, s, str(j))
            phis.append(phi)
        #print("phis: %s" % phis)
        distributions = t.getDistributions(phis)
        # print("distributions: ", distributions)
        softmaxed = t.softmaxCities(history, distributions)
        nextCity = t.sampleCities(softmaxed)
        phis = []
        history.append(str(nextCity))
    #history.append("0")                                         JUST CHANGED
    # end=time.time()
    # print("time taken in sampleTour(): %f" % (end-start))
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
        sum = calculateSum(t, history, s)             # this function takes too much time
        history.append(nextCity)
        oneIter=np.subtract(phi, sum)
        logGradP=np.add(logGradP, oneIter)

    return logGradP

def bigTerm(t, tour, s):
    """
    this function takes way too long - like 11-12 seconds
    """
    loss = t.loss(tour, s)
    # print("loss: %f" % loss)
    start=time.time()
    log_gradient_P = logP(t, tour, s)       # it's this function
    end=time.time()
    print("log time: %f" % (end-start))

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
    k=3
    
    for j in range(k):                          # update theta
        print("k: %d"%j)
        for i in range(N):                      # estimate gradient
            s=t.generateS(numCities)
            tour = sampleTour(history, s, t)
            #print("tour: %s" % tour)
            start=time.time()
            term, loss = bigTerm(t, tour, s)
            end=time.time()
            print("big term time: %f" % (end-start))
            #print("big term: %s" % term)
            term = term/N
            gradient=np.add(gradient,term)
            losses.append(loss)

            history=["0"]
        t.updateTheta(gradient)
        gradient = np.zeros(len(theta))

    # print("theta: ", t.getTheta())
    return losses

def plotLoss(losses, numCities, dimensions):
    x = range(len(losses))
    y = losses
    plt.scatter(x,y)
    plt.xlabel("iter # for %d cities in %s dimensions" % (numCities, dimensions))
    plt.ylabel("loss")
    plt.show()

def plotRoute(t,numCities,N, s): # plot route
    history = ["0"]
    # s = {'0': (9.200070937916058, 1.9005297767191398), '1': (3.107303095290762, 1.8626497329658154), '2': (5.425848660358286, 6.928469066883547), '3': (8.735053194060864, 5.946285255161302), '4': (1.4007237643529602, 0.5497217676924737), '5': (1.5459042521287603, 0.506417696664706), '6': (9.25039672166815, 9.04563043724498), '7': (2.8036884209473802, 7.775626335373714), '8': (8.560514079098972, 8.99416288912472), '9': (7.864592556703397, 5.120782550695817), '10': (1.8389584059232467, 4.6461714902557425), '11': (8.287826686609804, 5.113402659615832), '12': (2.2266638965878593, 4.930670246605951), '13': (4.470997163212656, 7.8831473455517065), '14': (3.0592639483499773, 0.18701700755213935), '15': (3.6528325876373993, 7.229186273884473), '16': (4.081826326343508, 5.235047501244864), '17': (6.8473761676371625, 6.281356111736109), '18': (7.707372962949345, 8.165243438194835), '19': (5.937561217053883, 4.206894036309289), '20': (8.42355758940321, 7.271771503800256), '21': (8.636374157094469, 6.408375044592468), '22': (2.3917535073394225, 6.230257552051858), '23': (5.384218080252502, 4.963074690151763), '24': (2.164400799116719, 4.15780528428655)}
    # 25 cities
    # s = {'0': (5.940449918010495, -103.63050033068373), '1': (4.170301808717579, -58.462187434152995), '2': (10.188257841158002, -4.588049636751361), '3': (13.10132308609155, -80.99145177096966), '4': (5.935758059481373, -50.06392918970092), '5': (13.837319614117488, -94.52656065810545), '6': (18.09357821061293, -93.67303487268154), '7': (18.834848082819416, -94.84031565919787), '8': (28.468500939997597, -33.46050781989564), '9': (16.788685001420514, -84.88950156964904), '10': (6.8044716995868955, -102.72683247260795), '11': (7.787016888461977, -1.2769223083056551), '12': (18.13535597320498, -37.75977028234969), '13': (20.264088055473145, -103.34839505912696), '14': (28.08692582251099, -98.18235587292182), '15': (10.6296623857429, -7.827119823514838), '16': (1.7661663453796912, -116.89561872613056), '17': (15.450379566451279, -4.731634592823961), '18': (23.070425676416317, -40.90320297061526), '19': (4.33928896501506, -94.27577005250072), '20': (12.412726893933025, -8.766196009082755), '21': (12.21625419182592, -8.875007305320247), '22': (21.98552219265104, -37.709827173644356), '23': (14.22945961399815, -62.936380038745426),'24': (17.779357591835762, -60.83437325882671), '25': (20.275521495897376, -5.9448165014705046), '26': (18.79624899306891, -94.79849220789242), '27': (26.894885791873875, -27.903499756912904), '28': (14.770323250116249, -104.01361816190126), '29': (24.659927532767846, -6.720055328659615), '30': (7.220041255919953, -15.782854632422978), '31': (10.89188096083964, -43.627827732033744), '32': (18.251944260927043, -94.88549114535952), '33': (1.6671179837588697, -112.27952908069115), '34': (9.171851022500347, -75.00863917908852), '35': (22.98398448122153, -12.801451196290355), '36': (10.269731895349459, -73.1537525890891), '37': (27.870492291679568, -111.87006683337738), '38': (26.587034873210726, -50.37861581749792), '39': (16.743443679241576, -82.2312682053974), '40': (22.027217267180237, -61.92407807040082), '41': (10.629445862986763, -25.59273703604983), '42': (19.194177434251536, -2.5447913515797667), '43': (25.559977061026444, -82.93349697076648), '44': (8.929110095851115, -8.665276602016267), '45': (13.047879980269396, -110.05656855024334), '46': (29.172455181621306, -85.03405677963534), '47': (26.8264896352283, -6.098396333418047)}
    # 50 cities
    # s={'0': (8.952319373406947, 5.800337761014988), '1': (3.5241918689174767, 3.5786528819962857), '2': (0.5250243320510839, 6.883099506526182), '3': (9.926570263061741, 0.7410044591467502), '4': (1.691038914592966, 1.357631567864882), '5': (5.139173859303917, 9.742485346814586), '6': (2.5004907977481414, 4.320036538074933), '7': (6.01286941258774, 5.762377207938558), '8': (8.152658682990797, 9.548808567053062), '9': (1.0283120816871472, 1.3065593267267106), '10': (1.43331978410983, 4.938316732270484), '11': (6.73511663520058, 4.6704284035344), '12': (0.6388093621977675, 7.363605803153601), '13': (9.327178395301024, 2.105613869126147), '14': (2.0809180153065276, 3.612737756235483), '15': (7.35909527704322, 6.438087216277987), '16': (0.5584208499157484, 1.2946379833563737), '17': (7.845147886606441, 5.953476575577399), '18': (7.255684539484673, 6.429337462239819), '19': (7.381648327426159, 8.890201799239742), '20': (8.438700576955354, 0.1062950913704297), '21': (1.669297571243853, 6.737153285447523), '22': (8.793430379245212, 0.7441302630292745), '23': (9.595443444894386, 5.73020608634727), '24': (8.378680223145833, 4.192719245980063), '25': (7.84785727341424, 9.50440418786783), '26': (3.238574650326269, 3.6834972730490847), '27': (0.8180848969200738, 0.4630685853987604), '28': (5.343615729022707, 8.15053623572741), '29': (5.167307073085836, 3.935934465650285)}
    fig, axs = plt.subplots(2)

    tour = sampleTour(history, s, t)    # permutation of cities

    for i in range(len(tour)-1):
        firstCity = s[str(tour[i])]
        secondCity = s[str(tour[i+1])]
        x_values = [firstCity[0], secondCity[0]]
        y_values = [firstCity[1], secondCity[1]]
        axs[0].plot(x_values,y_values)

    loss=t.loss(tour, s)
    plt.figtext(0.8,0.9, "Loss = %f" % loss)
    plt.title("TSP Solver for %d cities" % numCities)

    t.setTheta(np.zeros(len(t.getTheta())))

    history = ["0"]
    tour = sampleTour(history, s, t)    # permutation of cities

    for i in range(len(tour)-1):
        firstCity = s[str(tour[i])]
        secondCity = s[str(tour[i+1])]
        x_values = [firstCity[0], secondCity[0]]
        y_values = [firstCity[1], secondCity[1]]
        axs[1].plot(x_values,y_values)

    loss=t.loss(tour, s)
    plt.figtext(.8, .025, "Loss = %f" % loss)

    print("loss: %f" % t.loss(tour, s))

    plt.savefig("TSP_%d_Cities_%d_N.png" % (numCities, N))

def solveTSP(N, numCities, stepsize, dimensions):
    """
    stepsize 100, 0.1, 0.01 all returned the same permutation for s1
    """
    # s = {'0': (32.361538, -86.279118), '1': (33.448457, -112.073844), '2': (34.736009, -92.331122), '3': (38.555605, -121.468926), '4': (39.7391667, -104.984167), '5': (41.767, -72.677), '6': (39.161921, -75.526755), '7': (30.4518, -84.27277), '8': (33.76, -84.39), '9': (43.613739, -116.237651), '10': (39.78325, -89.650373), '11': (39.790942, -86.147685), '12': (41.590939, -93.620866), '13': (39.04, -95.69), '14': (38.197274, -84.86311), '15': (30.45809, -91.140229), '16': (44.323535, -69.765261), '17': (38.972945, -76.501157), '18': (42.2352, -71.0275), '19': (42.7335, -84.5467), '20': (44.95, -93.094), '21': (32.32, -90.207), '22': (38.572954, -92.189283), '23': (46.595805, -112.027031), '24': (40.809868, -96.675345), '25': (39.160949, -119.753877), '26': (43.220093, -71.549127), '27': (40.221741, -74.756138), '28': (35.667231, -105.964575), '29': (42.659829, -73.781339), '30': (35.771, -78.638), '31': (46.813343, -100.779004), '32': (39.962245, -83.000647), '33': (35.482309, -97.534994), '34': (44.931109, -123.029159), '35': (40.269789, -76.875613), '36': (41.82355, -71.422132), '37': (34.0, -81.035), '38': (44.367966, -100.336378), '39': (36.165, -86.784), '40': (30.266667, -97.75), '41': (40.7547, -111.892622), '42': (44.26639, -72.57194), '43': (37.54, -77.46), '44': (47.042418, -122.893077), '45': (38.349497, -81.633294), '46': (43.074722, -89.384444), '47': (41.145548, -104.802042)}
    s = {'0': (0.6355930464762073, 0.7513829151619608), '1': (0.6766084442819037, 0.9424999026618629), '2': (0.8695590302038431, 0.9827826244929699), '3': (0.3601568804589751, 0.7709754974751909), '4': (0.650193279164532, 0.6475187354382209), '5': (0.12241619792827585, 0.5620798056525347), '6': (0.7806028364565706, 0.8655786899043625), '7': (0.8291980297099166, 0.7879833188383185), '8': (0.06712526204136249, 0.24027896382433955), '9': (0.7977312754979686, 0.7170497064621041), '10': (
        0.6602844691003179, 0.696157566271456), '11': (0.9449193055275107, 0.1257938244624739), '12': (0.39628297436338134, 0.7752534844902441), '13': (0.385662514502488, 0.03763755109173783), '14': (0.20741955028025694, 0.03273983423410676), '15': (0.6834181377927429, 0.6072068811287631), '16': (0.17194798021732594, 0.5776072458099242), '17': (0.20055408651481488, 0.2154823376431938), '18': (0.9889395808934918, 0.29905751702460626), '19': (0.5420843927948603, 0.03519183292782657)}
    t = TSP(numCities, stepsize, dimensions, N)
    gradient = np.zeros(numCities)
    tours = []
    
    initS = t.generateS(numCities)
    t.initTheta(t.phi(["0"], initS, "1"))
    # print(initS)
    start=time.time()
    losses=updateGradient(t, gradient, tours, N, numCities)
    end=time.time()
    print("updating gradient time: %f" % (end-start))
    avgLoss = sum(losses)/N
    print("avg loss: %f" % avgLoss)
    plotLoss(losses, numCities, dimensions)

    totalLoss = 0
    sampleNum=100
    for i in range(sampleNum):
        history=['0']
        s=t.generateS(numCities)
        tour=sampleTour(history, s, t)
        loss=t.loss(tour, s)
        totalLoss+=loss
    print("avg loss: %f" % (totalLoss/sampleNum))
    # plotRoute(t, numCities,N, initS)

    return t

def main():
    N = 3
    numCities = 50
    stepsize = 2
    # dimensions = (30.266667,-123.029159)
    dimensions = (1,1)
    
    solveTSP(N, numCities, stepsize, dimensions)

main()
