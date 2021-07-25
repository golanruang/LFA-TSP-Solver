
#Grudge
import pandas as pd
import numpy as np
import math
import time

import urllib.request
url = 'https://raw.githubusercontent.com/jinchenghao/TSP/master/data/TSP100cities.tsp'
# url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/traveling_salesman/capitals.json"
data = urllib.request.urlopen(url)
dataframe = pd.read_table(data,sep=" ",header=None)
v = dataframe.iloc[:,1:3]
 
train_v= np.array(v)
train_d=train_v
dist = np.zeros((train_v.shape[0],train_d.shape[0]))
 
 #Calculate distance matrix
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
"""
 s: cities that have been traversed
 dist: distance matrix between cities
 sumpath: the current total length of the minimum path
 Dtemp: current minimum distance
 flag: access flag
"""
i=1
n=train_v.shape[0]
j=0
sumpath=0
s=[]
s.append(0)
start = time.time()
while True:
    k=1
    Detemp=10000000
    while True:
        l=0
        flag=0
        if k in s:
            flag = 1
        if (flag==0) and (dist[k][s[i-1]] < Detemp):
            j = k
            Detemp=dist[k][s[i - 1]]
        k+=1
        if k>=n:
            break
    s.append(j)
    i+=1
    sumpath+=Detemp
    if i>=n:
        break
sumpath+=dist[0][j]
end = time.time()
print("Result:")
print(sumpath)
for m in range(n):
    print("%s-> "%(s[m]),end='')
print()
print("The running time of the program is: %s"%(end-start))