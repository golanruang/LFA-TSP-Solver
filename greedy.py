
#Grudge
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

import urllib.request
# url = 'https://raw.githubusercontent.com/jinchenghao/TSP/master/data/TSP100cities.tsp'
# url = 'https://raw.githubusercontent.com/golanruang/TSP/main/TSP50cities.tsp.txt'
url = "https://raw.githubusercontent.com/golanruang/TSP/main/TSP10cities.tsp.txt"
# url = "https://raw.githubusercontent.com/golanruang/TSP/main/TSP25cities.tsp.txt"
# url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/traveling_salesman/capitals.json"
data = urllib.request.urlopen(url)
dataframe = pd.read_table(data,sep=" ",header=None)
v = dataframe.iloc[:,1:3]
 
train_v= np.array(v)
train_d=train_v
dist = np.zeros((train_v.shape[0],train_d.shape[0]))
print("hi")
 
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
print(s)

for m in range(n):
    print("%s-> "%(s[m]),end='')
print()
print("The running time of the program is: %s"%(end-start))

tour=s
tour.append(0)
print(type(tour[0]))

# plotting
s = {'0': (2915, 790), '1': (1194, 2834), '2': (536, 690), '3': (2776, 2572), '4': (390, 1265), '5': (96, 1323), '6': (360, 665), '7': (2923, 3069), '8': (614, 73), '9': (143, 1802), '10': (836, 653), '11': (670, 1011), '12': (1339, 1838), '13': (671, 69), '14': (2321, 2186), '15': (836, 22), '16': (2390, 1100), '17': (2527, 904), '18': (634, 661), '19': (2466, 2965), '20': (2109, 2375), '21': (3031, 523), '22': (1165, 194), '23': (2935, 1120), '24': (3088, 2592), '25': (2001, 1305), '26': (2224, 2586), '27': (487, 2892), '28': (1563, 93), '29': (307, 2795), '30': (56, 2114), '31': (106, 587), '32': (2318, 884), '33': (1454, 3000), '34': (2663, 2729), '35': (134, 2657), '36': (148, 713), '37': (2964, 2694), '38': (1591, 1247), '39': (523, 1609), '40': (274, 577), '41': (2925, 2689), '42': (2129, 1884), '43': (3079, 1802), '44': (1945, 1642), '45': (2612, 797), '46': (1161, 2899), '47': (2831, 2615), '48': (2226, 2318), '49': (722, 1634), '50': (
    1765, 1687), '51': (596, 1478), '52': (2317, 1423), '53': (1543, 2698), '54': (456, 1766), '55': (2975, 2923), '56': (1310, 1827), '57': (182, 2422), '58': (953, 369), '59': (919, 2865), '60': (1820, 2299), '61': (2277, 1029), '62': (964, 1609), '63': (2060, 1833), '64': (1558, 1540), '65': (1150, 2056), '66': (790, 1920), '67': (1796, 513), '68': (2430, 1097), '69': (2506, 2870), '70': (930, 2709), '71': (883, 2242), '72': (2985, 313), '73': (1682, 131), '74': (1935, 1221), '75': (554, 393), '76': (759, 2344), '77': (483, 2872), '78': (896, 1839), '79': (2844, 1804), '80': (2156, 445), '81': (192, 748), '82': (835, 266), '83': (2040, 2231), '84': (2005, 3032), '85': (739, 821), '86': (1597, 2860), '87': (2311, 1427), '88': (1539, 3095), '89': (545, 1255), '90': (712, 221), '91': (2152, 2917), '92': (2546, 704), '93': (1864, 42), '94': (1085, 1530), '95': (1636, 198), '96': (2255, 2637), '97': (2131, 109), '98': (1112, 1185), '99': (3094, 1558)}
for i in range(len(tour)-1):
    firstCity = s[str(tour[i])]
    secondCity = s[str(tour[i+1])]
    x_values = [firstCity[0], secondCity[0]]
    y_values = [firstCity[1], secondCity[1]]
    plt.plot(x_values, y_values)

plt.title("Greedy Solver for %d cities" % 50)
plt.show()
