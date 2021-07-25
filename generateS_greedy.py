import pandas as pd
import numpy as np
import math
import time

import urllib.request

url = "https://raw.githubusercontent.com/jinchenghao/TSP/master/data/TSP100cities.tsp"

with urllib.request.urlopen(url) as response:
    data=response.read()

strData=str(data)[2:]
print("data: %s" %strData)
print(type(strData))

lines=[]
currStr=""
numbers = ["0","1","2","3","4","5","6","7","8","9"]
for char in strData:
    currStr+=char
    if char == "n":
        lines.append(currStr)
        currStr=""

cleanedLines=[]
for line in lines:
    # line.replace("\","")
    print(line)
    line=line[:len(line)-2]
    cleanedLines.append(line)

print(cleanedLines)
s={}
largestX=-1
largestY=-1
for line in cleanedLines:
    listLines=line.split(" ")
    print(listLines)
    s[str(int(listLines[0])-1)]=(int(listLines[1]),int(listLines[2]))
    if int(listLines[1]) > largestX:
        largestX=int(listLines[1])
    if int(listLines[2]) > largestY:
        largestY=int(listLines[2])


print(largestX)
print(largestY)
print(s)


#print(lines)
# for char in str(data):
#     print("char: ", char)

# dataframe = pd.read_table(data,sep=" ",header=None)

# print(dataframe)
# v = dataframe.iloc[:,1:3]

# print(v)

# for index, row in dataframe.iterrows(): 
#     print("x: %s, y: %s" % (row['1'],row['2']))
