import time
import math
import numpy as np

def main():
    p=[(0.34123,0.534657),(0.5421,0.5423), (0.6542324,0.78954),(0.764534,0.0231452),(0.45134,0.123478)]
    timee=0
    for i in range(100000):
        start=time.time()
        for point in p:
            math.sqrt((point[0] - point[0])**2 + (point[1] - point[1])**2)
        end=time.time()
        print("time taken: %f" % (end-start))
        timee+=(end-start)
    print("avg time: %f" % (timee/1000))
main()
