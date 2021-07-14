phi = -1
phis = []
for i in range(self.numCities):
    if i in history:
        phis.append(0)
    else:
        currentCity = self.cities[str(self.currCity)]
        nextCity = self.cities[str(i)]
        if len(history) == 0:
            prevCity = currentCity
        else:
            prevCity = self.cities[str(history[-1])]
        dist1 = self.distance(currentCity, prevCity)
        dist2 = self.distance(currentCity, nextCity)
        phis.append([dist1+dist2])
return phis

"""
phi = []
availableCities = self.getAvailableCities(history)
for i in range(len(availableCities)):
    currentCity = self.cities[str(self.currCity)]
    nextCity = self.cities[str(availableCities[i])]
    if len(history)==0:
        prevCity = currentCity
    else: 
        prevCity = self.cities[str(history[-1])]
    dist1 = self.distance(currentCity, prevCity)
    dist2 = self.distance(currentCity, nextCity)
    if len(history)==0:
        phi.append([dist1+dist2, self.currCity,
                    availableCities[i], 0])
    else: 
        phi.append([dist1+dist2, self.currCity, availableCities[i], history[-1]])
return phi
"""