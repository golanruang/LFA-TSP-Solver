import json
def readJson():
    cityNum = 0
    s = {}
    f = open('capitals.json',)

    data = json.load(f)

    for i in data:
        s[str(cityNum)] = (data[i]['lat'], data[i]['long'])
        cityNum += 1

    f.close()
    print(s)
readJson()