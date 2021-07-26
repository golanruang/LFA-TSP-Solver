fname = "disneyCoords.txt"
f=open(fname,"r")
names=[]
s={}
cityNum = 0
minx=100000
maxx=-100000
miny=100000
maxy=-100000
for index, line in enumerate(f):
    split=line.split(" ")
    x, y = split[1].strip().split(',')
    x=float(x)
    y=float(y)
    if x < minx:
        minx=x
    if x > maxx:
        maxx=x
    if y < miny:
        miny=y
    if y > maxy:
        maxy=y
    s[str(index)]=(x,y)
    names.append(split[0])

print("minx: %f" % minx)
print("maxx: %f" % maxx)
print("miny: %f" % miny)
print("maxy: %f" % maxy)
print(names)
print(s)
