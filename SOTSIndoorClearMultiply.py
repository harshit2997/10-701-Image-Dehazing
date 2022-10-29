import os
import shutil

hazyNames = os.listdir(".\\ResideSOTS\\indoor\\hazy")
clearNames = os.listdir(".\\ResideSOTS\\indoor\\gt")

hazyNames = sorted(hazyNames, key = lambda s: int(s.split('_')[0]))
clearNames = sorted(clearNames, key = lambda s: int(s.split('.')[0]))

i=0
j=0

while(i<len(clearNames) and j<len(hazyNames)):
    clearName = clearNames[i]
    rootClearName = clearName.split('.')[0]
    hazyName = hazyNames[j]
    rootHazyName = hazyName.split('_')[0]

    if not (rootClearName == rootHazyName):
        os.remove(".\\ResideSOTS\\indoor\\gt\\"+clearName)
        i+=1
        continue
    else:
        shutil.copy(".\\ResideSOTS\\indoor\\gt\\"+clearName, ".\\ResideSOTS\\indoor\\gt\\"+hazyName)
        j+=1

os.remove(".\\ResideSOTS\\indoor\\gt\\"+clearNames[i])
