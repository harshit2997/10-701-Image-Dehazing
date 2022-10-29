import os
import shutil

hazyNames = os.listdir(".\\ResideITS\\hazy")
clearNames = os.listdir(".\\ResideITS\\clear")

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
        os.remove(".\\ResideITS\\clear\\"+clearName)
        i+=1
        continue
    else:
        shutil.copy(".\\ResideITS\\clear\\"+clearName, ".\\ResideITS\\clear\\"+hazyName)
        j+=1

os.remove(".\\ResideITS\\clear\\"+clearNames[i])
