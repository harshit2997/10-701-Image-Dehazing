from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image

def processIfRequired(im, H, W):
    curW = im.size[0]
    curH = im.size[1]

    if (curW==W and curH==H):
        return im

    # rotation required in case of a rectangle
    if ((curW-curH)*(W-H)<0):
        im = im.rotate(90, expand=True)
        curW = im.size[0]
        curH = im.size[1]

    #no resizing for now. just rotation
    return im

def LoaderNormalizer(data):
    W = 620
    H = 460

    fileNames = os.listdir(data.inputDir)

    data.totalLength = len(fileNames)
    data.inputs  = np.empty((len(fileNames), 3, H, W))
    data.targets = np.empty((len(fileNames), 3, H, W))

    for i, fileName in enumerate(fileNames):
        inputIm = Image.open(data.inputDir+fileName)
        targetIm = Image.open(data.targetDir+fileName)

        inputIm = processIfRequired(inputIm, H, W)
        targetIm = processIfRequired(targetIm, H, W)     
        
        inputAr = (np.asarray(inputIm).transpose([2,0,1]))/255.0
        targetAr = (np.asarray(targetIm).transpose([2,0,1]))/255.0

        data.inputs[i] = inputAr
        data.targets[i] = targetAr

    return data

class DehazeDataset(Dataset):
    def __init__(self, inputDir, targetDir, shuffle=0, valProp = 0.0):
        self.inputDir = inputDir
        self.targetDir = targetDir

        if not (self.inputDir[-1:]=='\\'):
            self.inputDir = self.inputDir + '\\'

        if not (self.targetDir[-1:]=='\\'):
            self.targetDir = self.targetDir + '\\'            
        
        self = LoaderNormalizer(self, shuffle=shuffle)

        valLength = int(self.totalLength*valProp)

        self.valInputs = self.inputs[(self.totalLength-valLength):]
        self.valTargets = self.targets[(self.totalLength-valLength):]
        self.valLength = valLength

        self.inputs = self.inputs[:(self.totalLength-valLength)]
        self.targets = self.targets[:(self.totalLength-valLength)]
        self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class DehazeDatasetVal(DehazeDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valInputs
        self.targets = dataset.valTargets
        self.totalLength = dataset.valLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]