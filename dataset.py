from typing_extensions import Self
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
import torchvision.transforms as T

W = 512
H = 512

def processIfRequired(im):
    curW = im.size[0]
    curH = im.size[1]

    if (curW==W and curH==H):
        return im

    # rotation required in case of a rectangle
    if ((curW-curH)*(W-H)<0):
        im = im.rotate(90, expand=True)
        curW = im.size[0]
        curH = im.size[1]

    return T.Resize((H,W))(im)

class DehazeDataset(Dataset):
    def __init__(self, inputDir, targetDir, shuffle=0, valProp = 0.0):
        self.inputDir = inputDir
        self.targetDir = targetDir

        if not (self.inputDir[-1:]=='/'):
            self.inputDir = self.inputDir + '/'

        if not (self.targetDir[-1:]=='/'):
            self.targetDir = self.targetDir + '/'            
        
        self.fileNames = os.listdir(self.inputDir)

        for i in range(shuffle):
            random.shuffle(self.fileNames)
        
        self.totalLength = len(self.fileNames)
        valLength = int(self.totalLength*valProp)

        self.valFileNames = self.fileNames[(self.totalLength-valLength):]
        self.valLength = valLength

        self.totalLength = self.totalLength-valLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]

        inputIm = Image.open(self.inputDir+fileName)
        targetIm = Image.open(self.targetDir+fileName)

        inputIm = processIfRequired(inputIm)
        targetIm = processIfRequired(targetIm)     
        
        inputAr = ((np.asarray(inputIm).transpose([2,0,1]))/255.0).astype(np.float32)
        targetAr = ((np.asarray(targetIm).transpose([2,0,1]))/255.0).astype(np.float32)

        return inputAr, targetAr

class DehazeDatasetVal(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.fileNames = dataset.valFileNames
        self.totalLength = dataset.valLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]

        inputIm = Image.open(self.dataset.inputDir+fileName)
        targetIm = Image.open(self.dataset.targetDir+fileName)

        inputIm = processIfRequired(inputIm)
        targetIm = processIfRequired(targetIm)     
        
        inputAr = ((np.asarray(inputIm).transpose([2,0,1]))/255.0).astype(np.float32)
        targetAr = ((np.asarray(targetIm).transpose([2,0,1]))/255.0).astype(np.float32)

        return inputAr, targetAr