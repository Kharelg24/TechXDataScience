# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import setuptools

# Keras is a high-level neural network Python API
# Runs on top of lower-level libraries like TensorFlow
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# %%
class MLClass:

    def __init__(self, data=None, predictorVar=None, targetCol=None,
                xTrain=None, xTest=None, yTest=None, yTrain=None):
        self.data = data
        self.predictorVar = predictorVar
        self.targetCol = targetCol
        self.xTrain = xTrain
        self.xTest= xTest
        self.yTest= yTest
        self.yTrain = yTrain

    
    def exploringDataset(self):      
        try:
            print("The shape of the data is: ", self.data.shape)
            print("The dataset has {} columns".format(self.data.columns))
            print("The name of the columns are: {}\n".foramt([x for x in self.data.columns]))

        except Exception as es:
            print("Data type is None, load the dataset first")


    def setPredictorVar(self, predictorCols):
        self.predictorVar = predictorCols

    def setTargetCol(self, targetColumn):
        self.targetCol = targetColumn

    def getTrain(self):
        try:
            return {"X-Train": self.xTrain,
                     "Y-Train": self.yTrain}
        except Exception as es:
            print("The Training dataset has not been created yet")

    def getTest(self):
        try:
            return{"X-Test": self.xTest,
                   "Y-Train": self.yTest}
        except Exception as es:
            print("The Testing dataset has not been created yet")
            
    def loadingDataset(self, filePath):
        self.data = pd.read_csv(filePath)

    def trainTestDataset(self, testSize=0.10, randomState=24):
        self.xTrain, self.xTest, self.yTrain, self.yTrain = \
        train_test_split(self.predictorVar, self.targetCol, test_size=testSize, random_state=randomState)



