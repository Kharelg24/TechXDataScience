# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import setuptools

# Keras is a high-level neural network Python API
# Runs on top of lower-level libraries like TensorFlow
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout

# %% [markdown]
# # Run source sklearn-env/bin/activate

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
            print("The dataset has {} columns".format(len(self.data.columns)))
            #print(["The names of the columns are:".format(x for x in self.data.columns)])

        except Exception as es:
            print("Data type is None, load the dataset first")


    def cleanData(self):
        playType = ['Run','Pass',"Field Goal",'Punt']
        self.data = self.data[self.data["PlayType"].isin(playType)]
        self.data = self.data.dropna(subset=["PlayType", "down"])
        self.data["FieldGoalDistance"] = self.data["FieldGoalDistance"].fillna(0)

        predictorCols = ["GameID", "Drive", "qtr", "down", "time", "yrdline100", "ydstogo", "GoalToGo", "PosTeamScore", "DefTeamScore",\
                 "FieldGoalDistance", "ScoreDiff"]
        self.__setPredictorVar(predictorCols)
        target = "PlayType"
        self.__setTargetCol(target)


    def __setPredictorVar(self, predictorCols):
        df = self.data[predictorCols].copy()
        df["time"] = df["time"].astype(str).apply(lambda x: float(x.replace(':','')))
        scaler = StandardScaler().set_output(transform='pandas')
        feature_mat = scaler.fit_transform(df)
        self.predictorVar = feature_mat

    def __setTargetCol(self, targetColumn):
        target = self.data[targetColumn]
        target_col = target.replace({'Pass': 0, 'Run': 1, 'Punt':2, "Field Goal":3})
        self.targetCol = target_col

    def getPredictorVar(self):
        return self.predictorVar

    def getTargetCol(self):
        return self.targetCol

    def getTrain(self):
        try:
            return {"X-Train": self.xTrain,
                     "Y-Train": self.yTrain}
        except Exception as es:
            print("The Training dataset has not been created yet")

    def getTest(self):
        try:
            return{"X-Test": self.xTest,
                   "Y-Test": self.yTest}
        except Exception as es:
            print("The Testing dataset has not been created yet")
            
    def loadDataset(self, filePath):
        self.data = pd.read_csv(filePath)
        return self.data

    def trainTestDataset(self, testSize=0.10, randomState=24):
        self.xTrain, self.xTest, self.yTrain, self.yTest = \
        train_test_split(self.predictorVar, self.targetCol, test_size=testSize, random_state=randomState)


# %%

        

