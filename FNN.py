# Keras is a high-level neural network Python API
# Runs on top of lower-level libraries like TensorFlow
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

from ParentClass import MLClass


class CNNClass(MLClass):

    def __init__(self, model=None, data=None, predictorVar=None, targetCol=None, 
                 xTrain=None, xTest=None, yTest=None, yTrain=None):
        
        self.model = model

        super().__init__(data, predictorVar, targetCol,
                xTrain, xTest, yTest, yTrain)
        
        
    def createCnnModel(self, featuresNum=None, outputDim=None):
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_dim=featuresNum)) 
        self.model.add(Dense(64, activation='relu')) 
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(outputDim, activation='softmax'))

        self.__compileModel()
        #self.__getModelSummary()


    def __compileModel(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def __getModelSummary(self):
        self.model.summary()

    
    def trainModel(self, numEpochs=1):
        train = self.getTrain()
        # verbose prevents epochs from printing out
        x_train = train["X-Train"]
        y_train = to_categorical((train["Y-Train"]))
        self.model.fit(x_train, y_train, epochs= numEpochs, verbose=0)

    def evaluate(self):
        self.__trainDataEvaluate()
        self.__testDataEvalutate()

    def __trainDataEvaluate(self):
        train = self.getTrain()
        x_train = train["X-Train"]
        y_train = to_categorical((train["Y-Train"]))
        scores = self.model.evaluate(x_train, y_train, verbose=0)
        #print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
        return scores

    def __testDataEvalutate(self):
        test = self.getTest()
        x_test = test["X-Test"]
        y_test = to_categorical((test["Y-Test"]))
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        #print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1], 1 - scores[1]))
        return scores

    def performanceMeasure(self):
        trainScore = self.__trainDataEvaluate()
        testScore = self.__testDataEvalutate()
        return self.__plotting(trainScore, testScore)

    def __plotting(self, trainScore, testScore):
        xLabels = ("Accuracy", "Loss")
        accuracy_data = {
            'Metric': ['Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss'],
            'Type': ['Training', 'Training', 'Testing', 'Testing'],
            'Value': [trainScore[1], 1 - trainScore[1], testScore[1], 1 - testScore[1]]
        }

        df = pd.DataFrame(accuracy_data)
        print(df)
        # Creating a bar plot using plotly express
        fig = px.bar(
            df,
            x='Metric',
            y='Value',
            color='Type',
            barmode='group',
            labels={'Value': 'Percentage'},
            title='Training and Testing (Accuracy and Loss)'
        )  

        return fig