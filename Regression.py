import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, auc

import plotly.express as px
from ParentClass import MLClass


class Multi_nom_reg(MLClass):

  def __init__(self, data=None, predictorVar=None, targetCol=None,
                 xTrain=None, xTest=None, yTest=None, yTrain=None):

    super().__init__(data, predictorVar, targetCol,
            xTrain, xTest, yTest, yTrain)

  def RegModel(self, max_iter=800):
    train = self.getTrain()
    self.modelReg = LogisticRegression(multi_class='multinomial', solver= 'lbfgs', max_iter = max_iter, class_weight='balanced')
    self.modelReg.fit(train["X-Train"], train["Y-Train"])

  def predict(self):
    test = self.getTest()
    self.predictions = self.modelReg.predict(test["X-Test"])

  def model_scores(self):
    train = self.getTrain()
    self.model_score = self.modelReg.score(train["X-Train"], train["Y-Train"])
    print('The model score/accuracy is:', self.model_score)

  def MSE(self):
    test = self.getTest()
    predictions = self.modelReg.predict(test["X-Test"])
    self.mse = np.mean((predictions-test["Y-Test"])**2)
    print('The MSE is:', self.mse)

  def predict_proba(self):
      self.y_pred_prob = self.modelReg.predict_proba(self.xTest)
      #return self.y_pred_prob

  def ROC_curve(self):
    test = self.getTest()
    y_test_binary = label_binarize(test['Y-Test'], classes=[0,1,2,3])
    n_classes = y_test_binary.shape[1]
    class_names = ['Pass', 'Run', 'Punt', 'Field Goal']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], self.y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create a dataframe for Plotly Express
    roc_data = []
    for i in range(n_classes):
        roc_data.append(pd.DataFrame({
            'False Positive Rate': fpr[i],
            'True Positive Rate': tpr[i],
            'Class': class_names[i],
            'AUC': [roc_auc[i]] * len(fpr[i])
        }))

    # Concatenate all class ROC data
    roc_data = pd.concat(roc_data)

    # Plot ROC curve for each class using Plotly Express
    fig = px.line(roc_data, x='False Positive Rate', y='True Positive Rate',
                  color='Class', line_group='Class',
                  hover_name='Class', hover_data={'AUC': True},
                  title='ROC Curve for Multinomial Logistic Regression')

    # Add diagonal line
    fig.add_shape(type='line', line=dict(dash='dash'),
                  x0=0, x1=1, y0=0, y1=1)

    # Customize layout
    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      legend_title='Class',
                      template='plotly_white')

    return fig

  def get_confusion_matrix(self):
    test = self.getTest()

    target_names =['Pass', 'Run', 'Punt', "Field Goal"]
    n_classes = len(target_names)
    for i in range(n_classes):

        # Binarize the y_test and predictions for the current class
        y_test_binary = (test['Y-Test'] == i).astype(int)
        predictions_binary = (self.predictions == i).astype(int)

        # Compute the classification report for each target variable
        print('Class:', target_names[i])
        print(metrics.classification_report(y_test_binary, predictions_binary, labels=[0, 1]))

        # Compute confusion matrix for the binary classification
        cm = confusion_matrix(y_test_binary, predictions_binary, labels=[0, 1])
        print("Confusion Matrix:\n", cm)
        print("-" * 40)




