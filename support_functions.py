
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import utils.tools as utils
import matplotlib.pyplot as plt 
from keras.models import load_model
from sklearn.metrics import roc_curve, auc 

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

from sklearn.preprocessing import scale 
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold


# Calculate performance for binary class classification
def calculate_model_performance(ytest, yhat):

    target_names = ['class 0', 'class 1']
    print(classification_report(ytest, yhat, target_names=target_names))
    # Generate confusion matrix
    cm = confusion_matrix(ytest, yhat)
    print("\nConfusion Matrix")
    display(pd.DataFrame(cm))

    TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    Sen = float(TP)/ (TP + FN + 1e-06)
    Spe = float(TN)/(TN + FP + 1e-06)
    # accuracy: (tp + tn) / (p + n)
    acc = accuracy_score(ytest, yhat)    
    # precision tp / (tp + fp)
    pre = precision_score(ytest, yhat)
    # recall: tp / (tp + fn)
    rec = recall_score(ytest, yhat)    
    # f1: 2 tp / (2 tp + fp + fn)
    F1 = f1_score(ytest, yhat)
    # ROC AUC
    auc = roc_auc_score(ytest, yhat)

    return acc, pre,rec,F1,auc, Sen, Spe



def get_data(data_path):
    
    # Read MATLAB array
    data_train = sio.loadmat(data_path)
    # extract array name from data_path
    array_name = data_path.split(".")
    # Extract train data fro MATLAB dictionary 
    data = data_train.get(array_name[0])
    # Get the dimention of data data     
    no_samples, no_features = np.shape(data)
    print("No. of samples ", no_samples)
    print("No. of features",no_features)

    # Training data hase no labels. Data is balanced, 50% positive class and 50% negative class
    # create build and populate a label's vector with size of number of samples in teh training data
    # Half of it filled with 1's and teh second half is filled with zeros's
    
    label1 = np.ones((544 , 1))
    label2 = np.zeros((544, 1))
    label = np.append(label1, label2)
    
    # Convert label vector y into a one dimentional dataframe vector
    y=pd.DataFrame(label, columns=["label"])

    # Intialize a dataset dataframe with generic F1, F2, ... F71 column headers. 
    # because Traininng data has no column headers 
    df_dataset = pd.DataFrame(data, columns=["F"+str(i) for i in range(1, 72)])

    # Merge label one vector dataframe with Training dataset dataframe.
    df = pd.concat([df_dataset, y], axis=1)

    return df



class DataTransforms():
    """ Perform data Ttransformation like scale between 0-1, power transformation bell like shape 
    distribution and Variance Threshold remove low contributing features """    

    def scale(self, df):
        X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
        df_scale = scale(X)
        y = pd.DataFrame(y, columns=["label"])
        df_scale_dataset = pd.DataFrame(df_scale, columns=["F"+str(i) for i in range(1, df_scale.shape[1]+1)])
        return pd.concat([df_scale_dataset, y], axis=1)
        
    def powertransform(self, df):
        """ Make skewed data distribution looks like bell shape"""
        X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
        pt = PowerTransformer(standardize=True)
        df_pt = pt.fit_transform(X)
        y = pd.DataFrame(y, columns=["label"])
        df_pt_dataset = pd.DataFrame(df_pt, columns=["F"+str(i) for i in range(1, df_pt.shape[1]+1)])
        return pd.concat([df_pt_dataset, y], axis=1)

    def variance_threshold(self, df):
        """ remove low variance features"""
        X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
        selector = VarianceThreshold(threshold=0.0)
        df_var = selector.fit_transform(X)
        y = pd.DataFrame(y, columns=["label"])
        df_var_dataset = pd.DataFrame(df_var, columns=["F"+str(i) for i in range(1, df_var.shape[1]+1)])
        return pd.concat([df_var_dataset, y], axis=1)



def indepTesting(Xtest, Ytest, saved_model_path):
     Sepscores = []
     ytest = np.ones((1, 2)) * 0.5
     yscore = np.ones((1, 2)) * 0.5
     print("this is the shape of Xtest", Xtest.shape)
     print("this is the shape of Ytest", Ytest.shape)
     xtest = np.vstack(Xtest)
     print("this is the shape of xtest", xtest.shape)
     y_test = np.vstack(Ytest)
   
     ldmodel = tf.keras.saving.load_model(saved_model_path)
     print("Loaded model from disk")
     y_score = ldmodel.predict(Xtest)
     yscore = np.vstack((yscore, y_score))
     y_class = utils.categorical_probas_to_classes(y_score)

     fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
     roc_auc = auc(fpr, tpr)
     acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                         y_test)
     Sepscores.append([acc, precision,npv,sensitivity, specificity, mcc,f1,roc_auc])
     print('acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
           % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))
     fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 1])
     auc_score = auc(fpr, tpr)
     scores = np.array(Sepscores)
     result1 = np.mean(scores, axis=0)
     H1 = result1.tolist()
     Sepscores.append(H1)
     result = Sepscores
     row = y_score.shape[0]
     yscore = y_score[np.array(range(1, row)), :]
     yscore_sum = pd.DataFrame(data=yscore)
     yscore_sum.to_csv('yscore_sum_CPSR_ind.csv')
     y_test = y_test[np.array(range(1, row)), :]
     ytest_sum = pd.DataFrame(data=y_test)
     ytest_sum.to_csv('ytest_sum_CPSR_ind.csv')
     colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp', 'MCC', 'F1', 'AUC']
     data_csv = pd.DataFrame(columns=colum, data=result)  # , index=ro)
     data_csv.to_csv('Result_CPSR_ind.csv')
     lw = 2
     plt.plot(fpr, tpr, color='darkorange',
              lw=lw, label='SVC ROC (area = %0.2f%%)' % auc_score)
     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
     plt.xlim([0.0, 1.05])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('Receiver operating characteristic')
     plt.legend(loc="lower right")
     plt.grid()
     plt.show()


