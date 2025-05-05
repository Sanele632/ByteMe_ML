#Version: v0.1
#Date Last Updated: 4/19/25

#%% STANDARDS   -DO NOT include this block in a new module
'''
Unless otherwise required, use the following guidelines
* Style:
    - Sort all alphabatically
    - Write the code in aesthetically-pleasing style
    - Names should be self-explanatory
    - Add brief comments
    - Use relative path
    - Use generic coding instead of manually-entered constant values

* Performance and Safety:
    - Avoid if-block in a loop-block
    - Avoid declarations in a loop-block
    - Initialize an array if size is known

    - Use immutable types
    - Use deep-copy
    - Use [None for i in Sequence] instead of [None]*len(Sequence)

'''

#%% MODULE BEGINS
module_name = 'DT_ANN'

'''
Version: v0.1

Description:
    DT and ANN algorithms

Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  4/19/25
Date Last Updated:  4/19/25

Doc:
    Project

Notes:
    <***>
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
from copy import deepcopy as dpcpy
from sklearn.model_selection import train_test_split
import numpy  as np 
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here
class ML:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_file(self):
        df = pd.read_csv(self.file_path)
        return df
    def split_data(pInput, pRatio, random):
        X = pInput(['RAINFALL'], axis = 'columns')
        y = pInput['RAINFALL']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y,  test_size= pRatio[1] + pRatio[2], random_state = random)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= 0.5, random_state = random)

        return X_train, y_train, X_val, X_test, y_val, y_test
    def DT(X_train, y_train, X_val, X_test, y_val, y_test):
        DT = DecisionTreeClassifier()
        DT = DT.fit(X_train, y_train)
        y_pred = DT.predict(X_val)
        y_pred1 = DT.predict(X_test)
    def SVM(X_train, y_train, X_val, X_test, y_val, y_test):
        SVM = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.
        SVM.fit(X_train, y_train)
        y_pred = SVM.predict(X_val)
        y_pred1 = SVM.predict(X_test)
        
    



#Function definitions Start Here
def main():
    data = ML('OUTPUT/data.csv')
    df = data.load_file()
    X_train, y_train, X_val, X_test, y_val, y_test = data.split(df, (0.6, 0.2, 0.2), 11)
    data.DT(X_train, y_train, X_val, X_test, y_val, y_test)
    
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()