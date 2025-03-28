#Version: v0.1
#Date Last Updated: 3/24/25


#%% MODULE BEGINS
module_name = 'preprocessing'

'''
Version: v0.1

Description:
    Preprocessing

Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  3/24/25
Date Last Updated:  3/24/25

Doc:
    PA1

Notes:
    Preprocessing
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
from   copy       import deepcopy as dpcpy
from   matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here
class Preprocess:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_file(self):
        df = pd.read_csv(self.file_path)
        return df
    def celcius_to_farenheit(self, df):
        df['MAX TEMP'] = (df['MAX TEMP'] - 32) * 5.0/9.0
        df['MIN TEMP'] = (df['MIN TEMP'] - 32) * 5.0/9.0
        df['TEMPERATURE'] = (df['TEMPERATURE'] - 32) * 5.0/9.0
        df.to_excel('OUTPUT/Preprocessed Rainfall.xlsx', index=False)
        return df
    def plot(df):
        pass

#Function definitions Start Here
def main():
    pass
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()
