#Version: v0.1


#%% MODULE BEGINS
module_name = 'preprocessing'

'''
Version: v0.1

Description:
    Preprocessing and Feature Extraction

Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  3/24/25
Date Last Updated:  4/28/25

Doc:
    Project

Notes:
    Preprocessing and Feature Extraction
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
from copy import deepcopy as dpcpy
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import tensorflow as tf
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
        df['MAX TEMP'] = (df['MAX TEMP'] * 9.0/5.0) + 32
        df['MIN TEMP'] = (df['MIN TEMP'] * 9.0/5.0) + 32
        df['TEMPERATURE'] = (df['TEMPERATURE'] * 9.0/5.0) + 32
        df.to_excel('OUTPUT/Preprocessed Rainfall.xlsx', index=False)
        return df
    
    def targetReclass(self, df):
        df.replace({'yes': 1, 'no': 0}, inplace=True)
        df.to_excel('OUTPUT/Preprocessed Rainfall.xlsx', index=False)
        return df
    
    def plot(self, df, plot1, plot2, plot3, measurement):
        sns.set()
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
        sns.lineplot(data=df, x=df.index, y=plot1, ax=axs[0]).set(title=f'{plot1} ({measurement})')
        sns.lineplot(data=df, x=df.index, y=plot2, ax=axs[1]).set(title=f'{plot2} ({measurement})')
        sns.lineplot(data=df, x=df.index, y=plot3, ax=axs[2]).set(title=f'{plot3} ({measurement})')
        plt.savefig(f"OUTPUT/{measurement} plots.png")
    
    def distribution(title, df, *plots):
        sns.set()
        fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(15, 11))
        axs = axs.flatten()
        for i, plot in enumerate(plots):
            sns.histplot(data=df, x=plot, ax=axs[i], kde=True).set(title=f'Frequency of {plot}')
        plt.savefig(f"OUTPUT/{title} Histograms.png")
        
    
    def histogram(df, column_name, bins=10, title=None, xlabel=None, ylabel=None):
        plt.figure(figsize=(10,6))
        plt.hist(df[column_name], bins=bins)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(column_name)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"OUTPUT/{title}.png")
    
    def heatmap(self, df):
        plt.figure(figsize=(10,10))
        sns.heatmap(df.corr() > 0.8, annot = True, cbar = False)
        plt.savefig(f"OUTPUT/heat map.png")
    
    
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

# %%
