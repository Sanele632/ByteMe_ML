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
from copy import deepcopy as dpcpy
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
    
    def lineplot(self, df, *plots):
        sns.set()
        fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(18, 10))
        axs = axs.flatten()
        for i, plot in enumerate(plots):
            if plot in df.columns:  # Ensure column exists
                sns.histplot(data=df, x=plot, ax=axs[i], kde=True).set(title=f'Frequency of {plot}')
        else:
            print(f"Warning: Column '{plot}' not found in DataFrame.")
        
    
    def histogram(self, df, column_name, bins=10, title=None, xlabel=None, ylabel=None):
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
        plt.show()
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
