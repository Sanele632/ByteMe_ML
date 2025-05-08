#Version: v0.1


#%% MODULE BEGINS
module_name = 'preprocessing'

'''
Version: v0.1

Description:
    Preprocessing

Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  3/24/25
Date Last Updated:  3/31/25

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
    
    def distribution(self, df, *plots):
        sns.set()
        fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(15, 15))
        axs = axs.flatten()
        for i, plot in enumerate(plots):
            if plot in df.columns:  # Ensure column exists
                sns.histplot(data=df, x=plot, ax=axs[i], kde=True).set(title=f'Frequency of {plot}')
        else:
            print(f"Warning: Column '{plot}' not found in DataFrame.")
        plt.savefig(f"OUTPUT/Distribution Histograms.png")
        
    
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
        plt.savefig(f"OUTPUT/{title}.png")
    
    def heatmap(self, df):
        plt.figure(figsize=(10,10))
        sns.heatmap(df.corr() > 0.8, annot = True, cbar = False)
        plt.savefig(f"OUTPUT/heat map.png")
    '''
    def split(self, df, feature_col, target, test_size = 0.2, random_state = 42):
        df = df.replace([float('inf'), float('-inf')], pd.NA)
        df = df.dropna()
        X = df[feature_col]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def kNN(self, x, y, xTest, yTest, n):
        score_list = []
        k=1

        for k in range(1, len(xTest) + 1):
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(x, y)
            y_pred = knn.predict(xTest)
            cm = confusion_matrix(yTest, y_pred)
            score = accuracy_score(yTest, y_pred)
            score_list.append(score)
        
        max_accuracy = max(score_list)
        best_k = score_list.index(max_accuracy) + 1
        knn = KNeighborsClassifier(n_neighbors= best_k)
        knn.fit(x, y)
        y_pred1 = knn.predict(xTest)
        test_cm = confusion_matrix(yTest, y_pred1)

        TP = test_cm[0, 0]  
        FP = test_cm[0, 1]  
        TN = test_cm[1, 1]  
        FN = test_cm[1, 0]

        precision = TP / (TP + FP) 
        sensitivity = TP / (TP + FN) 
        specificity = TN / (TN + FP)
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

        performance_measures = [precision, sensitivity, specificity, F1]
        return  performance_measures, best_k, score_list
        

    def ANN(self, xTr, xTst, yTr, yTst, hl, ep, bs):
        scaler = StandardScaler()
        xTr_scaled = scaler.fit_transform(xTr)
        xTst_scaled = scaler.fit_transform(xTst)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(xTr_scaled.shape[1],)))
        for units in hl:
            model.add(tf.keras.layers.Dense(units, activation = 'relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(xTr_scaled, yTr, epochs=ep, batch_size=bs, validation_split=0.1)

        loss, accuracy = model.evaluate(xTst_scaled, yTst)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        return model, history
'''
    
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
