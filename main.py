#Version: v0.1
#Date Last Updated: 12-20-2023


#%% MODULE BEGINS
module_name = 'main'

'''
Version: v0.1

Description:
    main

Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  3/24/25
Date Last Updated:  3/24/25

Doc:
    PA1

Notes:
    main
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
import preprocessing
#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here



#Function definitions Start Here
def main():
    pass
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = preprocessing.Preprocess('INPUT/Rainfall.csv')
df = data.load_file()
print("Raw Data Distribution:")
data.distribution(df, "PRESSURE ", "DEWPOINT", "HUMIDITY", "CLOUD ", "SUNSHINE", "         WIND DIRECTION", "WIND SPEED", "MAX TEMP",
              "MIN TEMP", "TEMPERATURE" ,"RAINFALL")
data.plot(df, 'MAX TEMP', 'TEMPERATURE', 'MIN TEMP', 'Celcius')
data.histogram(df, 'RAINFALL', bins=10, title="Rainfall")
print(df)

preprocessed_df = data.celcius_to_farenheit(df)
print("\nAfter temperature conversion: ")
print(preprocessed_df)

preprocessed_df = data.targetReclass(preprocessed_df)
print("\nAfter target reclassification: ")
print(preprocessed_df)
data.histogram(preprocessed_df, 'RAINFALL', bins=10, title="Rainfall")

data.plot(preprocessed_df, 'MAX TEMP', 'TEMPERATURE', 'MIN TEMP', 'Farenheit')


#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()

# %%
