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
data.lineplot(df, "PRESSURE ", "DEWPOINT", "HUMIDITY", "CLOUD ", "SUNSHINE", "WIND DIRECTION", "WIND SPEED")
data.plot(df, 'MAX TEMP', 'TEMPERATURE', 'MIN TEMP', 'Celcius')
data.histogram(df, 'RAINFALL', bins=10, title="Rainfall")
print(df)

preprocessed_df = data.celcius_to_farenheit(df)
print("\nAfter temperature conversion: ")
print(preprocessed_df)

preprocessed_df = data.targetReclass(preprocessed_df)
print("\nAfter target reclassification: ")
print(preprocessed_df)

data.plot(preprocessed_df, 'MAX TEMP', 'TEMPERATURE', 'MIN TEMP', 'Farenheit')
data.histogram(preprocessed_df, 'RAINFALL', bins=10, title="Rainfall")


#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()

# %%
