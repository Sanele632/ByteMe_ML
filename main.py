#Version: v0.1


#%% MODULE BEGINS
module_name = 'main'

'''
Version: v0.1

Description:
    main

Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  3/24/25
Date Last Updated:  5/8/25

Doc:
    Project

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
import ML
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

#%% MAIN CODE     
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
data = preprocessing.Preprocess('INPUT/Rainfall.csv')
df = data.load_file()
print("Raw Data Distribution:")
data.distribution(df, "PRESSURE ", "DEWPOINT", "HUMIDITY", "CLOUD ", "SUNSHINE", "         WIND DIRECTION", "WIND SPEED", "MAX TEMP",
              "MIN TEMP", "TEMPERATURE" ,"RAINFALL")
data.plot(df, 'MAX TEMP', 'TEMPERATURE', 'MIN TEMP', 'Celcius')
data.histogram(df, 'RAINFALL', bins=10, title="Raw Rainfall")
print(df)

preprocessed_df = data.celcius_to_farenheit(df)
print("\nAfter temperature conversion: ")
print(preprocessed_df)

preprocessed_df = data.targetReclass(preprocessed_df)
print("\nAfter target reclassification: ")
print(preprocessed_df)
data.histogram(preprocessed_df, 'RAINFALL', bins=10, title="Preprocessed Rainfall")

data.plot(preprocessed_df, 'MAX TEMP', 'TEMPERATURE', 'MIN TEMP', 'Farenheit')
data.heatmap(preprocessed_df)

feature_df = preprocessed_df.drop(['DAY', 'MAX TEMP', 'MIN TEMP'], axis=1, inplace=True)
feature_df = preprocessed_df
feature_df.to_excel('OUTPUT/data.xlsx', index=False)
feature_df.to_csv('OUTPUT/data.csv', index=False)
print(feature_df)

preprocessed_df.columns = preprocessed_df.columns.str.strip()


X_train, X_test, y_train, y_test = data.split(feature_df, feature_col=["PRESSURE", "DEWPOINT", "HUMIDITY", "CLOUD", "SUNSHINE", 
                                                                       "WIND DIRECTION", "WIND SPEED", "TEMPERATURE"], target=["RAINFALL"])

score=ML.kNN(x=X_train, y=y_train, xTest=X_test, yTest=y_test, n=30)
print("KNN Score: ", score)

data.ANN(xTr=X_train, xTst=X_test, yTr=y_train, yTst=y_test, hl=[3,3], ep=5, bs=32)
'''
data = ML.ML('OUTPUT/data.csv')
df = data.load_file()
X_train, y_train, X_val, X_test, y_val, y_test = data.split_data(df, (0.6, 0.2, 0.2), 11)
    
DT_y_test, DT_y_pred = ML.ML.DT(X_train, y_train, X_val, X_test, y_val, y_test)
SVM_y_test, SVM_y_pred = ML.ML.SVM(X_train, y_train, X_val, X_test, y_val, y_test)
KNN_y_test, KNN_y_pred = ML.ML.KNN(X_train, y_train, X_val, X_test, y_val, y_test)
ANN_Model, ANN_History = ML.ML.ANN(X_train, X_test, X_val, y_train, y_test, y_val, hl=[3,3], ep=10, bs=32)


ML.ML.performance_measures(DT_y_test, DT_y_pred, "DT")
ML.ML.performance_measures(SVM_y_test, SVM_y_pred, "SVM CV")
ML.ML.performance_measures(KNN_y_test, KNN_y_pred, "KNN")
ML.ML.EpochCurve(ANN_Model, ANN_History, "ANN Error")




#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()

# %%
