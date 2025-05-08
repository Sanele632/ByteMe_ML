#Version: v0.1
#Date Last Updated: 5/7/25


#%% MODULE BEGINS
module_name = 'DT_SVM'

'''
Version: v0.1

Description:
    ANN, DT, SVM, KNN algorithms 
Authors:
    Sanele Harmon, Taeden Kitchens

Date Created     :  4/19/25
Date Last Updated:  5/7/25

Doc:
    Project

Notes:
    implements SVM, DT, KNN, and ANN
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
from sklearn.model_selection import train_test_split
import numpy  as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
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
    def split_data(self, pInput, pRatio, random):
        pInput = pInput.dropna()
        pInput = pInput.drop(['SAMPLE ID'], axis='columns')
        X = pInput.drop(['RAINFALL'], axis = 'columns')
        y = pInput['RAINFALL']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y,  test_size= pRatio[1] + pRatio[2], random_state = random)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= 0.5, random_state = random)

        return X_train, y_train, X_val, X_test, y_val, y_test
    
    def DT(X_train, y_train, X_val, X_test, y_val, y_test):
        clf = DecisionTreeClassifier(random_state=42)
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas[:-1]  # Remove the largest alpha that prunes everything

        # Step 2: Train trees for each alpha and evaluate with cross-validation
        clfs = []
        cv_scores = []

        for alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
            scores = cross_val_score(clf, X_train, y_train, cv=5)  # 5-fold cross-validation
            clfs.append(clf)
            cv_scores.append(np.mean(scores))

        # Step 3: Plot alpha vs. cross-validated accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(ccp_alphas, cv_scores, marker='o', drawstyle="steps-post")
        plt.xlabel("ccp_alpha")
        plt.ylabel("Mean CV Accuracy")
        plt.title("Optimal ccp_alpha via Cross-Validation")
        plt.grid(True)
        plt.savefig("OUTPUT/Best ccp_alpha.png")
        plt.close()

        # Step 4: Train final model with best alpha
        best_alpha = ccp_alphas[np.argmax(cv_scores)]
        final_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
        final_tree.fit(X_train, y_train)
        final_tree.predict(X_val)
        y_pred = final_tree.predict(X_test)

        print(f"Best ccp_alpha: {best_alpha}")
        
        class_labels = [str(cls) for cls in final_tree.classes_]
        plt.figure(figsize=(12, 8))
        plot_tree(final_tree, filled=True, feature_names=X_train.columns.tolist(), class_names=class_labels)
        plt.title("Decision Tree Visualization")
        plt.savefig("OUTPUT/Decision Tree.png")
        plt.close()
        return y_test, y_pred
    
    def SVM(X_train, y_train, X_val, X_test, y_val, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        param_grid = [{
            'C': [0.5, 1, 10, 100],
            'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }]

        optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)
        optimal_params.fit(X_train_scaled, y_train)

        print(optimal_params.best_params_)

        SVM = SVC(random_state = 42, C=10, gamma = 0.01, kernel = 'rbf')
        SVM.fit(X_train_scaled, y_train)
        SVM.predict(X_val_scaled)
        y_pred = SVM.predict(X_test_scaled)
         
        return y_test, y_pred
    
    def KNN(X_train, y_train, X_val, X_test, y_val, y_test):
        accuracy_scores = []
        k = 1
        
        for k in range(1, len(X_val) + 1):
            KNN = KNeighborsClassifier(n_neighbors=k)
            KNN.fit(X_train, y_train)
            y_pred = KNN.predict(X_val)
            accuracy_scores.append(accuracy_score(y_val, y_pred))
            
        max_accuracy = max(accuracy_scores)
        best_k = accuracy_scores.index(max_accuracy) + 1
        knn = KNeighborsClassifier(n_neighbors= best_k)
        knn.fit(X_train, y_train)
        y_pred1 = knn.predict(X_test)
        return y_test, y_pred1
    
    def ANN(xTr, xTst, xVal, yTr, yTst, yVal, hl, ep, bs):
        scaler = StandardScaler()
        xTr_scaled = scaler.fit_transform(xTr)
        xVal_scaled = scaler.fit_transform(xVal)
        xTst_scaled = scaler.fit_transform(xTst)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(xTr_scaled.shape[1],)))
        for units in hl:
            model.add(tf.keras.layers.Dense(units, activation = 'relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(xTr_scaled, yTr, epochs=ep, batch_size=bs, validation_data=(xVal_scaled, yVal))

        loss, accuracy = model.evaluate(xTst_scaled, yTst)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        return model, history
    
    @staticmethod  
    def performance_measures(y_test, y_pred, Algorithm):
        test_cm = confusion_matrix(y_test, y_pred)
        print(f'{Algorithm} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        TP = test_cm[0, 0]  
        FP = test_cm[0, 1]  
        TN = test_cm[1, 1]  
        FN = test_cm[1, 0]

        precision = TP / (TP + FP) 
        sensitivity = TP / (TP + FN) 
        specificity = TN / (TN + FP)
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

        data = [
        {'Metric': 'Precision', 'Score': precision},
        {'Metric': 'Sensitivity', 'Score': sensitivity},
        {'Metric': 'Specificity', 'Score': specificity},
        {'Metric': 'F1', 'Score': F1},
        ]   
    
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Score', palette='Set2', data=df)
        plt.xlabel("Measure")
        plt.ylabel("Score")
        plt.title("Performance Measures")

        plt.tight_layout()
        plt.savefig(f"OUTPUT/{Algorithm} Performance Measures.png")
        plt.close()

    def EpochCurve(model, history, Algorithm):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs. Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs. Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"OUTPUT/{Algorithm} Epoch Error Curve.png")
        plt.close()    
            
        
    



#Function definitions Start Here
def main():
    data = ML('OUTPUT/data.csv')
    df = data.load_file()
    X_train, y_train, X_val, X_test, y_val, y_test = data.split_data(df, (0.6, 0.2, 0.2), 11)
    
    DT_y_test, DT_y_pred = ML.DT(X_train, y_train, X_val, X_test, y_val, y_test)
    SVM_y_test, SVM_y_pred = ML.SVM(X_train, y_train, X_val, X_test, y_val, y_test)
    
    ML.performance_measures(DT_y_test, DT_y_pred, "DT")
    ML.performance_measures(SVM_y_test, SVM_y_pred, "SVM CV")
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here



#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()