import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np




def prepare_data(df):
    features= ['PM2.5', 'PM10']
    X= df[features]
    y= df['Air_Quality_Category']

    #Divisione in dati di test e dati di train
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test):
    DecTree= DecisionTreeClassifier(random_state=42)


    DecTree.fit(X_train, y_train)

    y_pred= DecTree.predict(X_test)
    return DecTree,y_pred


