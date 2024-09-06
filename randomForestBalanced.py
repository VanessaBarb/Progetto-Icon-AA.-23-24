from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np

def load_data():
    df=pd.read_csv(r"globalAirQualityUp.csv")
    return df

def prepare_data(df_filt):
    #Divisione del dataset in feature e target
    X= df_filt[['PM2.5','PM10']]
    y= df_filt['Air_Quality_Category']

     # Applica oversampling e undersampling combinati
    X_res, y_res = prepare_data_with_sampling(X, y)

    # Divisione del dataset in training e test
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train,X_test, y_train):
    #Ottimizzazione degli iperparametri
    best_rf= optimize_hyperparameters(X_train, y_train)


    return best_rf

def prepare_data_with_sampling(X, y):
    # Mostra la distribuzione iniziale delle classi
    print("Distribuzione originale delle classi:", Counter(y))

    # Applica SMOTE per generare nuovi campioni della classe 'Good'
    smote = SMOTE(sampling_strategy={'Good': int(0.7 * Counter(y)['Unhealthy'])}, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    print("Distribuzione dopo SMOTE (Oversampling della classe 'Good'):", Counter(y_smote))

    # Applica RandomUnderSampler per ridurre il numero di campioni della classe 'Unhealthy'
    undersampler = RandomUnderSampler(sampling_strategy={'Unhealthy': int(0.7 * Counter(y_smote)['Unhealthy'])}, random_state=42)
    X_res, y_res = undersampler.fit_resample(X_smote, y_smote)

    print("Distribuzione finale dopo Undersampling della classe 'Unhealthy':", Counter(y_res))

    return X_res, y_res

#Ricerca iperparametri
def optimize_hyperparameters(X_train, y_train):
    # Definisci lo spazio degli iperparametri con distribuzioni casuali
    param_dist = {
        'n_estimators': np.arange(100, 1000, step=100),
        'max_depth': [10, 20, 30, None],
        'max_features': [ 'sqrt', 'log2', 0.3, 0.5],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 5),
        'bootstrap': [True, False]
    }

    # RandomizedSearchCV - Trovati i parametri migliori. Posso commentare il codice
    #random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)

    best_params = {
        'n_estimators': 200,
        'min_samples_split': 4,
        'min_samples_leaf': 1,
        'max_features': 0.3,
        'max_depth': None,
        'bootstrap': True
    }

    #Inizializza modello con gli iperparametri trovati
    rf = RandomForestClassifier(**best_params, random_state= 42, class_weight= "balanced")


    # Esegue la ricerca
    rf.fit(X_train, y_train)


    #print(f"\n\nMigliori iperparametri trovati: {random_search.best_params_}")

    return rf

