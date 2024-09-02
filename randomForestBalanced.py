from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import numpy as np

def load_data():
    df=pd.read_csv(r"globalAirQualityUp.csv")
    return df

def filter_data(df):
    #Conta il numero di dati etichettati come "Unhealthy"
    unhealthy_df= df[df['Air_Quality_Category']=='Unhealthy']
    unhealthy_count= unhealthy_df.shape[0]
    print(f"Numero di dati etichettati come 'Unhealthy': {unhealthy_count}")

    num_to_remove= unhealthy_count//2

    #Rimuove la metà dei dati Unhealthy
    unhealthy_to_remove= unhealthy_df.iloc[:num_to_remove]
    df_filtered = df[~df.index.isin(unhealthy_to_remove.index)]

    unhealthy_count_after_removal= df_filtered[df_filtered['Air_Quality_Category'] == 'Unhealthy'].shape[0]
    print(f"Numero di dati etichettati come 'Unhealthy' dopo la rimozione: {unhealthy_count_after_removal}")

    return df_filtered

def prepare_data(df_filt):
    #Divisione del dataset in feature e target
    X= df_filt[['PM2.5','PM10']]
    y= df_filt['Air_Quality_Category']

    #Divisione del dataset in training e testing
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train,X_test, y_train):
    #Ottimizzazione degli iperparametri
    best_rf= optimize_hyperparameters(X_train, y_train)


    return best_rf


#Ricerca iperparametri
def optimize_hyperparameters(X_train, y_train):
    # Definisci lo spazio degli iperparametri con distribuzioni casuali
    param_dist = {
        'n_estimators': np.arange(100, 1000, step=100),
        'max_depth': [10, 20, 30, None],
        'max_features': [ 'sqrt', 'log2', 0.5, 0.7],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 5),
        'bootstrap': [True, False]
    }

    # Inizializza il Random Forest
    rf = RandomForestClassifier(random_state=42)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

    # Esegue la ricerca
    random_search.fit(X_train, y_train)

    print(f"\n\nMigliori iperparametri trovati: {random_search.best_params_}")

    return random_search.best_estimator_

