import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE


#Carica il dataset
def load_data():
    df= pd.read_csv(r'globalAirNew.csv')
    return df

#Calcola l'AQI
def calculate_aqi(concentration, limit):
    aqi= (concentration/limit)*100
    return round(aqi)

#Calcola l'aqi complessivo per una riga
def calculate_overall_aqi(row):
    aqi_pm25 = calculate_aqi(row['PM2.5'], pm2_5_limit)
    aqi_pm10 =calculate_aqi(row['PM10'], pm10_limit)
    aqi_no2 =calculate_aqi(row['NO2'], no2_limit)
    aqi_so2 =calculate_aqi(row['SO2'], so2_limit)
    aqi_co =calculate_aqi(row['CO'], co_limit)
    aqi_o3 =calculate_aqi(row['O3'], o3_limit)
    aqi_values= [aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_co, aqi_o3]
    return max(aqi_values) if aqi_values else None

def aqi_to_category(aqi):
    if aqi is None:
        return 'Unknown'
    elif aqi <= 50:
        return 'Good'
    elif aqi >= 51 and aqi <= 100:
        return 'Moderate'
    elif aqi >= 101 and aqi <= 150:
        return 'Poor'
    elif aqi >= 151 and aqi <= 200:
        return 'Unhealthy'
    elif aqi >= 201 and aqi <= 300:
        return 'Very Unhealthy'
    elif aqi > 300:
        return 'Dangerous'

#Limiti per i vari inquinanti
pm2_5_limit = 25
pm10_limit = 50
co_limit = 10
o3_limit = 180
no2_limit = 200
so2_limit = 350



def prepare_data(df):
    #Calcola l'aqi di tutte le righe del dataset
    df['Air_Quality'] = df.apply(calculate_overall_aqi, axis= 1)
    df['Air_Quality_Category']= df['Air_Quality'].apply(aqi_to_category)
    df.to_csv("globalAirNew.csv", index=False)
    return df

def prepare_model_data(df):
    #seleziona le feature e l'etichetta di target
    X= df[['PM2.5', 'PM10', 'NO2','SO2','CO', 'O3']]
    y = df['Air_Quality_Category']
    #Dividi i dati in training e l'etichetta target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    #Normalizza le feature
    scaler = StandardScaler();
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
    #X_test = scaler.transform(X_test)

    # Applica SMOTE al dataset di training
    smote = SMOTE(k_neighbors= 2, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    #Rimuovi valori NaN
    X_train = pd.DataFrame(X_train).dropna().values
    X_test = pd.DataFrame(X_test).dropna().values
    y_train = y_train.dropna()
    y_test = y_test.dropna()

    #creazione e addestramento modello classificatore
    model = RandomForestClassifier(n_estimators= 100, random_state=42, class_weight= 'balanced')
    model.fit(X_train_res, y_train_res)
    print("\nDistribuzione delle classi dopo SMOTE:")
    print(y_train_res.value_counts())


    return model, scaler, X_train_res, y_train_res

# Valuta il modello
def evaluate_model(model,df, X_test, y_test,scaler, X_train_res, y_train_res):
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\nAccuratezza: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.show()



    occurrences = df['Air_Quality_Category'].value_counts()
    # Visualizza i risultati
    print("\nOccorrenze di Air_Quality_Category: ", occurrences)

    # Controlla la distribuzione delle categorie nell'insieme di test
    print("\nDistribuzione delle categorie nel test set:")
    print(y_test.value_counts())

#verifica overfitting
def check_of(model, X_train, y_train, X_test, y_test, X_train_res, y_train_res):
    scaler= StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    y_train_pred= model.predict(X_train_res)
    y_test_pred= model.predict(X_test)


    #Calcolo delle metriche sul training set
    train_accuracy = accuracy_score(y_train_res, y_train_pred)
    train_precision = precision_score(y_train_res, y_train_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train_res, y_train_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train_res, y_train_pred, average='weighted', zero_division=0)


    # Calcolo delle metriche sul test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    # Stampa dei risultati
    print("\n\nMetriche del training set:")
    print(f"Accuratezza: {train_accuracy:.2f}")
    print(f"Precision: {train_precision:.2f}")
    print(f"Recall: {train_recall:.2f}")
    print(f"F1-score: {train_f1:.2f}")

    print("\n\nMetriche del test set:")
    print(f"Accuratezza: {test_accuracy:.2f}")
    print(f"Precision: {test_precision:.2f}")
    print(f"Recall: {test_recall:.2f}")
    print(f"F1-score: {test_f1:.2f}")

    # Eseguire la cross-validation sul training set
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='accuracy')

    # Stampa dei risultati
    print(f"\nAccuratezza Cross-validation: {cv_scores}")
    print(f"\nMedia accuratezza Cross-validation: {cv_scores.mean():.2f}")