import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np


def load_data():
    df= pd.read_csv(r"globalAirQualityUp.csv")
    return df

#Bilancia il dataset per rimuovere istanze Unhealthy
def balance_dataset(df):
    unh_class = df[df['Air_Quality_Category'] == 'Unhealthy']
    # Verifica se il dataframe non è vuoto
    if not unh_class.empty:
        # Determina il numero di campioni da estrarre
        n_unh_samples = 4000
        # Riduce il numero di righe della classe Unhealthy
        unh_class_sampled = unh_class.sample(n=n_unh_samples, random_state=42)
    else:
        print("Non ci sono campioni nella classe 'Unhealty'.")
        unh_class_sampled = pd.DataFrame()  # Crea un dataframe vuoto
    # Seleziona le righe delle altre classi
    other_classes = df[df['Air_Quality_Category'] != 'Unhealthy']

    # Combina le classi
    if not unh_class_sampled.empty:
        df_balanced = pd.concat([unh_class_sampled, other_classes])
    else:
        df_balanced = other_classes

    # Verifica la distribuzione delle classi
    print("Distribuzione delle classi dopo il bilanciamento:")
    print(df_balanced['Air_Quality'].value_counts())

    # Salva il nuovo dataset bilanciato
    df_balanced.to_csv("globalAirNew_balanced.csv", index=False)
    print("Dataset bilanciato salvato come 'globalAirNew_balanced.csv'.")
    return df_balanced

def prepare_data(df):
    features= ['PM2.5', 'PM10']
    X= df[features]
    y= df['Air_Quality_Category']

    #Divisione in dati di test e dati di train
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train,X_test, y_train, y_test):

    # Addestramento dell'albero decisionale
    DecTree = DecisionTreeClassifier(max_depth= 10, min_samples_split=20, min_samples_leaf=5, random_state=42)
    DecTree.fit(X_train, y_train)

    return DecTree