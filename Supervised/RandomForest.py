from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


def preprocessing(df):
    # Rimuovi colonne non necessarie
    columns_to_drop = ['Unnamed: 0', 'Year', 'City']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    return df_cleaned

def feature_imp_matrix_corr(df_cleaned):
    # Matrice di correlazione
    numerical_df = df_cleaned.select_dtypes(include=[float, int])
    corr_matrix = numerical_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matrice di correlazione")
    plt.show()

    # Identificazione delle colonne categoriali
    categorical_columns = ['Country']
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_columns)

    # Definizione feature X e target y
    X = df_encoded.drop(columns=['Air_Quality_Category', 'Air_Quality'])
    y = df_encoded['Air_Quality_Category']

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Addestramento del modello con dati originali
    rf = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_depth=8, min_samples_leaf=15, random_state=1, max_features='sqrt')
    rf.fit(X_train, y_train)

    # Importanza delle feature
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Visualizzazione delle feature più importanti
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Crea un DataFrame per ordinare le feature
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    })
    features_df = features_df.sort_values(by='Importance', ascending=False)
    print(features_df)

    # Seleziona le prime 4 feature
    top_4_features = features_df.head(4)['Feature']
    X_top_4 = X[top_4_features]

    return X_top_4, y

def smote_good(X, y):
    # Calcola la media del numero di campioni delle altre classi
    class_counts = y.value_counts()
    average_count = int(class_counts.drop('Good').mean())

    # Crea un DataFrame con i dati originali
    df_balanced = pd.concat([X, y], axis=1)

    # Separazione delle classi
    X_balanced = df_balanced.drop(columns='Air_Quality_Category')
    y_balanced = df_balanced['Air_Quality_Category']

    # Applicazione di SMOTE
    smote = SMOTE(sampling_strategy={'Good': average_count}, random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_balanced, y_balanced)

    # Verifica la distribuzione delle classi dopo SMOTE
    print("Distribuzione delle classi dopo SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

def train_model(X, y):
    # Divisione del dataset bilanciato in training e test set
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X, y, test_size=0.3, random_state=1)

    # Addestramento del modello con i dati bilanciati
    rf_resampled = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_depth=8, min_samples_leaf=15, random_state=1, max_features='sqrt')
    rf_resampled.fit(X_train_resampled, y_train_resampled)
    return rf_resampled, X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled

# Esecuzione del flusso
"""

df_cleaned = preprocessing(df)
X_top, y = feature_imp_matrix_corr(df_cleaned)
X_resampled, y_resampled = smote_good(X_top, y)
rf, X_train, X_test, y_train, y_test = train_model(X_resampled, y_resampled)

# Assicurati che la funzione evaluate_model accetti i parametri corretti
evm.evaluate_model(rf, X_train, X_test,y_train, y_test, "Random Forest")
"""
