
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Carica il dataset
df= pd.read_csv(r'C:\Users\barba\Downloads\archive (1)\global_air_quality_data_10000.csv')


def calculate_aqi(concentration, limit):
    aqi = ( concentration / limit ) * 100
    return round(aqi)

pm2_5_limit = 25
pm10_limit = 50
co_limit = 10
o3_limit = 180
no2_limit = 200
so2_limit = 350


def calculate_overall_aqi(row):
    aqi_pm25 = calculate_aqi(row['PM2.5'], pm2_5_limit)
    aqi_pm10 =calculate_aqi(row['PM10'], pm10_limit)
    aqi_no2 =calculate_aqi(row['NO2'], co_limit)
    aqi_so2 =calculate_aqi(row['SO2'], o3_limit)
    aqi_co =calculate_aqi(row['CO'], no2_limit)
    aqi_o3 =calculate_aqi(row['O3'], so2_limit)
    aqi_values= [aqi for aqi in [aqi_pm25, aqi_pm10, aqi_so2, aqi_co, aqi_o3 ] if aqi is not None]
    return max(aqi_values) if aqi_values else None

#Calcola l'aqi di tutte le righe del dataset
df['Air_Quality'] = df.apply(calculate_overall_aqi, axis= 1)
df.to_csv("global_air_quality.csv", index= False)
#df = df.append('Air_Quality')

def aqi_to_category(aqi):
    if aqi is None:
        return 'Unknown'
    elif aqi <= 30:
        return 'Good'
    elif aqi <= 66 and aqi > 30:
        return 'Moderate'
    elif aqi <= 99 and aqi > 66:
        return 'Poor'
    elif aqi <= 150 and aqi > 99:
        return 'Unhealthy'
    elif aqi > 150:
        return 'Dangerous'


df['Air_Quality_Category'] = df['Air_Quality'].apply(aqi_to_category)
df.to_csv("global_air_quality.csv", index= False)
#df = df.append('Air_Quality_Category')

#Rimozione righe con valori nulli e di cons. rimuovere i label
#codifica le categorie come numeri per il modello
#Scrittura su file della colonna Air_Quality
df['Air_Quality_Label']= df['Air_Quality_Category'].astype('category').cat.codes


# Controlla valori mancanti
print(df[['Air_Quality_Label']].isna().sum())

# Rimuovi righe con valori mancanti
df = df.dropna(subset=['Air_Quality_Label'])


#seleziona le feature e l'etichetta di target
X= df[['PM2.5', 'PM10', 'NO2','SO2','CO', 'O3', 'Temperature','Humidity', 'Wind Speed']]
y = df['Air_Quality_Label']

#Dividi i dati in training e l'etichetta target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)

#Normalizza le feature
scaler = StandardScaler();
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(df.head())


#creazione e addestramento modello classificatore
model = RandomForestClassifier(n_estimators= 100, random_state=42, class_weight= 'balanced')
model.fit(X_train, y_train)

#predizione sui dati di test
y_pred = model.predict(X_test)

#valutaizone del modello
print("Accuratezza del modello: ", accuracy_score(y_test, y_pred))
print("Rapporto di classificazione:\n", classification_report(y_test, y_pred))

#esempio di predizione su nuovi dati
new_data = pd.DataFrame({
    'PM2.5': [35],
    'PM10': [80],
    'NO2': [50],
    'SO2': [20],
    'CO': [4.0],
    'O3': [60],
    'Temperature': [25],
    'Humidity': [60],
    'Wind Speed': [10]
})

#normalizza i nuovi dati
new_data = scaler.transform(new_data)


#Predici l'AQI per i nuovi dati
new_predictions= model.predict(new_data)
# Converti le predizioni numeriche in categorie
reverse_mapping = {i: category for i, category in enumerate(df['Air_Quality_Category'].astype('category').cat.categories)}
new_predictions_categories = [reverse_mapping[pred] for pred in new_predictions]
print("Mappatura inversa delle etichette:", reverse_mapping)
print("Predizione dell'AQI per i nuovi dati:", new_predictions_categories)


# Verifica la dimensione dei dataset di addestramento e test
print("Dimensione del training set:", len(X_train))
print("Dimensione del test set:", len(X_test))

# Controlla la distribuzione delle categorie nell'insieme di test
print("Distribuzione delle categorie nel test set:")
print(y_test.value_counts())


#matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice di confusione:\n", conf_matrix)


# Controlla se ci sono valori NaN nel dataset
print("Valori NaN nel dataset:")
print(df.isna().sum())


#cross validation
scores = cross_val_score(model, X, y, cv=5)
print("Accuratezza con cross-validation:", scores.mean())

#Filtrare le colonne numeriche (solo int e float)
df_numeric = df.select_dtypes(include=['float64', 'int64'])

#utilizzo del metodo di Pearson
corr_matrix = df_numeric.corr()

#Visualizzazione della matrice di correlazione con una heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot= True, cmap= 'coolwarm', fmt= '.2f', vmin =- 1, vmax= 1)
plt.title('Heatmap della matrice di correlazione')
plt.show()

#Scatter plot per visualizzare la relazione tra due variabili specifiche
sns.pairplot(df)
plt.show()