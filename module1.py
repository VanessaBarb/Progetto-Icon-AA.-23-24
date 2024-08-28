import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#Carica il dataset
def load_data():
    df= pd.read_csv(r'globalAir.csv')

def calculate_aqi(concentration, limit):
    aqi= (concentration/limit)*100
    return round(aqi)

#Limiti per i vari inquinanti
pm2_5_limit = 25
pm10_limit = 50
co_limit = 10
o3_limit = 180
no2_limit = 200
so2_limit = 350


def calculate_overall_aqi(row):
    aqi_pm25 = calculate_aqi(row['PM2.5'], pm2_5_limit)
    aqi_pm10 =calculate_aqi(row['PM10'], pm10_limit)
    aqi_no2 =calculate_aqi(row['NO2'], no2_limit)
    aqi_so2 =calculate_aqi(row['SO2'], so2_limit)
    aqi_co =calculate_aqi(row['CO'], co_limit)
    aqi_o3 =calculate_aqi(row['O3'], o3_limit)
    aqi_values= [aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_co, aqi_o3]
    return max(aqi_values) if aqi_values else None


#Calcola l'aqi di tutte le righe del dataset
df['Air_Quality'] = df.apply(calculate_overall_aqi, axis= 1)
df.to_csv("globalAirNew.csv", index= False)
#df = df.append('Air_Quality')


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

df['Air_Quality_Category'] = df['Air_Quality'].apply(aqi_to_category)
df.to_csv("globalAirNew.csv", index= False)
#df = df.append('Air_Quality_Category')

#seleziona le feature e l'etichetta di target
X= df[['PM2.5', 'PM10', 'NO2','SO2','CO', 'O3']]
y = df['Air_Quality_Category']


#Dividi i dati in training e l'etichetta target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)

#Normalizza le feature
scaler = StandardScaler();
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applica SMOTE al dataset di training
smote = SMOTE(k_neighbors= 2, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Verifica la nuova distribuzione delle classi
print("Distribuzione delle classi dopo SMOTE:")
print(y_train_res.value_counts())

print(df.head())

print(y_train.isnull().sum())
y_train= y_train.dropna()
y_train= y_train.fillna(y_train.mode()[0])

#creazione e addestramento modello classificatore
model = RandomForestClassifier(n_estimators= 100, random_state=42, class_weight= 'balanced')
model.fit(X_train_res, y_train_res)

#predizione sui dati di test
y_pred = model.predict(X_test)

#valutaizone del modello
accuracy= accuracy_score(y_test, y_pred)
precision= precision_score(y_test, y_pred, average='macro')
recall= recall_score(y_test, y_pred, average='macro')
f1= f1_score(y_test, y_pred, average= 'macro')
print(f"Precision: {prec_ma: .2f}" )
print(f"F1-score : {f1:.2f}")
print(f"Recall: {recall:.2f}")

prec_mi= precision_score(y_test, y_pred, average='macro')
recall_mi= recall_score(y_test, y_pred, average='macro')
f1_mi= f1_score(y_test, y_pred, average= 'macro')
print(f"Precision micro: {prec_mi: .2f}" )
print(f"F1-score micro: {f1_mi:.2f}")
print(f"Recall micro: {recall_mi:.2f}")



#esempio di predizione su nuovi dati
new_data = pd.DataFrame({
    'PM2.5': [35],
    'PM10': [80],
    'NO2': [50],
    'SO2': [20],
    'CO': [4.0],
    'O3': [60],
})

#normalizza i nuovi dati
new_data = scaler.transform(new_data)


#Predici l'AQI per i nuovi dati
new_predictions= model.predict(new_data)


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


occurrences = df['Air_Quality_Category'].value_counts()

# Visualizza i risultati
print(occurrences)


#VERIFICA OVERFITTING

# Predizioni sul training set
y_train_pred = model.predict(X_train_res)

# Calcolo delle metriche sul training set
train_accuracy = accuracy_score(y_train_res, y_train_pred)
train_precision = precision_score(y_train_res, y_train_pred, average='weighted')
train_recall = recall_score(y_train_res, y_train_pred, average='weighted')
train_f1 = f1_score(y_train_res, y_train_pred, average='weighted')

# Predizioni sul test set
y_test_pred = model.predict(X_test)

# Calcolo delle metriche sul test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Stampa dei risultati
print("Training Set Metrics:")
print(f"Accuracy: {train_accuracy:.2f}")
print(f"Precision: {train_precision:.2f}")
print(f"Recall: {train_recall:.2f}")
print(f"F1-score: {train_f1:.2f}")

print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1-score: {test_f1:.2f}")

# Eseguire la cross-validation sul training set
cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='accuracy')

# Stampa dei risultati
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
