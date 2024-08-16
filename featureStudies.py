import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"yearly_temperature_aqi.csv")

column_of_interest = [
    'avg_temperature_2017', 'avg_temperature_2018',
    'avg_temperature_2019', 'avg_temperature_2020',
    'avg_aqi_2017', 'avg_aqi_2018', 'avg_aqi_2019', 'avg_aqi_2020'
]

#estrai le colonne numeriche dal dataset
numeric_data = df[column_of_interest]

#calcola la matrice di correlazione
correlation_matrix = numeric_data.corr()

#visualizza la matrice con una heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot= True, cmap='coolwarm', fmt= '.2f', linewidths= 0.5)
plt.title('Correlation Matrix')
plt.show()

#Aggrega i dati per regione calcolando la media delle temperature e dell'Aqi per ciascun anno
df_region_temp = df.groupby('Region')[['avg_temperature_2017','avg_temperature_2018', 'avg_temperature_2019', 'avg_temperature_2020']].mean().reset_index()
df_region_aqi = df.groupby('Region')[['avg_aqi_2017','avg_aqi_2018','avg_aqi_2019', 'avg_aqi_2020' ]].mean().reset_index()

#Riordina i dati per creare un dataframe adatto al plotting
df_melted_temp= pd.melt(df_region_temp, id_vars=['Region'], value_vars=['avg_temperature_2017', 'avg_temperature_2018', 'avg_temperature_2019', 'avg_temperature_2020'],
var_name='Year', value_name='Average Temperature')

df_melted_aqi= pd.melt(df_region_aqi, id_vars=['Region'], value_vars= ['avg_aqi_2017', 'avg_aqi_2018', 'avg_aqi_2019', 'avg_aqi_2020'],
var_name= 'Year', value_name= 'Average AQI')

#rimuove il prefisso 'avg_temperature' o 'avg_aqi' dai nomi delle colonne
df_melted_temp['Year']= df_melted_temp['Year'].str.replace('avg_temperature_', '')
df_melted_aqi['Year']= df_melted_aqi['Year'].str.replace('avg_aqi_','')

# Grafico a linee per le temperature medie per regione
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_melted_temp, x='Year', y='Average Temperature', hue='Region')
plt.title('Trend delle Temperature Medie per Stato (2017-2020)')
plt.xticks(rotation=45)
plt.ylabel('Average Temperature (°C)')
plt.show()

# Grafico a linee per l'AQI medio per regione
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_melted_aqi, x='Year', y='Average AQI', hue='Region')
plt.title('Trend dell\'AQI Medio per Regione (2017-2020)')
plt.xticks(rotation=45)
plt.ylabel('Average AQI')
plt.show()



#Clusterizzazione con K-MEANS
# Seleziona le variabili di interesse
features = ['avg_temperature_2017', 'avg_temperature_2018',
            'avg_temperature_2019', 'avg_temperature_2020',
            'avg_aqi_2017', 'avg_aqi_2018',
            'avg_aqi_2019', 'avg_aqi_2020']

# Calcola la media delle temperature e dell'AQI per ciascuna città
df['avg_temperature'] = df[features[0:4]].mean(axis=1)
df['avg_aqi'] = df[features[4:]].mean(axis=1)

# Prepara i dati per la clusterizzazione
data = df[['avg_temperature', 'avg_aqi']]

# Normalizza i dati
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Determina il numero ottimale di cluster con il metodo del gomito
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(data_normalized)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Metodo del Gomito per K-means')
plt.xlabel('Numero di Cluster')
plt.ylabel('Inertia')
plt.show()

# Applica K-means con il numero ottimale di cluster (es. 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_normalized)

# Aggiungi i cluster al DataFrame
df['Cluster'] = clusters

# Visualizza i cluster
plt.figure(figsize=(10, 7))
sns.scatterplot(x='avg_temperature', y='avg_aqi', hue='Cluster', palette='viridis', data=df)
plt.title('Clusterizzazione delle Città')
plt.xlabel('Temperatura Media')
plt.ylabel('AQI Medio')
plt.legend(title='Cluster')
plt.show()

# Valuta la qualità del clustering
silhouette_avg = silhouette_score(data_normalized, clusters)
print(f"Indice di Silhouette: {silhouette_avg:.2f}")
