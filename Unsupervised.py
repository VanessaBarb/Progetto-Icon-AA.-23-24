import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns



#metodo per il ritrovamento e l'eliminazione degli outliers
def manage_outliers(df):
    df_no_outliers=df.copy()
    #selezione delle features di interesse
    features=df[['PM2.5','Air_Quality']]
    #si calcolano i quartili per entrambe le features e si effettua un filtraggio
    for feature in features:
       plt.figure(figsize=(10, 6))
       #si visualizza il range di valori tramite boxplot
       sns.boxplot(x=df[feature])
       plt.title(f'Box Plot per {feature}')
       plt.show()

       # Calcolo dell'IQR per la feature corrente
       Q1 = df[feature].quantile(0.25)
       Q3 = df[feature].quantile(0.75)
       IQR = Q3 - Q1
       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR

       # filtraggio dei dati per rimuovere gli outliers
       df_no_outliers = df_no_outliers[(df_no_outliers[feature] >= lower_bound) &
                                        (df_no_outliers[feature] <= upper_bound)]
    #si ritorna il dataset filtrato
    return df_no_outliers

#metodo per il ritrovamento del valore k ottimale
def find_k(df):
    inertia = []
    silhouette_scores = []
    K = range(2, 20)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, kmeans.labels_))

    # Visualizzazione del Metodo del Gomito
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bo-', markersize=8)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow')
    plt.show()

    #Visualizzazione del Silhouette score in relazione al valore k
    plt.plot(K, silhouette_scores, 'bo-', markersize=8)
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score per Numero di Cluster')
    plt.show()

#Metodo K-means
def K_means_clustering(optimal_clusters,df):
   features = ['PM2.5','Air_Quality']
   X = df[features]

   kmeans = KMeans(optimal_clusters, random_state=42)
   kmeans.fit(X)

   df['Cluster'] = kmeans.labels_

   plt.figure(figsize=(10,6))

    #Visualizzazione dei risultati della clusterizzazione
   plt.scatter(df['PM2.5'], df['Air_Quality'], c=df['Cluster'], cmap='viridis', alpha=0.85, edgecolor='k')


   plt.title('Visualizzazione dei Cluster')
   plt.xlabel('PM2.5')
   plt.ylabel('AQI ')
   plt.colorbar(label='Cluster')
   plt.legend
   plt.show()

    #stampa del valore di Silhouette per il risultato del clustering
   silhouette_avg = silhouette_score(X, kmeans.labels_)
   print(f'Silhouette Score: {silhouette_avg}')



#Caricamento del dataset
df = pd.read_csv(r"C:\Users\Stefano\Desktop\ICON\ProgettoIcon\globalAirNew.csv")
df=manage_outliers(df)
#selezione unicamente delle colonne numeriche
numerical_columns = df.select_dtypes(include=['float64']).columns
scaler = StandardScaler()
#scalarizzazione dei valori nelle colonne numeriche
df_scaled = scaler.fit_transform(df[numerical_columns])
#assegnazione dei valori scalati al dataset di base
df[numerical_columns]=df_scaled

find_k(df_scaled)
#numero ottimale di clusters trovato tramite il metodo precedente
oc=3
K_means_clustering(oc,df)
