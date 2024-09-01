import pandas as pd
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import resample
import numpy as np


#metodo per il ritrovamento e l'eliminazione degli outliers
def manage_outliers():
    #Caricamento del dataset
    df = pd.read_csv(r"globalAirNew.csv")
    df_no_outliers=df.copy()
    #selezione delle features di interesse
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    features=df[numerical_columns]
    #si calcolano i quartili per entrambe le features e si effettua un filtraggio
    for feature in features:
       #plt.figure(figsize=(10, 6))
       #visualizzazione del range di valori tramite boxplot
       #sns.boxplot(x=df[feature])
       #plt.title(f'Box Plot per {feature}')
       #plt.show()

       #calcolo dell'IQR per la feature corrente
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


#scalarizzazione dei valori delle colonne numeriche
def scale_values(df):
    #selezione delle sole colonne contenenti valori float
    numerical_columns = df.select_dtypes(include=['float64','int64']).columns
    scaler = Normalizer()
    #scalarizzazione dei valori nelle colonne numeriche
    df_scaled = scaler.fit_transform(df[numerical_columns])
    df[numerical_columns]=df_scaled
    #assegnazione dei valori scalati al dataset di base
    return df



#metodo per il ritrovamento del valore k ottimale
def find_k(df):
    #Caricamento del dataset
    #df = pd.read_csv(r"globalAirNew.csv")
    numerical_columns = df.select_dtypes(include=['float64','int64']).columns
    inertia = []
    silhouette_scores = []
    K = range(2, 21)
    for k in K:
        kmeans = KMeans(n_clusters=k,n_init=5, init='random')
        kmeans.fit(df[numerical_columns])
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df[numerical_columns], kmeans.labels_))


    plt.figure(figsize=(8, 5))
    # Visualizza il grafico con la nota per il miglior k
    plt.plot(K, inertia, 'bx-')
    #plt.plot(K, inertia, 'bo-', markersize=8)
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
def K_means_clustering_clim(optimal_clusters,df):
    # Encoding della categoria della qualità dell'aria
    label_encoder = LabelEncoder()
    # Selezione delle caratteristiche meteorologiche
    features = ['Temperature','Humidity','Wind Speed']
    X = df[features]

    # Esecuzione del clustering
    kmeans = KMeans(n_clusters=optimal_clusters,n_init=10, init='random')
    kmeans.fit(X)

    df['Cluster'] = kmeans.labels_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Temperature'], df['Humidity'], df['Wind Speed'], c=df['Cluster'], cmap='viridis', alpha=0.85, edgecolor='k')

    ax.set_title('Visualizzazione 3D dei Cluster Basati su Condizioni Meteorologiche')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    ax.set_zlabel('Wind Speed')
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(optimal_clusters))
    cbar.set_label('Cluster')
    plt.show()

    # Calcolo del valore di silhouette per valutare i cluster
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f'Silhouette Score: {silhouette_avg}')

    colors = plt.cm.viridis(np.linspace(0, 1, optimal_clusters))

    # Grafico a torta della distribuzione dei cluster
    cluster_counts = df['Cluster'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in range(optimal_clusters)], autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Distribuzione dei Punti nei Cluster')
    plt.axis('equal')
    plt.show()

     # Grafico della distribuzione della qualità dell'aria per ogni cluster
    plt.figure(figsize=(10, 6))
    for cluster in range(optimal_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        plt.hist(cluster_data['Air_Quality_Category'], bins=len(df['Air_Quality_Category'].unique()), alpha=0.85, color=colors[cluster], label=f'Cluster {cluster}')
        plt.title('Distribuzione della Qualità dell’Aria nei Cluster')
        plt.xlabel('Air Quality Category')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


#Metodo K-means
def K_means_clustering_pol(optimal_clusters,df):
    # Encoding della categoria della qualità dell'aria
    label_encoder = LabelEncoder()
    # Selezione delle caratteristiche meteorologiche
    features = ['PM2.5','PM10','NO2','SO2','CO','O3']
    X = df[features]

    # Esecuzione del clustering
    kmeans = KMeans(n_clusters=optimal_clusters,n_init=10, init='random')
    kmeans.fit(X)

    df['Cluster'] = kmeans.labels_

    # Calcolo del valore di silhouette per valutare i cluster
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f'Silhouette Score: {silhouette_avg}')

    colors = plt.cm.viridis(np.linspace(0, 1, optimal_clusters))

    # Grafico a torta della distribuzione dei cluster
    cluster_counts = df['Cluster'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in range(optimal_clusters)], autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Distribuzione dei Punti nei Cluster')
    plt.axis('equal')
    plt.show()

     # Grafico della distribuzione della qualità dell'aria per ogni cluster
    plt.figure(figsize=(10, 6))
    for cluster in range(optimal_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        plt.hist(cluster_data['Air_Quality_Category'], bins=len(df['Air_Quality_Category'].unique()), alpha=0.85, color=colors[cluster], label=f'Cluster {cluster}')
        plt.title('Distribuzione della Qualità dell’Aria nei Cluster')
        plt.xlabel('Air Quality Category')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()




#dataset con filtraggio degli outliers
df_out=manage_outliers()

df_scaled=scale_values(df_out)

#find_k(df_scaled)

#numero ottimale di clusters trovato tramite il metodo precedente
oc=3
K_means_clustering_clim(oc,df_scaled)
K_means_clustering_pol(oc,df_scaled)