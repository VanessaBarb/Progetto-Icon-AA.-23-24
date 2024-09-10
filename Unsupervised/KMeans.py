import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np


#metodo per il ritrovamento e l'eliminazione degli outliers
def manage_outliers():
    #Caricamento del dataset
    df = pd.read_csv(r"Final_globalAirSmote.csv")
    df=df.dropna()
    df_no_outliers=df.copy()
    #selezione delle features di interesse
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Air_Quality','Dispersion_Index']
    #si calcolano i quartili per entrambe le features e si effettua un filtraggio
    for feature in features:
       plt.figure(figsize=(10, 6))
       #visualizzazione del range di valori tramite boxplot
       sns.boxplot(x=df[feature])
       plt.title(f'Box Plot per {feature}')
       plt.show()

       #calcolo dell'IQR per la feature corrente
       Q1 = df[feature].quantile(0.25)
       Q3 = df[feature].quantile(0.75)
       IQR = Q3 - Q1
       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR

       # filtraggio dei dati per rimuovere gli outliers
       df_no_outliers = df_no_outliers[(df_no_outliers[feature] >= lower_bound) &
                                        (df_no_outliers[feature] <= upper_bound)]
    return df_no_outliers


#scalarizzazione dei valori delle colonne numeriche
def scale_values(df):
    #selezione colonne di interesse
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Air_Quality','Dispersion_Index']
    scaler = Normalizer()
    #scalarizzazione dei valori
    df_scaled = scaler.fit_transform(df[features])
    df[features]=df_scaled
    return df



#metodo per il ritrovamento del valore k ottimale
def find_k(df):
    # Encoding della colonna Air_Quality_Category
    label_encoder = LabelEncoder()
    df['Air_Quality_Category_encoded'] = label_encoder.fit_transform(df['Air_Quality_Category'])

    features =  ['Temperature','Humidity','Wind Speed', 'HasRained', 'Is_Stagnant', 'Dispersion_Index']
    inertia = []
    silhouette_scores = []
    #metodo del gomito
    K = range(2, 21)
    for k in K:
        kmeans = KMeans(n_clusters=k,n_init=5, init='random')
        kmeans.fit(df[features])
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df[features], kmeans.labels_))


    plt.figure(figsize=(8, 5))
    # Visualizza il grafico con la nota per il miglior k
    plt.plot(K, inertia, 'bx-')
    #plt.plot(K, inertia, 'bo-', markersize=8)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow')
    plt.show()

    #Visualizzazione del Silhouette score
    plt.plot(K, silhouette_scores, 'bo-', markersize=8)
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score per Numero di Cluster')
    plt.show()



#clusterizzazione K-means
def K_means_clustering(optimal_clusters, df):
    # Selezione delle caratteristiche meteorologiche
    features = ['PM2.5','Temperature','Humidity','Wind Speed', 'HasRained', 'Is_Stagnant', 'Dispersion_Index']
    X = df[features]

    # Esecuzione del clustering
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, init='random')
    kmeans.fit(X)

    df['Cluster'] = kmeans.labels_

    # Calcolo del valore di silhouette per valutare i cluster
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f'Silhouette Score: {silhouette_avg}')

    # Generazione dei colori per ogni cluster
    colors = plt.cm.viridis(np.linspace(0, 1, optimal_clusters))
    cluster_color_map = {i: colors[i] for i in range(optimal_clusters)}

    # Grafico a torta della distribuzione dei cluster
    cluster_counts = df['Cluster'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(cluster_counts,
            labels=[f'Cluster {i}' for i in range(optimal_clusters)],
            autopct='%1.1f%%',
            startangle=140,
            colors=[cluster_color_map[i] for i in range(optimal_clusters)])  # Usiamo il colore mappato
    plt.title('Distribuzione dei Punti nei Cluster')
    plt.axis('equal')
    plt.show()

    # Grafico della distribuzione della qualità dell'aria per ogni cluster (4 istogrammi separati)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Creazione di una griglia 2x2 per 4 cluster
    axes = axes.flatten()  # Appiattimento degli assi per un accesso più semplice

    for cluster in range(min(optimal_clusters, 4)):  # Limito ai primi 4 cluster, se ce ne sono di più
        cluster_data = df[df['Cluster'] == cluster]
        axes[cluster].hist(cluster_data['Air_Quality_Category'],
                           bins=len(df['Air_Quality_Category'].unique()),
                           alpha=0.85,
                           color=cluster_color_map[cluster])  # Colore mappato per il cluster
        axes[cluster].set_title(f'Cluster {cluster}')
        axes[cluster].set_xlabel('Air Quality Category')
        axes[cluster].set_ylabel('Frequency')

    plt.tight_layout()
    plt.suptitle('Distribuzione della Qualità dell’Aria nei Cluster', fontsize=16, y=1.02)
    plt.show()

    from sklearn.metrics import v_measure_score

    # Calcolo della V-measure
    v_measure = v_measure_score(df['Air_Quality_Category'], df['Cluster'])
    print(f'V-Measure: {v_measure}')




#dataset con filtraggio degli outliers
#df_out=manage_outliers()
#df_scaled=scale_values(df_out)
#find_k(df_scaled)
#numero ottimale di clusters trovato tramite il metodo precedente
oc=4
#K_means_clustering(oc,df_scaled)