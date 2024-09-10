import os
import pickle
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2Score
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import networkx as nx

#metodo per la creazione del dicretizzatore dei dati
def create_discretizer():
    df = pd.read_csv("Final_globalAir.csv")
    numerical_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Dispersion_Index']
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='quantile')
    discretizer.fit(df[numerical_columns])
    with open('discretizer.pkl', 'wb') as file:
        pickle.dump(discretizer, file)
    print("Discretizzatore creato e salvato.")
    return discretizer

#metodo per la creaszione del modello
def create_model():
    df = pd.read_csv("Final_globalAir.csv")
    #eliminazioni delle feature superflue
    df.drop(['Air_Quality','City','Country','Year','Month','Monthly_Avg_Temperature',
             'Monthly_Avg_Wind_Speed'], axis=1, inplace=True)
    #apertura del discretizzatore
    with open('discretizer.pkl', 'rb') as file:
        discretizer = pickle.load(file)
    relevant_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Dispersion_Index']
    df[relevant_columns] = discretizer.transform(df[relevant_columns])
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=K2Score(df), max_indegree=14)
    #creazione della rete bayesiana
    model = BayesianNetwork(best_model.edges())
    print(f"Archi trovati: {model.edges()}")
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    with open('Bayesian_model.pkl', 'wb') as file:
        pickle.dump({'model': model}, file)
    return model

#caricamento della rete bayesiana
def load_model():
    file_path = 'Bayesian_model.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data['model']
    else:
        print(f"Errore: Il file '{file_path}' non esiste. Creazione del modello:")
        return create_model()

#visualizzazione della rete bayesiana
def visualize_model(model):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, iterations=100, k=2)
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=7, edge_color="purple")
    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()

#stampa i valori di cpd della feature indicata
def print_cpds(model):
    cpd=model.get_cpds('Air_Quality_Category')
    print(f"CPD per il nodo {cpd.variable}:")
    print(cpd)
    print("\nVariabili coinvolte:")
    print(cpd.variables)

    print("\nValori del CPD (matrice delle probabilità):")
    print(cpd.values)
    print("\n")


#inferenza
def bayesian_Infer(pm25, pm10, so2, no2, co, o3, temp, hum, wind, rain, stag, dis_index, model):
    with open('discretizer.pkl', 'rb') as file:
        discretizer = pickle.load(file)
    evidence = {
        'PM2.5': pm25,
        'PM10': pm10,
        'SO2': so2,
        'NO2': no2,
        'CO': co,
        'O3': o3,
        'Temperature': temp,
        'Humidity': hum,
        'Wind Speed': wind,
        'HasRained': rain,
        'Is_Stagnant': stag,
        'Dispersion_Index': dis_index
    }
    #discretizzazione dei valori su cui fare inferenza
    evidence_df = pd.DataFrame([evidence])
    numerical_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'Dispersion_Index']
    evidence_df[numerical_columns] = discretizer.transform(evidence_df[numerical_columns])
    evidence_df['HasRained'] = evidence_df['HasRained'].astype(int)
    evidence_df['Is_Stagnant'] = evidence_df['Is_Stagnant'].astype(int)
    discretized_evidence = evidence_df.iloc[0].to_dict()
    inference = VariableElimination(model)
    result = inference.query(variables=['Air_Quality_Category'], evidence=discretized_evidence)
    print(result)




#create_and_save_discretizer()
#model = load_model()
#print_cpds(model)
#visualize_model(model)
#bayesian_Infer(9.73,154.34,45.91,34.14,1.62,152.4,37.25,64.83,4.7,False,False,2.85, model)
