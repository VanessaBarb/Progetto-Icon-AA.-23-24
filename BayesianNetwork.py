import os
import pickle
import pandas as pd
from pgmpy.models import BayesianNetwork,BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore,K2Score,AICScore,BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer,StandardScaler
import matplotlib.pyplot as plt
import networkx as nx


def create_model():
    # Carica i dati
    df = pd.read_csv("globalAirNew.csv")

    # Discretizza le variabili continue
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
    df[numerical_columns] = discretizer.fit_transform(df[numerical_columns])
    df_b=df.copy()
    df_b=df.drop(['City','Date','Country'],axis=1)

    # Apprendimento della struttura della rete
    hc = HillClimbSearch(df_b)
    best_model = hc.estimate(scoring_method='K2score')

    #Creazione del modello di rete bayesiana
    model = BayesianNetwork(best_model.edges())

    print(model.edges())
    # Apprendimento dei parametri
    model.fit(df_b, estimator=MaximumLikelihoodEstimator,n_jobs=-1)
    with open('Bayesian_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model


def load_model():
    file_path=('Bayesian_model.pkl')
    # Controlla sull'esistenza del file
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    else:
        print(f"Errore: Il file '{file_path}' non esiste. Creazione del modello:")
        model=create_model()
        print(f"modello creato.")
        return model

def visualize_model(model):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="purple",
        connectionstyle="angle3,angleA=90,angleB=0",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )
    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()


def bayesian_Infer(pm25,pm10,so2,no2,co,o3,model):
    # Inferenza
    inference = VariableElimination(model)
    result = inference.query(variables=['Air_Quality_Category'], evidence={'PM2.5': pm25,'PM10':pm10,'SO2':so2,'NO2':no2,'CO':co,'O3':o3})
    print(result)

model=load_model()
visualize_model(model)
# Chiamata della funzione `bayesian_Infer`
#bayesian_Infer(pm25, pm10, so2, no2, co, o3, model)


