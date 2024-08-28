from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


def load_data():
    df=pd.read_csv(r"globalAirNew.csv")
    return df

def filter_data(df):
    #Conta il numero di dati etichettati come "Dangerous"
    dangerous_df= df[df['Air_Quality_Category']=='Dangerous']
    dangerous_count= dangerous_df.shape[0]
    print(f"Numero di dati etichettati come 'Dangerous': {dangerous_count}")

    num_to_remove= dangerous_count//2

    #Rimuove la metà dei dati Dangerous
    dangerous_to_remove= dangerous_df.iloc[:num_to_remove]
    df_filtered = df[~df.index.isin(dangerous_to_remove.index)]

    dangerous_count_after_removal= df_filtered[df_filtered['Air_Quality_Category'] == 'Dangerous'].shape[0]
    print(f"Numero di dati etichettati come 'Dangerous' dopo la rimozione: {dangerous_count_after_removal}")

    return df_filtered

def prepare_data(df_filt):
    #Divisione del dataset in feature e target
    X= df_filt[['PM2.5','PM10']]
    y= df_filt['Air_Quality_Category']

    #Divisione del dataset in training e testing
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train,X_test, y_train):
    #Pesi manuali:
    class_weights= {
        'Dangerous':1,
        'Good':100,
        'Moderate':1,
        'Poor':1,
        'Unhealthy':1,
        'Very Unhealthy':1
    }

    rf= RandomForestClassifier(random_state=42, class_weight=class_weights)
    rf.fit(X_train, y_train)
    y_pred= rf.predict(X_test)
    return y_pred, rf



