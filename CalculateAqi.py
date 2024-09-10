import pandas as pd
import matplotlib.pyplot as plt


def calculate_aqi(C, C_LO, C_HI, I_LO, I_HI):
    """
    Calcola l'AQI dato un certo livello di concentrazione di inquinante.

    C: Concentrazione dell'inquinante misurata
    C_LO: Limite inferiore della concentrazione per la categoria
    C_HI: Limite superiore della concentrazione per la categoria
    I_LO: Limite inferiore dell'AQI per la categoria
    I_HI: Limite superiore dell'AQI per la categoria
    """
    AQI = ((I_HI - I_LO) / (C_HI - C_LO)) * (C - C_LO) + I_LO
    return round(AQI)


# Limiti per ciascun inquinante
aqi_breakpoints = {
    'PM2.5': [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ],
    'PM10': [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ],
    'O3': [
        (0.0, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
        (0.201, 0.604, 301, 500)
    ],
    'CO': [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500)
    ],
    'SO2': [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500)
    ],
    'NO2': [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500)
    ]
}

def calculate_aqi(concentration, breakpoints):
    """
    Calcola l'AQI per un dato inquinante basato sui suoi breakpoint.

    concentration: Concentrazione dell'inquinante
    breakpoints: Lista di tuple con i breakpoint (C_LO, C_HI, I_LO, I_HI)

    """
    for (C_LO, C_HI, I_LO, I_HI) in breakpoints:
        if C_LO <= concentration <= C_HI:
            return ((I_HI - I_LO) / (C_HI - C_LO)) * (concentration - C_LO) + I_LO
    return None  # Nel caso in cui la concentrazione sia fuori dai limiti definiti

def categorize_aqi(aqi):
    #Converte un valore AQI in una categoria.

    if aqi is None:
        return 'No Data'  # Gestisci i valori nulli o mancanti
    elif 0 <= aqi <= 50:
        return 'Good'
    elif 51 <= aqi <= 100:
        return 'Moderate'
    elif 101 <= aqi <= 150:
        return 'Poor'
    elif 151 <= aqi <= 200:
        return 'Unhealthy'
    elif 201 <= aqi <= 300:
        return 'Very Unhealthy'
    elif aqi > 300:
        return 'Dangerous'
    else:
        return 'Invalid'  # Gestisci valori fuori dal range


# Carica il dataset
df = pd.read_csv(r'globalAir.csv')

def calculate_And_Save(df):
    # Calcola l'AQI per ciascun inquinante
    for pollutant in aqi_breakpoints.keys():
        if pollutant in df.columns:
            df[f'AQI_{pollutant}'] = df[pollutant].apply(calculate_aqi, args=(aqi_breakpoints[pollutant],))

    # Calcola l'AQI totale prendendo il massimo tra gli AQI calcolati
    aqi_columns = [f'AQI_{pollutant}' for pollutant in aqi_breakpoints.keys() if f'AQI_{pollutant}' in df.columns]
    df['Air_Quality'] = df[aqi_columns].max(axis=1)

    # Categorizza l'AQI totale
    df['Air_Quality_Category'] = df['Air_Quality'].apply(categorize_aqi)

    # Visualizza i risultati
    print(df.head())

    #Salvataggio nel nuovo file
    columns_to_exclude= aqi_columns
    df_filtered= df.drop(columns=columns_to_exclude)
    df_filtered.to_csv('globalAirQuality00.csv')
    return df_filtered

def AQI_Distribution(df):
    # Conta le frequenze di ciascuna categoria
    category_counts = df['Air_Quality_Category'].value_counts()


    # Visualizza le distribuzioni
    print("Distribuzione delle categorie AQI:")
    print(category_counts)


    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribuzione delle Categorie AQI')
    plt.xlabel('Categoria AQI')
    plt.ylabel('Numero di Osservazioni')
    plt.xticks(rotation=45)
    plt.show()