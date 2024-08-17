import pandas as pd
import numpy as np

df_t=pd.read_csv(r"C:\Users\Stefano\Desktop\ICON\city_temperature.csv")
df_a=pd.read_csv(r"C:\Users\Stefano\Desktop\ICON\air_index.csv")
df_2023=pd.read_csv(r"C:\Users\Stefano\Desktop\ICON\ProgettoIcon\global_air_quality.csv")








def city_tuning(aqi_data,temperature_data):
     # Usa un set per ottenere le città uniche
    city_set = set(aqi_data['City'])

    # Mantieni solo le righe in temperature_data dove la città è presente in city_set
    temperature_data = temperature_data[temperature_data['City'].isin(city_set)]

    # Salva il DataFrame filtrato in un nuovo file CSV
    temperature_data.to_csv(r"city_temperature_fined1.csv", index=False)

    print("done")

    return temperature_data



def yearly_temperature_avg(temperature_dataset):
    monthly_mean_temp = temperature_dataset.groupby(['Region', 'Country', 'City', 'Year', 'Month'])['AvgTemperature'].mean().reset_index()

    # Ora calcola la media annua per ogni gruppo
    annual_mean_temp_per_city = monthly_mean_temp.groupby(['Region', 'Country' ,'City', 'Year'])['AvgTemperature'].mean().reset_index()

    annual_mean_temp_per_city.sort_values(by=['Country','City', 'Year'])
    # Salva il nuovo DataFrame in un file CSV
    output_file_path = r"city_temperature_yearly.csv"
    annual_mean_temp_per_city.to_csv(output_file_path, index=False)

    return annual_mean_temp_per_city



def restructure_temperature(temperature_dataset):
    df_pivot = temperature_dataset.pivot_table(index=['Region', 'Country',  'City'],
                          columns='Year',
                          values='AvgTemperature')

    # Rinomina le colonne per aggiungere il prefisso 'avg_temperature_'
    df_pivot.columns = [f'avg_temperature_{int(col)}' for col in df_pivot.columns]

    # Resetta l'indice per trasformare l'indice in colonne
    df_pivot.reset_index(inplace=True)

    # Salva il DataFrame ristrutturato in un nuovo file CSV
    output_file_path = r"city_temperature_yearly.csv"
    df_pivot.to_csv(output_file_path, index=False)

    return df_pivot

    print("Il dataset ristrutturato è stato creato e salvato con successo.")




def aqi_data_refinement(aqi_data):
     # Elimina colonne specifiche se presenti
    columns_to_drop = ["2022", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "2021"]
    # Droppa solo le colonne che esistono nel DataFrame
    aqi_data.drop(columns=[col for col in columns_to_drop if col in aqi_data.columns], inplace=True)
     # Sostituisci valori vuoti (come "" o " ") con NaN
    aqi_data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

    # Elimina tutte le righe che contengono NaN (che corrisponde ai valori vuoti sostituiti sopra)
    aqi_data.dropna(inplace=True)

    # Salva il DataFrame raffinato in un nuovo file CSV
    aqi_data.to_csv(r"air_index_refined.csv", index=False)
    print("done")

    return aqi_data



def date_restriction(temperature_dataset):
    indices_to_drop = temperature_dataset[temperature_dataset["Year"] < 2017].index

    # Elimina le righe dal DataFrame
    temperature_dataset.drop(indices_to_drop, inplace=True)

    temperature_dataset.to_csv(r"city_temperature_fined.csv", index=False)

    print("done")

    return temperature_dataset




def fuse_datasets(aqi_dataset,temperature_dataset):
    aqi_columns=aqi_dataset[["City","2020","2019","2018","2017"]]
    aqi_columns=aqi_columns.rename(columns={
    "2017" : "avg_aqi_2017",
    "2018" : "avg_aqi_2018",
    "2019" : "avg_aqi_2019",
    "2020" : "avg_aqi_2020"})
    merged_df=pd.merge(temperature_dataset,aqi_columns,on="City",how="left")
    merged_df = merged_df.drop_duplicates(subset='City', keep='first')
    merged_df.to_csv(r"yearly_temperature_aqi.csv", index=False)

    return merged_df


def avg_2023(yearly_dataset,dataset_2023):
   # Calcola la media della temperatura per ogni città nel 2023
    city_avg_temp = dataset_2023.groupby(['City'])['Temperature'].mean().reset_index()
    city_avg_temp.rename(columns={"Temperature": "avg_temperature_2023"}, inplace=True)

    # Calcola la media della qualità dell'aria per ogni città nel 2023
    city_avg_aqi = dataset_2023.groupby([ 'City'])['Air_Quality'].mean().reset_index()
    city_avg_aqi.rename(columns={"Air_Quality": "avg_aqi_2023"}, inplace=True)

    # Unisci le due medie calcolate con il dataset originale
    city_avg = pd.merge(city_avg_temp, city_avg_aqi, on=[ 'City'])
    # Unisci il dataset annuale con il dataset delle medie del 2023
    merged_df = pd.merge( yearly_dataset,city_avg, on=["City"], how="left")

    merged_df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

    # Elimina tutte le righe che contengono NaN (che corrisponde ai valori vuoti sostituiti sopra)
    merged_df.dropna(inplace=True)

    # Salva il risultato in un file CSV
    output_file_path = r"final_yearly_temperature_aqi.csv"
    merged_df.to_csv(output_file_path, index=False)

    return merged_df



df_a=aqi_data_refinement(df_a)
df_t=date_restriction(df_t)
df_t=city_tuning(df_a,df_t)
df_t=yearly_temperature_avg(df_t)
df_t=restructure_temperature((df_t))
df_y=fuse_datasets(df_a,df_t)
avg_2023(df_y,df_2023)


