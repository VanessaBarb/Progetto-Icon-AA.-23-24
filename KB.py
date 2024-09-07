from pyswip import Prolog
import pandas as pd
import csv


def create_kb(dataset):
    df = pd.read_csv(dataset)

    # Dizionari per raccogliere fatti separati per predicato
    geography_facts = []
    pollutants_facts = []
    climate_facts = []
    monthly_averages_facts = []
    AQI_facts = []

    #dizionari per gestire le eventuali ripetizioni
    geo_dict = []
    avg_dict = {}

    for _, row in df.iterrows():
        #controlli per evitare la ripetizione del fatto
        if row['City'] not in geo_dict:
            geo_dict.append(row['City'])
            geography_facts.append(f"geography('{row['City']}', '{row['Country']}').")

        if row['City'] not in avg_dict:
            avg_dict[row['City']] = set()

        if row['Month'] not in avg_dict[row['City']]:
            avg_dict[row['City']].add(row['Month'])
            monthly_averages_facts.append(
                f"monthly_averages('{row['City']}', {int(row['Month'])}, "
                f"{float(row['Monthly_Avg_Temperature'])}, {float(row['Monthly_Avg_Wind_Speed'])})."
            )


        pollutants_facts.append(
            f"pollutants('{row['City']}', {int(row['Month'])}, {int(row['Day'])}, "
            f"{float(row['PM2.5'])}, {float(row['PM10'])}, {float(row['NO2'])}, "
            f"{float(row['SO2'])}, {float(row['O3'])}, {float(row['CO'])})."
        )


        climate_facts.append(
            f"climate_factors('{row['City']}', {int(row['Month'])}, {int(row['Day'])}, "
            f"{float(row['Temperature'])}, {float(row['Humidity'])}, {float(row['Wind Speed'])})."
        )

        AQI_facts.append(
            f"aqi('{row['City']}', {int(row['Month'])}, {int(row['Day'])}, "
            f"{row['Air_Quality']}, '{row['Air_Quality_Category']}')."
        )

    with open('knowledge_base.pl', 'w') as file:
        # Scrivi tutti i fatti raggruppati per predicato
        file.write("\n% Fatti geografici\n")
        file.write("\n".join(geography_facts) + "\n")

        file.write("\n% Fatti sugli inquinanti\n")
        file.write("\n".join(pollutants_facts) + "\n")

        file.write("\n% Fatti sui fattori climatici\n")
        file.write("\n".join(climate_facts) + "\n")

        file.write("\n% Fatti sulle medie mensili\n")
        file.write("\n".join(monthly_averages_facts) + "\n")

        file.write("\n% Fatti sulla qualità dell'aria \n")
        file.write("\n".join(AQI_facts) + "\n")

    print("KB creata")

#metodo per la definizione delle regole della KB
def rule_definition():
    with open('knowledge_base.pl', 'a') as file:
        rule = """
            %restituzioni delle misurazioni per una città in un giorno
            get_measurements(City, Month, Day, PM2_5, PM10, NO2, SO2, O3, CO, Temperature, Humidity, Wind_Speed, AQI_value, AQI_category) :-
            pollutants(City, Month, Day, PM2_5, PM10, NO2, SO2, O3, CO),
            climate_factors(City, Month, Day, Temperature, Humidity, Wind_Speed),
            aqi(City, Month, Day, AQI_value, AQI_category).
            
            
            % Trova la città con l AQI_value più alto
            highest_aqi(City, Month, Day, Max_AQI) :-
            findall(AQI_value, aqi(_, _, _, AQI_value, _), AQI_list),  
            max_list(AQI_list, Max_AQI),                               
            aqi(City, Month, Day, Max_AQI, _).   
        
        
            %verifica se ha piovuto
            rainy_day(City, Month, Day,HasRained):-
            climate_factors(City, Month, Day, Temperature, Humidity, Wind_Speed),
            monthly_averages(City, Month, Average_Temperature,Average_Wind_Speed),
            (   Temperature < Average_Temperature,
            Wind_Speed > Average_Wind_Speed,
            Humidity > 90
        ->  HasRained = true
        ;   HasRained = false
        ).
        
        
         %verifica della stagnazione dell aria
            stagnant_air(City, Month, Day,Is_Stagnant):-
            climate_factors(City, Month, Day, Temperature, Humidity, Wind_Speed),
            (   Temperature > 28,
            Wind_Speed < 3.5,
            Humidity > 70
        ->  Is_Stagnant = true
        ;   Is_Stagnant = false
        ).
        
        
        % Calcolo dell indice di dispersione 
        pollutant_dispersion(City, Month, Day, Dispersion_Index) :-
        climate_factors(City, Month, Day, _, Humidity, Wind_Speed),
        Dispersion_Index is Wind_Speed * (1 / (1 + (Humidity / 100))).
        """

        file.write(rule)
        print("Create regole.")


#metodo per la creazione del dataset definitivo
def create_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    prolog = Prolog()
    prolog.consult("knowledge_base.pl")


    for index, row in df.iterrows():
        #informazioni principali su cui eseguire le query
        city = row['City']
        month = row['Month']
        day = row['Day']
        pm2_5 = row['PM2.5']
        # Esegui le query Prolog per le nuove feature
        try:
            rainy_day_query = f"rainy_day('{city}', {month}, {day}, HasRained)."
            rainy_day_result = list(prolog.query(rainy_day_query))
            has_rained = rainy_day_result[0]['HasRained']

            stagnant_air_query = f"stagnant_air('{city}', {month}, {day}, Is_Stagnant)."
            stagnant_air_result = list(prolog.query(stagnant_air_query))
            is_stagnant = stagnant_air_result[0]['Is_Stagnant']

            dispersion_query = f"pollutant_dispersion('{city}', {month}, {day}, Dispersion_Index)."
            dispersion_result = list(prolog.query(dispersion_query))
            dispersion_index = round(dispersion_result[0]['Dispersion_Index'], 2)

            # Aggiungi le feature mancanti al DataFrame
            df.at[index, 'HasRained'] = has_rained
            df.at[index, 'Is_Stagnant'] = is_stagnant
            df.at[index, 'Dispersion_Index'] = dispersion_index

        except Exception as e:
            print(f"Errore durante la query per {city}, {month}, {day}: {e}")

    df.to_csv(output_file, index=False)

    print(f"Dataset creato")

create_csv(r"globalAir01.csv","Final_globalAir.csv")